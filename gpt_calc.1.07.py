import math
import threading
import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import ttk, messagebox
from datetime import datetime, date
from typing import Dict, Tuple, Optional, List, Set

import pandas as pd
import pytz
import yfinance as yf


# =============================== Utilities ===============================

PACIFIC = pytz.timezone("America/Los_Angeles")


def _now_pacific_date() -> date:
    return datetime.now(PACIFIC).date()


def _first_float(val) -> Optional[float]:
    try:
        x = float(val)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _safe_mid(row: pd.Series) -> Optional[float]:
    """Return mid=(bid+ask)/2 if both present; else lastPrice; else None."""
    try:
        bid = float(row.get("bid", float("nan")))
        ask = float(row.get("ask", float("nan")))
        last = float(row.get("lastPrice", float("nan")))
    except Exception:
        return None

    if not math.isnan(bid) and not math.isnan(ask):
        return 0.5 * (bid + ask)
    if not math.isnan(last):
        return last
    return None


# ============================ Manual Calculator ===========================

class ManualCalculator(ttk.Frame):
    """Single-ticker forward-vol + FF calculator (manual pair)."""

    def __init__(self, parent):
        super().__init__(parent, padding=16)
        self.tz_local = PACIFIC
        self.expiries: List[str] = []
        self.spot: Optional[float] = None
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        ttk.Label(self, text="Manual Mode — Forward Volatility Calculator",
                  font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 10))

        src = ttk.LabelFrame(self, text="Market Source", padding=10)
        src.grid(row=1, column=0, columnspan=8, sticky="ew")
        for c in range(8):
            src.columnconfigure(c, weight=1)

        self.ticker_var = tk.StringVar(value="SPY")
        ttk.Label(src, text="Ticker").grid(row=0, column=0, sticky="w")
        ttk.Entry(src, textvariable=self.ticker_var, width=12).grid(row=0, column=1, sticky="w")
        ttk.Button(src, text="Load Expirations", command=self.load_expirations).grid(row=0, column=2, padx=(8, 0))

        self.spot_lbl = ttk.Label(src, text="Spot: —")
        self.spot_lbl.grid(row=0, column=3, sticky="w", padx=(12, 0))

        ttk.Label(src, text="Expiry₁").grid(row=1, column=0, sticky="w")
        ttk.Label(src, text="Expiry₂").grid(row=1, column=2, sticky="w")
        self.exp1_var = tk.StringVar()
        self.exp2_var = tk.StringVar()
        self.exp1_combo = ttk.Combobox(src, textvariable=self.exp1_var, width=14, state="readonly")
        self.exp2_combo = ttk.Combobox(src, textvariable=self.exp2_var, width=14, state="readonly")
        self.exp1_combo.grid(row=1, column=1, sticky="w")
        self.exp2_combo.grid(row=1, column=3, sticky="w")
        ttk.Button(src, text="Fetch IVs", command=self.fetch_ivs_and_dtes).grid(row=1, column=4, padx=(12, 0))

        inputs = ttk.LabelFrame(self, text="Inputs (auto-filled; editable)", padding=10)
        inputs.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        for c in range(8):
            inputs.columnconfigure(c, weight=1)

        self.dte1_var = tk.StringVar()
        self.iv1_var = tk.StringVar()
        self.dte2_var = tk.StringVar()
        self.iv2_var = tk.StringVar()

        ttk.Label(inputs, text="DTE₁").grid(row=0, column=0, sticky="w")
        ttk.Entry(inputs, textvariable=self.dte1_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(inputs, text="IV₁ (%)").grid(row=0, column=2, sticky="w")
        ttk.Entry(inputs, textvariable=self.iv1_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(inputs, text="DTE₂").grid(row=1, column=0, sticky="w")
        ttk.Entry(inputs, textvariable=self.dte2_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(inputs, text="IV₂ (%)").grid(row=1, column=2, sticky="w")
        ttk.Entry(inputs, textvariable=self.iv2_var, width=10).grid(row=1, column=3, sticky="w")

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, sticky="w", pady=8)
        ttk.Button(btns, text="Compute", command=self.compute).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Clear", command=self.clear).grid(row=0, column=1, padx=4)

        out = ttk.LabelFrame(self, text="Outputs", padding=10)
        out.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.fwd_vol_lbl = ttk.Label(out, text="Forward Vol: —")
        self.ff_lbl = ttk.Label(out, text="FF: —")
        self.fwd_vol_lbl.grid(row=0, column=0, sticky="w")
        self.ff_lbl.grid(row=1, column=0, sticky="w")

    # ---------- Helpers ----------
    def _calc_dte(self, expiry_iso: str) -> int:
        y, m, d = map(int, expiry_iso.split("-"))
        return max((date(y, m, d) - _now_pacific_date()).days, 0)

    def _get_spot(self, ticker: str) -> Optional[float]:
        tk_obj = yf.Ticker(ticker)
        spot = tk_obj.info.get("regularMarketPrice")
        if spot is None:
            px = tk_obj.history(period="1d")
            if not px.empty:
                spot = float(px["Close"].iloc[-1])
        return _first_float(spot)

    def load_expirations(self):
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            return messagebox.showerror("Ticker required", "Enter a ticker.")
        try:
            tk_obj = yf.Ticker(ticker)
            self.expiries = tk_obj.options or []
            if not self.expiries:
                return messagebox.showerror("No expiries", f"No option expirations for {ticker}.")
            self.exp1_combo["values"] = self.expiries
            self.exp2_combo["values"] = self.expiries
            self.exp1_combo.set(self.expiries[0])
            self.exp2_combo.set(self.expiries[min(1, len(self.expiries) - 1)])
            self.spot = self._get_spot(ticker)
            self.spot_lbl.config(text=f"Spot: {self.spot:.2f}" if self.spot is not None else "Spot: —")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _atm_iv(self, ticker: str, expiry: str, spot: float) -> Optional[float]:
        """Average call/put IV at nearest-to-spot (ATM) strike for one expiry."""
        oc = yf.Ticker(ticker).option_chain(expiry)
        calls, puts = oc.calls, oc.puts
        if calls.empty and puts.empty:
            return None
        strikes = pd.Index(sorted(set(calls["strike"]).union(set(puts["strike"]))))
        if len(strikes) == 0:
            return None
        atm = float(min(strikes, key=lambda s: abs(float(s) - spot)))

        def iv_from(df: pd.DataFrame) -> Optional[float]:
            row = df.loc[df["strike"] == atm]
            if row.empty:
                return None
            return _first_float(row["impliedVolatility"].iloc[0])

        c_iv = iv_from(calls)
        p_iv = iv_from(puts)
        if c_iv is None and p_iv is None:
            return None
        if c_iv is None:
            return p_iv * 100.0
        if p_iv is None:
            return c_iv * 100.0
        return 0.5 * (c_iv + p_iv) * 100.0

    # ---------- Actions ----------
    def fetch_ivs_and_dtes(self):
        ticker = self.ticker_var.get().strip().upper()
        e1, e2 = self.exp1_var.get(), self.exp2_var.get()
        if not (ticker and e1 and e2):
            return messagebox.showerror("Missing", "Load expirations and select both expiries.")
        if e1 == e2:
            return messagebox.showerror("Invalid", "Expiry₂ must be later/different than Expiry₁.")
        self.spot = self._get_spot(ticker)
        if self.spot is None:
            return messagebox.showerror("Spot error", "Could not fetch spot.")

        iv1 = self._atm_iv(ticker, e1, self.spot)
        iv2 = self._atm_iv(ticker, e2, self.spot)
        if iv1 is None or iv2 is None:
            return messagebox.showerror("IV error", "Could not fetch ATM IVs.")
        dte1, dte2 = self._calc_dte(e1), self._calc_dte(e2)
        if dte2 <= dte1:
            return messagebox.showerror("Order", "Require DTE₂ > DTE₁.")

        self.iv1_var.set(f"{iv1:.2f}")
        self.iv2_var.set(f"{iv2:.2f}")
        self.dte1_var.set(str(dte1))
        self.dte2_var.set(str(dte2))
        self.spot_lbl.config(text=f"Spot: {self.spot:.2f}")

    def compute(self):
        try:
            dte1, iv1 = float(self.dte1_var.get()), float(self.iv1_var.get())
            dte2, iv2 = float(self.dte2_var.get()), float(self.iv2_var.get())
        except ValueError:
            return messagebox.showerror("Invalid", "Numeric DTE/IV required.")

        T1, T2 = dte1 / 365.0, dte2 / 365.0
        s1, s2 = iv1 / 100.0, iv2 / 100.0
        denom = T2 - T1
        if denom <= 0:
            return messagebox.showerror("Order", "T₂ must be > T₁.")
        tv1, tv2 = (s1 ** 2) * T1, (s2 ** 2) * T2
        fwd_var = (tv2 - tv1) / denom
        if fwd_var < 0:
            return messagebox.showerror("Math", "Negative forward variance (check inputs).")
        fwd_sigma = math.sqrt(fwd_var)
        ff = (s1 - fwd_sigma) / fwd_sigma if fwd_sigma else None

        self.fwd_vol_lbl.config(text=f"Forward Vol: {fwd_sigma * 100:.3f}%")
        self.ff_lbl.config(text=("FF: n/a" if ff is None else f"FF: {ff * 100:.3f}%"))

    def clear(self):
        for v in (self.dte1_var, self.iv1_var, self.dte2_var, self.iv2_var):
            v.set("")
        self.fwd_vol_lbl.config(text="Forward Vol: —")
        self.ff_lbl.config(text="FF: —")


# ============================== Screener Tab ==============================

class ScreenerFrame(ttk.Frame):
    """
    Screener:
      - Pairs: 30–60, 30–90, 60–90
      - ATM IVs, Forward Vol, FF
      - ATM Call Calendar Spread debit ($) using common strike
      - EARN_IN_WINDOW column shows earnings date if between leg dates
      - Row tag 'earn' (yellow) when earnings in window; 'hot' (green) when FF ≥ 0.20
      - Click column header to sort (toggle desc/asc)
      - Unlimited tickers (bounded concurrency under the hood)
    """

    COLS = ["ticker", "pair", "strike", "exp1", "dte1", "iv1", "call1_mid",
            "exp2", "dte2", "iv2", "call2_mid", "fwd_vol", "ff", "cal_debit", "earn_in_window"]

    # Max parallel per-ticker workers (network-bound; 8–16 is usually safe)
    MAX_WORKERS = 12

    def __init__(self, parent):
        super().__init__(parent, padding=16)
        self.tz_local = PACIFIC
        self._chain_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._rows: List[Dict] = []
        self._sort_state: Dict[str, bool] = {}

        # cross-thread queue + drain loop
        self._q: "queue.Queue[Tuple[int, str, dict]]" = queue.Queue()
        self._drain_job: Optional[str] = None

        # scan control (prevents duplicates across runs)
        self._scan_id_counter: int = 0
        self._active_scan_id: Optional[int] = None
        self._active_lock = threading.Lock()

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        ttk.Label(self, text="Screener Mode — Multiple Tickers",
                  font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 10))

        ctrl = ttk.LabelFrame(self, text="Controls", padding=10)
        ctrl.grid(row=1, column=0, columnspan=8, sticky="ew")
        for c in range(8):
            ctrl.columnconfigure(c, weight=1)

        self.ticker_text = tk.Text(ctrl, width=80, height=3)
        self.ticker_text.insert("1.0", "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, NFLX, AMD, AVGO")
        self.ticker_text.grid(row=0, column=0, columnspan=5, sticky="w")

        self.btn_run = ttk.Button(ctrl, text="Run Screener", command=self.run_manual)
        self.btn_run.grid(row=0, column=5, padx=6, sticky="w")
        self.btn_gainers = ttk.Button(ctrl, text="Top 10 Gainers", command=self.run_gainers)
        self.btn_gainers.grid(row=0, column=6, padx=6, sticky="w")

        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings", height=20)
        for c in self.COLS:
            self.tree.heading(c, text=c.upper(), command=lambda col=c: self._sort_by(col))
            self.tree.column(c, width=100, anchor="center")
        self.tree.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.grid(row=2, column=1, sticky="ns")

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.tree.tag_configure("hot", background="#d7ffd9")
        self.tree.tag_configure("earn", background="#fff3b0")

    # ---------- Queue/UI drain ----------
    def _enqueue(self, scan_id: int, kind: str, payload: Dict):
        """Called from worker thread ONLY."""
        self._q.put((scan_id, kind, payload))

    def _start_drain(self):
        if self._drain_job is None:
            self._drain_job = self.after(30, self._drain_once)

    def _stop_drain(self):
        if self._drain_job is not None:
            self.after_cancel(self._drain_job)
            self._drain_job = None

    def _drain_once(self):
        processed = 0
        try:
            while processed < 300:
                scan_id, kind, payload = self._q.get_nowait()

                # Ignore stale messages from older scans
                if self._active_scan_id is None or scan_id != self._active_scan_id:
                    processed += 1
                    continue

                if kind == "row":
                    row = payload["row"]
                    tags = tuple(payload.get("tags", ()))
                    self._rows.append({**row, "_tags": tags})
                    values = [row.get(c, "—") for c in self.COLS]
                    self.tree.insert("", "end", values=values, tags=tags)

                elif kind == "clear":
                    self.tree.delete(*self.tree.get_children())
                    self._rows.clear()

                elif kind == "error":
                    row = payload["row"]
                    self._rows.append({**row, "_tags": ()})
                    values = [row.get(c, "—") for c in self.COLS]
                    self.tree.insert("", "end", values=values)

                elif kind == "done":
                    # apply active sort once at the end
                    if self._sort_state:
                        active_col = next(iter(self._sort_state))
                        self._resort_and_rebuild(active_col, self._sort_state[active_col])
                    # re-enable controls
                    self.btn_run.state(["!disabled"])
                    self.btn_gainers.state(["!disabled"])
                    with self._active_lock:
                        self._active_scan_id = None  # mark scan finished

                processed += 1
        except queue.Empty:
            pass
        finally:
            # keep draining; low cost when idle
            self._drain_job = self.after(30, self._drain_once)

    # ---------- Sorting ----------
    def _sort_key_for_value(self, col: str, val: str):
        if not val or val in ("—", "-"):
            return (1, None)
        s = str(val).strip()
        # Fast checks
        if s.endswith("%"):
            try:
                return (0, float(s[:-1]))
            except Exception:
                return (0, s)
        if s[0] == "$":
            try:
                return (0, float(s[1:]))
            except Exception:
                return (0, s)
        # numeric?
        try:
            return (0, float(s))
        except Exception:
            pass
        # date yyyy-mm-dd?
        if len(s) == 10 and s[4:5] == "-" and s[7:8] == "-":
            try:
                y, m, d = map(int, s.split("-"))
                return (0, (y, m, d))
            except Exception:
                return (0, s.lower())
        return (0, s.lower())

    def _update_headers_with_arrow(self, active_col: Optional[str]):
        for c in self.COLS:
            base = c.upper()
            if c == active_col:
                asc = self._sort_state.get(c, False)
                arrow = "▲" if asc else "▼"
                txt = f"{base} {arrow}"
            else:
                txt = base
            self.tree.heading(c, text=txt, command=lambda col=c: self._sort_by(col))

    def _resort_and_rebuild(self, col: str, asc: bool):
        def row_val(r: Dict):
            return self._sort_key_for_value(col, r.get(col))
        sorted_rows = sorted(self._rows, key=row_val, reverse=not asc)
        self.tree.delete(*self.tree.get_children())
        ins = self.tree.insert
        for r in sorted_rows:
            ins("", "end", values=[r.get(c, "—") for c in self.COLS], tags=r.get("_tags", ()))

    def _sort_by(self, col: str):
        cur = self._sort_state.get(col, None)
        asc = False if cur is None else not cur
        self._sort_state = {col: asc}
        self._update_headers_with_arrow(col)
        self._resort_and_rebuild(col, asc)

    # ---------- Parsing ----------
    @staticmethod
    def _parse_tickers(raw: str) -> List[str]:
        # split on commas, whitespace, and newlines; preserve order; de-dup
        tokens = [t.strip().upper() for chunk in raw.replace(",", " ").split() for t in [chunk] if t.strip()]
        # preserve order while deduping
        return list(dict.fromkeys(tokens))

    # ---------- Actions ----------
    def run_manual(self):
        tickers = self._parse_tickers(self.ticker_text.get("1.0", "end"))
        if not tickers:
            return messagebox.showerror("Input", "Provide at least one ticker.")
        self._start_scan(tickers)

    def run_gainers(self):
        try:
            tickers = self._fetch_top_gainers()
            self.ticker_text.delete("1.0", "end")
            self.ticker_text.insert("1.0", ", ".join(tickers))
            self._start_scan(tickers)
        except Exception as e:
            messagebox.showerror("Screener", f"Could not fetch top gainers.\n{e}")

    def _fetch_top_gainers(self) -> List[str]:
        try:
            if hasattr(yf, "get_day_gainers"):
                df = yf.get_day_gainers()
                if df is not None and not df.empty and "Symbol" in df.columns:
                    syms = df["Symbol"].dropna().astype(str).head(10).tolist()
                    if syms:
                        return syms
        except Exception:
            pass
        try:
            url = "https://finance.yahoo.com/screener/predefined/day_gainers?offset=0&count=25"
            tables = pd.read_html(url)
            for t in tables:
                if "Symbol" in t.columns:
                    syms = t["Symbol"].dropna().astype(str).head(10).tolist()
                    if syms:
                        return syms
        except Exception:
            pass
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "AMD", "AVGO"]

    def _start_scan(self, tickers: List[str]):
        # generate new scan id
        with self._active_lock:
            self._scan_id_counter += 1
            scan_id = self._scan_id_counter
            self._active_scan_id = scan_id

        # disable controls during scan
        self.btn_run.state(["disabled"])
        self.btn_gainers.state(["disabled"])

        # reset state and start
        self._chain_cache.clear()
        self._sort_state.clear()
        self._update_headers_with_arrow(None)
        self._enqueue(scan_id, "clear", {})
        self._start_drain()

        # Run per-ticker jobs concurrently (bounded)
        threading.Thread(target=self._scan_pool, args=(scan_id, tickers), daemon=True).start()

    def _is_scan_active(self, scan_id: int) -> bool:
        with self._active_lock:
            return self._active_scan_id == scan_id

    # ---------- Pool orchestration ----------
    def _scan_pool(self, scan_id: int, tickers: List[str]):
        # Using a pool improves latency significantly on many tickers (network-bound)
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS, thread_name_prefix="screener") as pool:
            futures = [pool.submit(self._scan_one_ticker, scan_id, t) for t in tickers]
            # Iterate as they complete; results are already enqueued inside _scan_one_ticker
            for _ in as_completed(futures):
                # Early stop if a new scan has started
                if not self._is_scan_active(scan_id):
                    break

        # signal finish for this scan id (even if canceled; drain ignores stale)
        self._enqueue(scan_id, "done", {})

    # ---------- One ticker ----------
    def _scan_one_ticker(self, scan_id: int, ticker: str):
        if not self._is_scan_active(scan_id):
            return
        try:
            spot = self._get_spot(ticker)
            if spot is None:
                self._enqueue(scan_id, "error", {"row": {
                    "ticker": ticker, "pair": "—", "strike": "—",
                    "exp1": "—", "dte1": "—", "iv1": "—", "call1_mid": "—",
                    "exp2": "—", "dte2": "—", "iv2": "—", "call2_mid": "—",
                    "fwd_vol": "—", "ff": "No spot", "cal_debit": "—", "earn_in_window": "—",
                }})
                return

            expiries = yf.Ticker(ticker).options or []
            if not expiries:
                self._enqueue(scan_id, "error", {"row": {
                    "ticker": ticker, "pair": "—", "strike": "—",
                    "exp1": "—", "dte1": "—", "iv1": "—", "call1_mid": "—",
                    "exp2": "—", "dte2": "—", "iv2": "—", "call2_mid": "—",
                    "fwd_vol": "—", "ff": "No expirations", "cal_debit": "—", "earn_in_window": "—",
                }})
                return

            ed = [(e, self._calc_dte(e)) for e in expiries]

            def nearest(target):
                return min(ed, key=lambda x: abs(x[1] - target)) if ed else None

            e30 = nearest(30)
            e60 = nearest(60)
            e90 = nearest(90)

            pairs = []
            if e30 and e60 and e60[1] > e30[1]: pairs.append(("30–60", e30, e60))
            if e30 and e90 and e90[1] > e30[1]: pairs.append(("30–90", e30, e90))
            if e60 and e90 and e90[1] > e60[1]: pairs.append(("60–90", e60, e90))
            if not pairs:
                self._enqueue(scan_id, "error", {"row": {
                    "ticker": ticker, "pair": "—", "strike": "—",
                    "exp1": "—", "dte1": "—", "iv1": "—", "call1_mid": "—",
                    "exp2": "—", "dte2": "—", "iv2": "—", "call2_mid": "—",
                    "fwd_vol": "—", "ff": "No valid pairs", "cal_debit": "—", "earn_in_window": "—",
                }})
                return

            next_earn_dt = self._next_earnings_date(ticker)

            for label, (exp1, dte1), (exp2, dte2) in pairs:
                if not self._is_scan_active(scan_id):
                    return
                try:
                    iv1 = self._atm_iv(ticker, exp1, spot)
                    iv2 = self._atm_iv(ticker, exp2, spot)
                    if iv1 is None or iv2 is None:
                        self._enqueue(scan_id, "row", {"row": {
                            "ticker": ticker, "pair": label, "strike": "-",
                            "exp1": exp1, "dte1": dte1, "iv1": "—", "call1_mid": "—",
                            "exp2": exp2, "dte2": dte2, "iv2": "—", "call2_mid": "—",
                            "fwd_vol": "—", "ff": "—", "cal_debit": "—", "earn_in_window": "—",
                        }, "tags": ()})
                        continue

                    s1, s2 = iv1 / 100.0, iv2 / 100.0
                    T1, T2 = dte1 / 365.0, dte2 / 365.0
                    fwd_sigma, ff = self._forward_and_ff(s1, T1, s2, T2)

                    strike, call1_mid, call2_mid, cal_debit = self._calendar_debit(ticker, exp1, exp2, spot)

                    earn_txt = "—"
                    row_tags = []
                    if next_earn_dt is not None:
                        exp1_date = self._expiry_to_date(exp1)
                        exp2_date = self._expiry_to_date(exp2)
                        if exp1_date and exp2_date:
                            start = min(exp1_date, exp2_date)
                            end = max(exp1_date, exp2_date)
                            if start <= next_earn_dt <= end:
                                earn_txt = next_earn_dt.strftime("%Y-%m-%d")
                                row_tags.append("earn")

                    if ff is not None and ff >= 0.20:
                        row_tags.append("hot")

                    row = {
                        "ticker": ticker,
                        "pair": label,
                        "strike": f"{strike:.2f}" if strike else "-",
                        "exp1": exp1,
                        "dte1": dte1,
                        "iv1": f"{iv1:.2f}%",
                        "call1_mid": (f"{call1_mid:.2f}" if call1_mid is not None else "—"),
                        "exp2": exp2,
                        "dte2": dte2,
                        "iv2": f"{iv2:.2f}%",
                        "call2_mid": (f"{call2_mid:.2f}" if call2_mid is not None else "—"),
                        "fwd_vol": (f"{fwd_sigma * 100.0:.2f}%" if fwd_sigma is not None else "—"),
                        "ff": (f"{ff * 100.0:.2f}%" if ff is not None else "—"),
                        "cal_debit": (f"{cal_debit:.2f}" if cal_debit is not None else "—"),
                        "earn_in_window": earn_txt,
                    }
                    self._enqueue(scan_id, "row", {"row": row, "tags": tuple(row_tags)})
                except Exception as e:
                    self._enqueue(scan_id, "error", {"row": {
                        "ticker": ticker, "pair": label, "strike": "—",
                        "exp1": "—", "dte1": "—", "iv1": "—", "call1_mid": "—",
                        "exp2": "—", "dte2": "—", "iv2": "—", "call2_mid": "—",
                        "fwd_vol": "—", "ff": f"{label} err: {e}", "cal_debit": "—", "earn_in_window": "—",
                    }})
                    continue
        except Exception as e:
            self._enqueue(scan_id, "error", {"row": {
                "ticker": ticker, "pair": "—", "strike": "—",
                "exp1": "—", "dte1": "—", "iv1": "—", "call1_mid": "—",
                "exp2": "—", "dte2": "—", "iv2": "—", "call2_mid": "—",
                "fwd_vol": "—", "ff": f"scan err: {e}", "cal_debit": "—", "earn_in_window": "—",
            }})

    # ---------- Market helpers ----------
    def _get_spot(self, ticker: str) -> Optional[float]:
        tk_obj = yf.Ticker(ticker)
        spot = tk_obj.info.get("regularMarketPrice")
        if spot is None:
            px = tk_obj.history(period="1d")
            if not px.empty:
                spot = float(px["Close"].iloc[-1])
        return _first_float(spot)

    def _calc_dte(self, expiry_iso: str) -> int:
        y, m, d = map(int, expiry_iso.split("-"))
        return max((date(y, m, d) - _now_pacific_date()).days, 0)

    def _expiry_to_date(self, expiry_iso: str) -> Optional[date]:
        try:
            y, m, d = map(int, expiry_iso.split("-"))
            return date(y, m, d)
        except Exception:
            return None

    def _load_chain(self, ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        key = (ticker, expiry)
        if key in self._chain_cache:
            return self._chain_cache[key]
        oc = yf.Ticker(ticker).option_chain(expiry)
        self._chain_cache[key] = (oc.calls, oc.puts)
        return self._chain_cache[key]

    def _atm_iv(self, ticker: str, expiry: str, spot: float) -> Optional[float]:
        calls, puts = self._load_chain(ticker, expiry)
        if calls.empty and puts.empty:
            return None
        strikes = pd.Index(sorted(set(calls["strike"]).union(set(puts["strike"]))))
        if len(strikes) == 0:
            return None
        atm = float(min(strikes, key=lambda s: abs(float(s) - spot)))

        def iv_from(df: pd.DataFrame) -> Optional[float]:
            row = df.loc[df["strike"] == atm]
            if row.empty:
                return None
            return _first_float(row["impliedVolatility"].iloc[0])

        c_iv = iv_from(calls)
        p_iv = iv_from(puts)
        if c_iv is None and p_iv is None:
            return None
        if c_iv is None:
            return p_iv * 100.0
        if p_iv is None:
            return c_iv * 100.0
        return 0.5 * (c_iv + p_iv) * 100.0

    def _common_atm_strike(self, ticker: str, exp1: str, exp2: str, spot: float) -> Optional[float]:
        """Pick a common strike (closest to spot) present in both expiries."""
        calls1, puts1 = self._load_chain(ticker, exp1)
        calls2, puts2 = self._load_chain(ticker, exp2)
        S1: Set[float] = set(map(float, pd.Index(sorted(set(calls1["strike"]).union(set(puts1["strike"])))))) if not (calls1.empty and puts1.empty) else set()
        S2: Set[float] = set(map(float, pd.Index(sorted(set(calls2["strike"]).union(set(puts2["strike"])))))) if not (calls2.empty and puts2.empty) else set()
        inter = list(S1.intersection(S2))
        if not inter:
            return None
        return float(min(inter, key=lambda s: abs(s - spot)))

    def _call_mid_at(self, calls_df: pd.DataFrame, strike: float) -> Optional[float]:
        row = calls_df.loc[calls_df["strike"] == strike]
        if row.empty:
            return None
        return _safe_mid(row.iloc[0])

    def _calendar_debit(self, ticker: str, exp1: str, exp2: str, spot: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Return (strike, short_mid, long_mid, long_mid - short_mid) for ATM call calendar.
        Uses a common strike present in both expiries when available.
        """
        calls1, _ = self._load_chain(ticker, exp1)
        calls2, _ = self._load_chain(ticker, exp2)

        strike = self._common_atm_strike(ticker, exp1, exp2, spot)
        if strike is None:
            return None, None, None, None

        short_mid = self._call_mid_at(calls1, strike)
        long_mid = self._call_mid_at(calls2, strike)
        if short_mid is None or long_mid is None:
            return strike, short_mid, long_mid, None
        return strike, short_mid, long_mid, (long_mid - short_mid)

    # ---------- Forward vol math ----------
    @staticmethod
    def _forward_and_ff(s1: float, T1: float, s2: float, T2: float) -> Tuple[Optional[float], Optional[float]]:
        denom = T2 - T1
        if denom <= 0:
            return None, None
        fwd_var = (s2 ** 2 * T2 - s1 ** 2 * T1) / denom
        if fwd_var < 0:
            return None, None
        fwd_sigma = math.sqrt(fwd_var)
        ff = None if fwd_sigma == 0 else (s1 - fwd_sigma) / fwd_sigma
        return fwd_sigma, ff

    # ---------- Earnings ----------
    def _next_earnings_date(self, ticker: str) -> Optional[date]:
        today = _now_pacific_date()
        try:
            tk_obj = yf.Ticker(ticker)
            try:
                df = tk_obj.get_earnings_dates(limit=8)
                if df is not None and not df.empty:
                    if "Earnings Date" in df.columns:
                        dates = pd.to_datetime(df["Earnings Date"], utc=True, errors="coerce").dt.date
                    else:
                        dates = pd.to_datetime(df.index, utc=True, errors="coerce").date
                    fut = [d for d in dates if d is not None and d >= today]
                    if fut:
                        return min(fut)
            except Exception:
                pass
            try:
                cal = tk_obj.calendar
                if cal is not None and not cal.empty:
                    c = cal.copy()
                    if "Earnings Date" in c.index:
                        vals = c.loc["Earnings Date"].values
                        for v in vals:
                            d = pd.to_datetime(v, utc=True, errors="coerce")
                            if pd.notna(d):
                                dd = d.date()
                                if dd >= today:
                                    return dd
                    else:
                        ct = c.T
                        if "Earnings Date" in ct.columns:
                            vals = ct["Earnings Date"].values
                            for v in vals:
                                d = pd.to_datetime(v, utc=True, errors="coerce")
                                if pd.notna(d):
                                    dd = d.date()
                                    if dd >= today:
                                        return dd
            except Exception:
                pass
            try:
                raw = tk_obj.info.get("earningsDate")
                if raw:
                    if isinstance(raw, (list, tuple)) and raw:
                        raw = raw[0]
                    d = pd.to_datetime(raw, utc=True, errors="coerce")
                    if pd.notna(d):
                        dd = d.date()
                        if dd >= today:
                            return dd
            except Exception:
                pass
            for key in ("earnings_date", "earningsDate", "next_earnings_date", "nextEarningsDate"):
                try:
                    raw = getattr(tk_obj, "fast_info", {}).get(key, None)
                    if raw:
                        d = pd.to_datetime(raw, utc=True, errors="coerce")
                        if pd.notna(d):
                            dd = d.date()
                            if dd >= today:
                                return dd
                except Exception:
                    continue
        except Exception:
            return None
        return None


# ============================== Main Window ==============================

class ForwardVolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Forward Volatility — Manual & Screener")
        self.geometry("1200x860")
        self.minsize(950, 720)
        self.resizable(True, True)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self._fullscreen = False

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.manual = ManualCalculator(nb)
        self.screener = ScreenerFrame(nb)

        nb.add(self.manual, text="Manual Mode")
        nb.add(self.screener, text="Screener Mode")

    def _toggle_fullscreen(self, _e=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, _e=None):
        self._fullscreen = False
        self.attributes("-fullscreen", False)


if __name__ == "__main__":
    app = ForwardVolApp()
    app.mainloop()
