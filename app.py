import math
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# App/session setup
# =========================
PACIFIC = pytz.timezone("America/Los_Angeles")
EASTERN = pytz.timezone("America/New_York")
st.set_page_config(page_title="Forward Vol Screener — Top 27 (All US Stocks)", layout="wide")

# Persist results across reruns (don’t overwrite if already set)
if "df" not in st.session_state: st.session_state.df = None
if "tickers" not in st.session_state: st.session_state.tickers = []
if "vol_topN" not in st.session_state: st.session_state.vol_topN = None

# --- Config ---
TOP_STOCKS = 27            # number of stocks to select
ADD_ETFS = True            # set False to omit ETFs
FIXED_ETFS = ["VOO", "SPY", "QQQ"]
MAX_WORKERS = 12           # options-chain parallelism

# Download tuning
DL_CHUNK_STAGE1 = 80       # small chunks → more responsive UI for the quick pass
DL_CHUNK_STAGE2 = 50       # shortlist chunks (heavier columns, keep small)
SHORTLIST_SIZE = 400       # how many symbols to carry to the accurate pass

# =========================
# Helpers
# =========================
def _now_pacific_date() -> date:
    return datetime.now(PACIFIC).date()

def _first_float(val) -> Optional[float]:
    try:
        x = float(val)
        return None if math.isnan(x) else x
    except Exception:
        return None

def _safe_mid(row: pd.Series) -> Optional[float]:
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

def _calc_dte(expiry_iso: str) -> int:
    y, m, d = map(int, expiry_iso.split("-"))
    return max((date(y, m, d) - _now_pacific_date()).days, 0)

def _expiry_to_date(expiry_iso: str) -> Optional[date]:
    try:
        y, m, d = map(int, expiry_iso.split("-"))
        return date(y, m, d)
    except Exception:
        return None

def _normalize_tickers(raw: str) -> List[str]:
    seen, out = set(), []
    for t in raw.replace(",", " ").split():
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# =========================
# yfinance (cached)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_spot(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    spot = tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice")
    if spot is None:
        px = tk.history(period="1d")
        if not px.empty:
            spot = float(px["Close"].iloc[-1])
    return _first_float(spot)

@st.cache_data(ttl=600, show_spinner=False)
def get_options(ticker: str) -> List[str]:
    try:
        opts = yf.Ticker(ticker).options or []
        return [e for e in opts if isinstance(e, str) and len(e) == 10 and e[4] == "-" and e[7] == "-"]
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oc = yf.Ticker(ticker).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

@st.cache_data(ttl=600, show_spinner=False)
def get_next_earnings_date(ticker: str) -> Optional[date]:
    today = _now_pacific_date()
    tk = yf.Ticker(ticker)
    try:
        df = tk.get_earnings_dates(limit=8)
        if df is not None and not df.empty:
            col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
            dates = pd.to_datetime(df[col], utc=True, errors="coerce").dt.date
            fut = [d for d in dates if d and d >= today]
            if fut:
                return min(fut)
    except Exception:
        pass
    for src in (getattr(tk, "calendar", None), tk.info, getattr(tk, "fast_info", {})):
        if src is None:
            continue
        for key in ("Earnings Date", "earningsDate", "earnings_date", "nextEarningsDate"):
            try:
                val = src[key] if isinstance(src, dict) else src.loc[key].values
                if isinstance(val, (list, tuple, pd.Series, pd.Index)):
                    val = val[0]
                d = pd.to_datetime(val, utc=True, errors="coerce")
                if pd.notna(d) and d.date() >= today:
                    return d.date()
            except Exception:
                continue
    return None

# =========================
# Volume ranking (ALL US stocks) — two-stage faster pipeline
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def get_us_stock_universe() -> List[str]:
    """
    Build a broad U.S. stock universe from yfinance helpers.
    Falls back gracefully if any helper is missing on your yfinance version.
    """
    tickers = []
    # NASDAQ
    try:
        nas = getattr(yf, "tickers_nasdaq", None)
        if callable(nas):
            r = nas()
            if isinstance(r, (list, tuple)) and r:
                tickers.extend([t.strip().upper() for t in r if t])
    except Exception:
        pass
    # "Other" (NYSE/AMEX/misc)
    try:
        oth = getattr(yf, "tickers_other", None)
        if callable(oth):
            r = oth()
            if isinstance(r, (list, tuple)) and r:
                tickers.extend([t.strip().upper() for t in r if t])
    except Exception:
        pass
    # If nothing returned, use S&P500 as a minimal fallback so the app still runs
    if not tickers:
        try:
            spx = getattr(yf, "tickers_sp500", None)
            if callable(spx):
                r = spx()
                if isinstance(r, (list, tuple)) and r:
                    tickers.extend([t.strip().upper() for t in r if t])
        except Exception:
            pass
    # Dedup, preserve order
    seen, out = set(), []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _exclude_today_if_open(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude today's partial bar based on US market close (after ~4:05pm ET is 'closed')."""
    if df is None or df.empty:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(EASTERN)
    df = df.copy()
    df["et_date"] = idx_et.date

    now_et = datetime.now(EASTERN)
    today_et = now_et.date()
    market_closed = (now_et.hour > 16) or (now_et.hour == 16 and now_et.minute >= 5)
    if not market_closed:
        df = df[df["et_date"] < today_et]
    return df

def _avg5(vol: pd.Series) -> Optional[float]:
    vols = vol.dropna()
    last5 = vols.tail(5)
    if len(last5) < 5:
        return None
    return float(last5.mean())

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _safe_multi_download(tickers, period, interval):
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def rank_top_stocks_all_us(extras_tuple: tuple, top_n: int = TOP_STOCKS, add_etfs: bool = ADD_ETFS) -> pd.DataFrame:
    """
    Fast two-stage volume ranking across the full U.S. universe:
      Stage 1: 7d data, use last completed session volume → shortlist (SHORTLIST_SIZE).
      Stage 2: 20d data, compute true 5-day avg → pick top_n.
      Append ETFs if requested.
    extras_tuple must be a tuple for caching.
    """
    extras = list(extras_tuple)
    universe = get_us_stock_universe()
    # Include extras in the universe so they get considered in ranking
    for e in extras:
        if e not in universe:
            universe.append(e)

    # ---------- Stage 1: fast shortlist by last completed session volume ----------
    stage1_rows = []
    total = len(universe)
    prog1 = st.progress(0.0, text="Stage 1/2: Scanning universe (latest volumes)…")
    processed = 0

    for block in _chunks(universe, DL_CHUNK_STAGE1):
        data = _safe_multi_download(block, period="7d", interval="1d")
        # Figure which tickers we actually got back
        if isinstance(data.columns, pd.MultiIndex):
            got = [c[0] for c in data.columns.unique(level=0)]
        else:
            got = block

        for t in got:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if sub is None or sub.empty or "Volume" not in sub.columns:
                    continue
                sub = _exclude_today_if_open(sub)
                if "Volume" not in sub.columns or sub["Volume"].dropna().empty:
                    continue
                last_vol = float(sub["Volume"].dropna().iloc[-1])
                last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
                stage1_rows.append({"ticker": t, "last_vol": last_vol, "last_close": last_close})
            except Exception:
                continue

        processed += len(block)
        prog1.progress(min(processed/total, 1.0))
    prog1.empty()

    if not stage1_rows:
        return pd.DataFrame(columns=["ticker","avg_week_volume","last_close","source"])

    stage1 = pd.DataFrame(stage1_rows).dropna(subset=["last_vol"]).sort_values("last_vol", ascending=False)
    shortlist = stage1["ticker"].head(SHORTLIST_SIZE).tolist()

    # ---------- Stage 2: accurate 5-day average on shortlist ----------
    stage2_rows = []
    prog2 = st.progress(0.0, text="Stage 2/2: Computing 5-day average volume on shortlist…")
    processed = 0
    for block in _chunks(shortlist, DL_CHUNK_STAGE2):
        data = _safe_multi_download(block, period="20d", interval="1d")
        if isinstance(data.columns, pd.MultiIndex):
            got = [c[0] for c in data.columns.unique(level=0)]
        else:
            got = block

        for t in got:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if sub is None or sub.empty or "Volume" not in sub.columns:
                    continue
                sub = _exclude_today_if_open(sub)
                if "Volume" not in sub.columns:
                    continue
                avgv = _avg5(sub["Volume"])
                if avgv is None:
                    continue
                last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
                stage2_rows.append({"ticker": t, "avg_week_volume": avgv, "last_close": last_close, "source": "Stock"})
            except Exception:
                continue

        processed += len(block)
        prog2.progress(min(processed/len(shortlist), 1.0))
    prog2.empty()

    vol_df = pd.DataFrame(stage2_rows)
    if vol_df.empty:
        return pd.DataFrame(columns=["ticker","avg_week_volume","last_close","source"])

    vol_df = vol_df.dropna(subset=["avg_week_volume"]).sort_values("avg_week_volume", ascending=False)
    top_stocks = vol_df.head(top_n).copy()

    # Append ETFs explicitly (no ranking impact)
    if add_etfs:
        etf_rows = []
        for etf in FIXED_ETFS:
            try:
                hist = yf.download(etf, period="20d", interval="1d", auto_adjust=False, progress=False)
                hist = _exclude_today_if_open(hist)
                avgv = _avg5(hist["Volume"]) if "Volume" in hist.columns else None
                last_close = float(hist["Close"].dropna().iloc[-1]) if "Close" in hist.columns and not hist["Close"].dropna().empty else None
            except Exception:
                avgv, last_close = None, None
            etf_rows.append({"ticker": etf, "avg_week_volume": avgv, "last_close": last_close, "source": "ETF"})
        etf_df = pd.DataFrame(etf_rows)
        top_full = pd.concat([top_stocks, etf_df], ignore_index=True)
    else:
        top_full = top_stocks

    top_full = top_full.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return top_full

# =========================
# IV/FF calculations
# =========================
def atm_iv(ticker: str, expiry: str, spot: float) -> Optional[float]:
    calls, puts = get_chain(ticker, expiry)
    if calls.empty and puts.empty:
        return None
    strikes = pd.Index(sorted(set(calls["strike"]).union(set(puts["strike"]))))
    if len(strikes) == 0:
        return None
    atm = float(min(strikes, key=lambda s: abs(float(s) - spot)))

    def iv_from(df: pd.DataFrame) -> Optional[float]:
        row = df.loc[df["strike"] == atm]
        return None if row.empty else _first_float(row["impliedVolatility"].iloc[0])

    c_iv, p_iv = iv_from(calls), iv_from(puts)
    if c_iv is None and p_iv is None:
        return None
    if c_iv is None:
        return p_iv * 100.0
    if p_iv is None:
        return c_iv * 100.0
    return 0.5 * (c_iv + p_iv) * 100.0

def common_atm_strike(ticker: str, exp1: str, exp2: str, spot: float) -> Optional[float]:
    c1, p1 = get_chain(ticker, exp1)
    c2, p2 = get_chain(ticker, exp2)
    s1 = set(map(float, pd.Index(sorted(set(c1["strike"]).union(set(p1["strike"]))))))
    s2 = set(map(float, pd.Index(sorted(set(c2["strike"]).union(set(p2["strike"]))))))
    inter = list(s1.intersection(s2))
    return float(min(inter, key=lambda s: abs(s - spot))) if inter else None

def call_mid_at(calls: pd.DataFrame, strike: float) -> Optional[float]:
    row = calls.loc[calls["strike"] == strike]
    return None if row.empty else _safe_mid(row.iloc[0])

def calendar_debit(ticker: str, e1: str, e2: str, spot: float):
    c1, _ = get_chain(ticker, e1)
    c2, _ = get_chain(ticker, e2)
    strike = common_atm_strike(ticker, e1, e2, spot)
    if strike is None:
        return None, None, None, None
    short_mid, long_mid = call_mid_at(c1, strike), call_mid_at(c2, strike)
    if short_mid is None or long_mid is None:
        return strike, short_mid, long_mid, None
    return strike, short_mid, long_mid, long_mid - short_mid

def forward_and_ff(s1: float, T1: float, s2: float, T2: float):
    denom = T2 - T1
    if denom <= 0:
        return None, None
    fwd_var = (s2**2 * T2 - s1**2 * T1) / denom
    if fwd_var < 0:
        return None, None
    fwd_sigma = math.sqrt(fwd_var)
    ff = None if fwd_sigma == 0 else (s1 - fwd_sigma) / fwd_sigma
    return fwd_sigma, ff

# =========================
# Screener (per ticker; cached)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def screen_ticker(ticker: str) -> List[Dict]:
    spot = get_spot(ticker)
    if spot is None:
        return [{"ticker": ticker, "pair": "—","exp1": "—","dte1": "—","iv1": "—",
                 "exp2": "—","dte2": "—","iv2": "—","fwd_vol": "—","ff": "—",
                 "cal_debit": "—","earn_in_window": "—","_tags": ["no_spot"]}]
    expiries = get_options(ticker)
    if not expiries:
        return [{"ticker": ticker, "pair": "—","exp1": "—","dte1": "—","iv1": "—",
                 "exp2": "—","dte2": "—","iv2": "—","fwd_vol": "—","ff": "—",
                 "cal_debit": "—","earn_in_window": "—","_tags": ["no_exp"]}]

    ed = [(e, _calc_dte(e)) for e in expiries]
    nearest = lambda t: min(ed, key=lambda x: abs(x[1] - t)) if ed else None
    e30, e60, e90 = nearest(30), nearest(60), nearest(90)
    pairs = []
    if e30 and e60 and e60[1] > e30[1]: pairs.append(("30–60", e30, e60))
    if e30 and e90 and e90[1] > e30[1]: pairs.append(("30–90", e30, e90))
    if e60 and e90 and e90[1] > e60[1]: pairs.append(("60–90", e60, e90))
    earn_dt = get_next_earnings_date(ticker)

    rows = []
    for label, (exp1, dte1), (exp2, dte2) in pairs:
        iv1, iv2 = atm_iv(ticker, exp1, spot), atm_iv(ticker, exp2, spot)
        if iv1 is None or iv2 is None:
            continue
        s1, s2 = iv1/100.0, iv2/100.0
        T1, T2 = dte1/365.0, dte2/365.0
        fwd_sigma, ff = forward_and_ff(s1, T1, s2, T2)
        _, _, _, debit = calendar_debit(ticker, exp1, exp2, spot)

        earn_txt, tags = "—", []
        if earn_dt:
            e1d, e2d = _expiry_to_date(exp1), _expiry_to_date(exp2)
            if e1d and e2d and min(e1d, e2d) <= earn_dt <= max(e1d, e2d):
                earn_txt, tags = earn_dt.strftime("%Y-%m-%d"), ["earn"]
        if ff and ff >= 0.20:
            tags.append("hot")

        rows.append({
            "ticker": ticker, "pair": label,
            "exp1": exp1, "dte1": dte1, "iv1": f"{iv1:.2f}%",
            "exp2": exp2, "dte2": dte2, "iv2": f"{iv2:.2f}%",
            "fwd_vol": f"{(fwd_sigma*100):.2f}%" if fwd_sigma else "—",
            "ff": f"{(ff*100):.2f}%" if ff is not None else "—",
            "cal_debit": f"{debit:.2f}" if debit is not None else "—",
            "earn_in_window": earn_txt,
            "_tags": tags,
        })
    return rows

def scan_many(tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    if not tickers:
        return pd.DataFrame(rows)
    progress = st.progress(0.0, text="Scanning option chains…")
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tickers)))) as ex:
        fut_map = {ex.submit(screen_ticker, t): t for t in tickers}
        done = 0
        for fut in as_completed(fut_map):
            t = fut_map[fut]
            try:
                rows.extend(fut.result())
            except Exception as e:
                rows.append({"ticker": t, "pair": "—","exp1": "—","dte1": "—","iv1": "—",
                             "exp2": "—","dte2": "—","iv2": "—","fwd_vol": "—","ff": "—",
                             "cal_debit": "—","earn_in_window": "—","_tags": [f"error:{type(e).__name__}"]})
            done += 1
            progress.progress(done / len(tickers), text=f"Scanned {t} ({done}/{len(tickers)})")
    progress.empty()
    df = pd.DataFrame(rows)
    if "_tags" not in df.columns:
        df["_tags"] = [[] for _ in range(len(df))]
    return df

# =========================
# Robust sorting helpers for the main table
# =========================
_BLANK_SET = {"", "-", "—", "–"}

def _build_sort_columns(series: pd.Series) -> pd.DataFrame:
    s = series.copy()
    s_str = s.astype(str).str.strip()
    is_blank = s.isna() | s_str.isin(_BLANK_SET)

    pct_val = pd.to_numeric(s_str.str.rstrip("%").str.replace(",", "", regex=False), errors="coerce")
    pct_mask = s_str.str.endswith("%") & ~pd.isna(pct_val)

    cur_mask = s_str.str.match(r"^\(?\$\s*[\d,]+(?:\.\d+)?\)?$")
    cur_core = (s_str
                .str.replace(r"^\(", "", regex=True)
                .str.replace(r"\)$", "", regex=True)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip())
    cur_val = pd.to_numeric(cur_core, errors="coerce")
    cur_neg = s_str.str.startswith("(") & s_str.str.endswith(")")
    cur_val = np.where(cur_neg, -cur_val, cur_val)

    num_val_plain = pd.to_numeric(s_str.str.replace(",", "", regex=False), errors="coerce")
    num_key = np.where(~pd.isna(pct_val) & pct_mask, pct_val,
              np.where(~pd.isna(cur_val) & cur_mask, cur_val, num_val_plain))
    num_key = pd.to_numeric(num_key, errors="coerce")

    dt = pd.to_datetime(s_str, errors="coerce", utc=True)
    date_key = dt.view("int64")
    text_key = s_str.str.lower()

    is_num = ~pd.isna(num_key)
    is_date = pd.isna(num_key) & ~pd.isna(date_key)
    is_text = ~(is_num | is_date)

    t_code = np.select([is_num, is_date, is_text], [0, 1, 2], default=2)
    val = pd.Series(np.nan, index=s.index, dtype="object")
    val[is_num] = num_key[is_num]
    val[is_date] = date_key[is_date].astype("float64")
    val[is_text] = text_key[is_text]

    grp = np.where(is_blank, 1, 0)
    t_code = np.where(is_blank, 9, t_code)
    val = np.where(is_blank, "", val)
    return pd.DataFrame({"_grp": grp, "_t": t_code, "_val": val}, index=s.index)

def sort_df(df: pd.DataFrame, col: str, ascending: bool) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    aux = _build_sort_columns(df[col])
    return (df.join(aux)
              .sort_values(by=["_grp","_t","_val"], ascending=[True, True, ascending], kind="mergesort")
              .drop(columns=["_grp","_t","_val"])
              .reset_index(drop=True))

# =========================
# UI
# =========================
DISPLAY_MAP = {
    "ticker": "Ticker",
    "pair": "Pair",
    "exp1": "Exp 1",
    "dte1": "Dte 1",
    "iv1": "IV 1",
    "exp2": "Exp 2",
    "dte2": "Dte 2",
    "iv2": "IV 2",
    "fwd_vol": "Forward Vol",
    "ff": "FF",
    "cal_debit": "Call Debit",
    "earn_in_window": "Earnings Date",
}
LABEL_TO_KEY = {v: k for k, v in DISPLAY_MAP.items()}
DISPLAY_KEYS = ["ticker","pair","exp1","dte1","iv1","exp2","dte2","iv2","fwd_vol","ff","cal_debit","earn_in_window"]

st.title("📈 Forward Volatility Screener (Top 27 by 5-Day Avg Volume — All US Stocks)")

st.markdown(
    "This scans **all U.S. stocks (NASDAQ + NYSE/AMEX)** in two stages, ranks by **average volume over the last 5 completed sessions**, "
    f"selects the **Top {TOP_STOCKS} stocks**, and {'adds **VOO, SPY, QQQ**' if ADD_ETFS else 'does not add ETFs'}."
)

raw_extra = st.text_input("Optional: Add tickers (comma/space separated)", "", placeholder="e.g., NVDA, TSLA, META")

# --- The button now always triggers immediate visual feedback via Stage 1 progress bar ---
colA, colB = st.columns([1, 3])
with colA:
    run = st.button("Build List & Run", type="primary", help="Two-stage fast scan across the US universe")
with colB:
    st.caption("Excludes today's partial volume until after the 4:05pm ET close.")

if run:
    extras = tuple(_normalize_tickers(raw_extra))  # tuple → cache-friendly
    vol_top_df = rank_top_stocks_all_us(extras, top_n=TOP_STOCKS, add_etfs=ADD_ETFS)
    st.session_state.vol_topN = vol_top_df
    tickers = vol_top_df["ticker"].tolist()
    st.session_state.tickers = tickers
    st.session_state.df = scan_many(tickers)

# --------- Selected list (stocks + optional ETFs) ---------
if st.session_state.vol_topN is not None and not st.session_state.vol_topN.empty:
    st.subheader(f"Selected Tickers (Top {TOP_STOCKS} Stocks by 5-Day Avg Vol{' + ETFs' if ADD_ETFS else ''})")
    base = st.session_state.vol_topN.copy()
    # Order: Stocks first, then ETFs
    source_order = pd.api.types.CategoricalDtype(categories=["Stock", "ETF"], ordered=True)
    base["Source"] = base["source"].astype(source_order)
    base["SourceOrder"] = base["Source"].cat.codes
    base["AvgVolNum"] = pd.to_numeric(base["avg_week_volume"], errors="coerce")

    base = base.sort_values(by=["SourceOrder", "AvgVolNum"], ascending=[True, False], kind="mergesort")

    disp = pd.DataFrame({
        "Ticker": base["ticker"],
        "Avg Vol (5d)": base["AvgVolNum"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—"),
        "Last Close": base["last_close"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Source": base["Source"].astype(str),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

# --------- Forward-vol table ---------
df_current = st.session_state.df
if df_current is None or df_current.empty:
    st.info("Click **Build List & Run** to rank the entire US universe and scan options.")
else:
    display_labels = [DISPLAY_MAP[k] for k in DISPLAY_KEYS if k in df_current.columns]
    default_label = DISPLAY_MAP.get("ff", display_labels[0])

    c1, c2, c3 = st.columns([3, 1.2, 1.8], vertical_alignment="bottom")
    with c1:
        sort_label = st.selectbox("Sort by", options=display_labels,
                                  index=display_labels.index(default_label), key="sort_col_label")
    with c2:
        sort_ascending = st.toggle("Ascending", value=False, key="sort_asc")
    with c3:
        st.caption("Blanks always shown last")

    sort_key = LABEL_TO_KEY.get(sort_label, "ff")
    df_sorted = sort_df(df_current, sort_key, sort_ascending)

    have_keys = [k for k in DISPLAY_KEYS if k in df_sorted.columns]
    df_display = df_sorted[have_keys].copy()
    df_display.rename(columns={k: DISPLAY_MAP[k] for k in have_keys}, inplace=True)

    tags_series = df_sorted["_tags"] if "_tags" in df_sorted.columns else pd.Series([[]]*len(df_sorted), index=df_sorted.index)

    def _highlight_row(row: pd.Series):
        tags = tags_series.iloc[row.name] if row.name in tags_series.index else []
        earn = isinstance(tags, (list, tuple, set)) and ("earn" in tags)
        hot = isinstance(tags, (list, tuple, set)) and ("hot" in tags)
        color = "#ffffff"
        if earn and hot:
            color = "#ffe0b2"
        elif earn:
            color = "#fff9c4"
        elif hot:
            color = "#dcedc8"
        return [f"background-color:{color}; color:#000000;"] * len(row)

    styled = (df_display.style
              .apply(_highlight_row, axis=1)
              .set_properties(**{"border":"1px solid #bbb","color":"#000","font-size":"14px"}))

    st.markdown("""
        <style>
        table {border-collapse: collapse; width: 100%;}
        th {position: sticky; top: 0; background-color: #f0f0f0; color: #000; font-weight: bold;}
        tr:nth-child(even) {background-color: #fafafa;}
        tr:hover {background-color: #e0e0e0;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(styled.to_html(), unsafe_allow_html=True)
    st.caption("🟩 FF ≥ 0.20 🟨 Earnings in window 🟧 Both conditions true")

    st.markdown(
        "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b></p>",
        unsafe_allow_html=True,
    )
