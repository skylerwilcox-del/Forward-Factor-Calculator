import math
import time
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict
import re
import os

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
st.set_page_config(page_title="Forward Vol Screener (Debug Safe Mode Enabled)", layout="wide")

# Persistent session state
st.session_state.setdefault("df", None)
st.session_state.setdefault("tickers", [])
st.session_state.setdefault("vol_topN", None)
st.session_state.setdefault("trigger_run", False)
st.session_state.setdefault("first_visit", True)

# --- Config ---
TOP_STOCKS = 27
FIXED_ETFS = ["SPY", "QQQ", "VOO"]
ETF_SET = set(FIXED_ETFS)
EM_DASH = "—"
BLANKS = {"", "-", "—", "–", "---"}

# Download tuning (conservative to reduce throttling)
BATCH_3MO = 20
MAX_WORKERS = 5

# Known-good tickers for debug
DEBUG_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]

# Large liquid fallback universe
FALLBACK_LIQUID = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","AMD","NFLX","CRM","COST","PEP","ADBE","INTC",
    "UBER","QCOM","AMAT","TSM","ORCL","PFE","MRK","JNJ","LLY","V","MA","JPM","BAC","WFC","C","GS","MS","SCHW","BLK",
    "GE","HON","CAT","BA","NKE","KO","PG","MCD","WMT","HD","LOW","TGT","DIS","CMCSA","T","VZ","TMUS","CSCO",
    "PANW","NOW","SNOW","DDOG","NET","MDB","ZS","PLTR","SHOP","ABNB","PYPL","SQ","ROKU","TTD","SPOT","DKNG","RBLX",
    "MU","NXPI","TXN","ADI","ON","LRCX","KLAC","ASML","CVX","XOM","COP","OXY","SLB",
    "F","GM","AAL","UAL","DAL","UPS","FDX","SOFI","COIN","HOOD","GME","OPEN"
]

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

def _calc_dte(expiry_iso: str) -> int:
    try:
        y, m, d = map(int, expiry_iso.split("-"))
        return max((date(y, m, d) - _now_pacific_date()).days, 0)
    except Exception:
        return 0

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
            seen.add(t); out.append(t)
    return out

def _is_clean_us_symbol(sym: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}(?:-[A-Z0-9]{1,2})?", sym))

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _fmt_pct(x: Optional[float]) -> str:
    return EM_DASH if x is None else f"{x:.2f}%"

def _fmt_money(x: Optional[float]) -> str:
    return EM_DASH if x is None else f"{x:.2f}"

def _dash_if_none(x):
    return EM_DASH if x is None else x

def _log(msg: str):
    st.session_state.setdefault("_debug_logs", [])
    st.session_state["_debug_logs"].append(msg)

# Simple retry decorator
def _retry(n=3, wait=0.5):
    def deco(fn):
        def wrap(*a, **k):
            last = None
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception as e:
                    last = e
                    time.sleep(wait)
            raise last
        return wrap
    return deco

# =========================
# yfinance (cached + hardened)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_us_stock_universe() -> List[str]:
    # Use only fallback for now (minimize external surface while debugging)
    return list(dict.fromkeys(FALLBACK_LIQUID + FIXED_ETFS + ["OPEN"]))

def _exclude_today_if_open(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(EASTERN)
    dfc = df.copy()
    dfc["et_date"] = idx_et.date
    now_et = datetime.now(EASTERN)
    today_et = now_et.date()
    market_closed = (now_et.hour > 16) or (now_et.hour == 16 and now_et.minute >= 5)
    if not market_closed:
        dfc = dfc[dfc["et_date"] < today_et]
    return dfc.drop(columns=["et_date"])

@st.cache_data(ttl=600, show_spinner=False)
def _safe_multi_download(tickers, period, interval):
    if not tickers:
        return pd.DataFrame()
    try:
        return yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=False, group_by="ticker", threads=True, progress=False
        )
    except Exception as e:
        _log(f"yf.download multi failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def _safe_single_download(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception as e:
        _log(f"yf.download single failed {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
@_retry(n=3, wait=0.6)
def get_spot(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    # 1) fast_info
    try:
        fi = getattr(tk, "fast_info", {}) or {}
        v = fi.get("last_price") or fi.get("last_price_raw") or fi.get("regular_market_price")
        v = _first_float(v)
        if v is not None: return v
    except Exception as e:
        _log(f"{ticker} fast_info err: {e}")
    # 2) info
    try:
        inf = getattr(tk, "info", {}) or {}
        v = inf.get("regularMarketPrice") or inf.get("currentPrice")
        v = _first_float(v)
        if v is not None: return v
    except Exception as e:
        _log(f"{ticker} info err: {e}")
    # 3) history fallbacks
    for per, intrv in [("1d","1m"), ("5d","1d")]:
        try:
            px = tk.history(period=per, interval=intrv, auto_adjust=False)
            if not px.empty:
                if "Close" in px.columns and not px["Close"].dropna().empty:
                    return float(px["Close"].dropna().iloc[-1])
                if "close" in px.columns and not px["close"].dropna().empty:
                    return float(px["close"].dropna().iloc[-1])
        except Exception as e:
            _log(f"{ticker} hist {per}/{intrv} err: {e}")
    return None

@st.cache_data(ttl=600, show_spinner=False)
@_retry(n=3, wait=0.6)
def get_options(ticker: str) -> List[str]:
    opts = []
    tk = yf.Ticker(ticker)
    try:
        opts = tk.options or []
    except Exception as e:
        _log(f"{ticker} options err: {e}")
    # Validate yyyy-mm-dd strings only
    exp = [e for e in opts if isinstance(e, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", e)]
    if not exp:
        _log(f"{ticker} no expiries")
    return exp

@st.cache_data(ttl=480, show_spinner=False)
@_retry(n=3, wait=0.7)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        oc = yf.Ticker(ticker).option_chain(expiry)
        calls = oc.calls.copy() if hasattr(oc, "calls") else pd.DataFrame()
        puts  = oc.puts.copy()  if hasattr(oc, "puts")  else pd.DataFrame()
    except Exception as e:
        _log(f"{ticker} chain {expiry} err: {e}")
        return pd.DataFrame(), pd.DataFrame()
    # Normalize column names we rely on
    for df in (calls, puts):
        if not df.empty:
            if "lastPrice" not in df.columns and "last_price" in df.columns:
                df.rename(columns={"last_price":"lastPrice"}, inplace=True)
            if "impliedVolatility" not in df.columns and "implied_volatility" in df.columns:
                df.rename(columns={"implied_volatility":"impliedVolatility"}, inplace=True)
    if calls.empty and puts.empty:
        _log(f"{ticker} chain empty for {expiry}")
    return calls, puts

@st.cache_data(ttl=1800, show_spinner=False)
def get_next_earnings_date(ticker: str) -> Optional[date]:
    today = _now_pacific_date()
    tk = yf.Ticker(ticker)
    # Preferred: get_earnings_dates
    try:
        df = tk.get_earnings_dates(limit=10)
        if df is not None and not df.empty:
            col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
            dates = pd.to_datetime(df[col], utc=True, errors="coerce").dt.date
            fut = [d for d in dates if d and d >= today]
            if fut:
                return min(fut)
    except Exception as e:
        _log(f"{ticker} get_earnings_dates err: {e}")
    # Fallback: calendar/info
    try_sources = (getattr(tk, "calendar", None), getattr(tk, "info", {}), getattr(tk, "fast_info", {}))
    for src in try_sources:
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
# IV/FF calculations
# =========================
def _nearest_strike(series: pd.Series, target: float) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return None
    return float(min(s, key=lambda v: abs(v - target)))

def _safe_mid(row: pd.Series) -> Optional[float]:
    try:
        bid = _first_float(row.get("bid"))
        ask = _first_float(row.get("ask"))
        last = _first_float(row.get("lastPrice"))
    except Exception:
        return None
    if bid is not None and ask is not None:
        return 0.5*(bid+ask)
    return last

def _iv_from_df_at_strike(df: pd.DataFrame, strike: float) -> Optional[float]:
    if df is None or df.empty:
        return None
    strikes = pd.to_numeric(df["strike"], errors="coerce")
    # tolerant match
    idx = np.where(np.isclose(strikes, strike, rtol=0, atol=max(0.01, 0.002*max(1.0, strike))))[0]
    if len(idx) == 0:
        nearest = _nearest_strike(df["strike"], strike)
        if nearest is None:
            return None
        idx = np.where(np.isclose(strikes, nearest, rtol=0, atol=max(0.01, 0.002*max(1.0, nearest))))[0]
        if len(idx) == 0:
            return None
    row = df.iloc[idx[0]]
    iv = _first_float(row.get("impliedVolatility"))
    if iv is None:
        return None
    return float(iv)*100.0 if iv <= 1.5 else float(iv)

def atm_iv(ticker: str, expiry: str, spot: float) -> Optional[float]:
    calls, puts = get_chain(ticker, expiry)
    strikes = []
    for df in (calls, puts):
        if df is not None and not df.empty and "strike" in df.columns:
            strikes.extend(pd.to_numeric(df["strike"], errors="coerce").dropna().astype(float).tolist())
    if not strikes:
        _log(f"{ticker} no strikes for {expiry}")
        return None
    atm = min(strikes, key=lambda s: abs(s - spot))
    c_iv = _iv_from_df_at_strike(calls, atm)
    p_iv = _iv_from_df_at_strike(puts, atm)
    if c_iv is None and p_iv is None:
        _log(f"{ticker} no IV at ATM for {expiry}")
    return (0.5*(c_iv+p_iv) if (c_iv is not None and p_iv is not None) else (c_iv if c_iv is not None else p_iv))

def _strikes_from_chain(df: pd.DataFrame) -> set:
    if df is None or df.empty or "strike" not in df.columns:
        return set()
    s = pd.to_numeric(df["strike"], errors="coerce").dropna().astype(float)
    return set(s.tolist())

def common_atm_strike(ticker: str, exp1: str, exp2: str, spot: float) -> Optional[float]:
    c1, p1 = get_chain(ticker, exp1)
    c2, p2 = get_chain(ticker, exp2)
    s1 = _strikes_from_chain(c1) | _strikes_from_chain(p1)
    s2 = _strikes_from_chain(c2) | _strikes_from_chain(p2)
    inter = s1 & s2
    if not inter:
        if not s1 or not s2:
            _log(f"{ticker} no intersect strikes {exp1}/{exp2}")
            return None
        a = min(s1, key=lambda s: abs(s-spot))
        b = min(s2, key=lambda s: abs(s-spot))
        if abs(a-b) > max(0.1, 0.003*spot):
            _log(f"{ticker} ATM mismatch {exp1}/{exp2}")
            return None
        return float(0.5*(a+b))
    return float(min(inter, key=lambda s: abs(s - spot)))

def call_mid_at(calls: pd.DataFrame, strike: float) -> Optional[float]:
    if calls is None or calls.empty:
        return None
    strikes = pd.to_numeric(calls["strike"], errors="coerce")
    idx = np.where(np.isclose(strikes, strike, rtol=0, atol=max(0.01, 0.002*max(1.0, strike))))[0]
    if len(idx) == 0:
        return None
    return _safe_mid(calls.iloc[idx[0]])

def calendar_debit(ticker: str, e1: str, e2: str, spot: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    c1, _ = get_chain(ticker, e1)
    c2, _ = get_chain(ticker, e2)
    strike = common_atm_strike(ticker, e1, e2, spot)
    if strike is None:
        return None, None, None, None
    short_mid, long_mid = call_mid_at(c1, strike), call_mid_at(c2, strike)
    if short_mid is None or long_mid is None:
        return strike, short_mid, long_mid, None
    return strike, short_mid, long_mid, float(long_mid - short_mid)

def forward_and_ff(s1: float, T1: float, s2: float, T2: float):
    denom = T2 - T1
    if denom <= 0:
        return None, None
    fwd_var = (s2**2 * T2 - s1**2 * T1) / denom
    if fwd_var <= 0:
        return None, None
    fwd_sigma = math.sqrt(fwd_var)
    ff = None if fwd_sigma == 0 else (s1 - fwd_sigma) / fwd_sigma
    return fwd_sigma, ff

# =========================
# Screener core
# =========================
def _expiry_dtes(expiries: List[str]) -> List[Tuple[str, int]]:
    out = []
    for e in expiries:
        try:
            dte = _calc_dte(e)
            out.append((e, dte))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[1])

def _pick_next(ed: List[Tuple[str, int]], target_dte: int) -> Optional[Tuple[str, int]]:
    if not ed:
        return None
    for e in ed:
        if e[1] >= target_dte:
            return e
    return ed[-1]

def screen_ticker(ticker: str) -> List[Dict]:
    rows: List[Dict] = []

    spot = get_spot(ticker)
    expiries = get_options(ticker)

    # DEBUG surface
    st.session_state.setdefault("_diag_rows", [])
    st.session_state["_diag_rows"].append({
        "Ticker": ticker,
        "Spot?": spot is not None,
        "Exp count": len(expiries or []),
        "First 3 exp": ", ".join(expiries[:3]) if expiries else "",
    })

    if spot is None or not expiries:
        rows.append({
            "ticker": ticker, "pair": EM_DASH, "exp1": EM_DASH, "dte1": EM_DASH, "iv1": EM_DASH,
            "exp2": EM_DASH, "dte2": EM_DASH, "iv2": EM_DASH, "fwd_vol": EM_DASH, "ff": EM_DASH,
            "cal_debit": EM_DASH, "earn_in_window": EM_DASH,
            "_tags": [("no_spot" if spot is None else "no_exp")], "reason": ("no_spot" if spot is None else "no_exp")
        })
        return rows

    ed = _expiry_dtes(expiries)
    e7  = _pick_next(ed, 7)
    e14 = _pick_next(ed, 14)
    e30 = _pick_next(ed, 30)
    e60 = _pick_next(ed, 60)
    e90 = _pick_next(ed, 90)

    pairs: List[Tuple[str, Tuple[str,int], Tuple[str,int]]] = []
    def _add(label, a, b):
        if a and b and b[1] > a[1]:
            pairs.append((label, a, b))
    _add("7–14",  e7,  e14)
    _add("7–30",  e7,  e30)
    _add("30–60", e30, e60)
    _add("30–90", e30, e90)
    _add("60–90", e60, e90)

    earn_dt = get_next_earnings_date(ticker)

    if not pairs:
        rows.append({
            "ticker": ticker, "pair": EM_DASH,
            "exp1": EM_DASH, "dte1": EM_DASH, "iv1": EM_DASH,
            "exp2": EM_DASH, "dte2": EM_DASH, "iv2": EM_DASH,
            "fwd_vol": EM_DASH, "ff": EM_DASH, "cal_debit": EM_DASH,
            "earn_in_window": earn_dt.strftime("%Y-%m-%d") if earn_dt else EM_DASH,
            "_tags": ["no_pairs"], "reason": "no_pairs"
        })
        return rows

    for label, (exp1, dte1), (exp2, dte2) in pairs:
        tags: List[str] = []
        earn_txt = EM_DASH
        e1d, e2d = _expiry_to_date(exp1), _expiry_to_date(exp2)

        if earn_dt:
            earn_txt = earn_dt.strftime("%Y-%m-%d")
            if e1d and earn_dt < e1d:
                tags.append("blocked")
            elif e1d and e2d and e1d <= earn_dt <= e2d:
                tags.append("earn")

        iv1 = atm_iv(ticker, exp1, spot)
        iv2 = atm_iv(ticker, exp2, spot)

        # add diag line for IV reachability
        st.session_state.setdefault("_diag_iv", [])
        st.session_state["_diag_iv"].append({
            "Ticker": ticker, "Pair": label,
            "IV1?": iv1 is not None, "IV2?": iv2 is not None, "exp1": exp1, "exp2": exp2
        })

        fwd_txt, ff_txt = EM_DASH, EM_DASH
        reason = "ok"

        if iv1 is not None and iv2 is not None:
            s1, s2 = iv1/100.0, iv2/100.0
            T1, T2 = dte1/365.0, dte2/365.0
            fwd_sigma, ff = forward_and_ff(s1, T1, s2, T2)
            if fwd_sigma is not None:
                fwd_txt = _fmt_pct(fwd_sigma*100.0)
            if ff is not None:
                ff_txt = _fmt_pct(ff*100.0)
                if ff >= 0.20:
                    tags.append("hot")
        else:
            reason = "iv1_missing" if iv1 is None and iv2 is not None else (
                     "iv2_missing" if iv2 is None and iv1 is not None else "no_iv_both")
            if reason == "no_iv_both":
                tags.append("no_iv")

        _, _, _, debit = calendar_debit(ticker, exp1, exp2, spot)

        rows.append({
            "ticker": ticker, "pair": label,
            "exp1": exp1, "dte1": dte1, "iv1": _fmt_pct(iv1) if iv1 is not None else EM_DASH,
            "exp2": exp2, "dte2": dte2, "iv2": _fmt_pct(iv2) if iv2 is not None else EM_DASH,
            "fwd_vol": fwd_txt, "ff": ff_txt,
            "cal_debit": _fmt_money(debit),
            "earn_in_window": earn_txt,
            "_tags": tags, "reason": reason
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
                _log(f"scan error {t}: {e}")
                rows.append({"ticker": t, "pair": EM_DASH,"exp1": EM_DASH,"dte1": EM_DASH,"iv1": EM_DASH,
                             "exp2": EM_DASH,"dte2": EM_DASH,"iv2": EM_DASH,"fwd_vol": EM_DASH,"ff": EM_DASH,
                             "cal_debit": EM_DASH,"earn_in_window": EM_DASH,"_tags": [f"error:{type(e).__name__}"], "reason": "error"})
            done += 1
            progress.progress(done / len(tickers), text=f"Scanned {t} ({done}/{len(tickers)})")
            time.sleep(0.08)
    progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        _log("scan_many produced empty df")
    if "_tags" not in df.columns and not df.empty:
        df["_tags"] = [[] for _ in range(len(df))]
    return df

# =========================
# Volume ranking (kept simple in safe mode)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def _batch_3mo_stats(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker","avg_vol_3m","last_close"])
    raw = _safe_multi_download(tickers, period="3mo", interval="1d")
    rows = []

    if isinstance(getattr(raw, "columns", None), pd.MultiIndex):
        got_syms = [c[0] for c in raw.columns.unique(level=0)]
        for t in got_syms:
            try:
                sub = raw[t]
                sub = _exclude_today_if_open(sub)
                if sub is None or sub.empty or "Volume" not in sub.columns:
                    continue
                vols = sub["Volume"].dropna()
                if vols.empty:
                    continue
                avgv = float(vols.mean())
                last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
                rows.append({"ticker": t, "avg_vol_3m": avgv, "last_close": last_close})
            except Exception as e:
                _log(f"vol stats err {t}: {e}")
                continue
        missing = set(tickers) - set(got_syms)
        for t in list(missing):
            sub = _safe_single_download(t, period="3mo", interval="1d")
            if sub is None or sub.empty or "Volume" not in sub.columns:
                continue
            sub = _exclude_today_if_open(sub)
            vols = sub["Volume"].dropna()
            if vols.empty:
                continue
            avgv = float(vols.mean())
            last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
            rows.append({"ticker": t, "avg_vol_3m": avgv, "last_close": last_close})
    else:
        for t in tickers:
            sub = _safe_single_download(t, period="3mo", interval="1d")
            if sub is None or sub.empty or "Volume" not in sub.columns:
                continue
            sub = _exclude_today_if_open(sub)
            vols = sub["Volume"].dropna()
            if vols.empty:
                continue
            avgv = float(vols.mean())
            last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
            rows.append({"ticker": t, "avg_vol_3m": avgv, "last_close": last_close})

    return pd.DataFrame(rows)

@st.cache_data(ttl=600, show_spinner=False)
def rank_top27_by_3m_avg_with_etfs(extras: List[str], debug_mode: bool) -> pd.DataFrame:
    if debug_mode:
        # In safe mode, do not scan the universe — just return the debug tickers and ETFs
        rows = []
        # Attach minimal stats for display
        stats = _batch_3mo_stats(DEBUG_TICKERS + FIXED_ETFS)
        for t in DEBUG_TICKERS:
            r = stats.loc[stats["ticker"]==t]
            if not r.empty:
                rr = r.iloc[0]
                rows.append({"ticker": t, "avg_vol_3m": rr["avg_vol_3m"], "last_close": rr["last_close"], "source": "Stock"})
            else:
                rows.append({"ticker": t, "avg_vol_3m": None, "last_close": None, "source": "Stock"})
        for etf in FIXED_ETFS:
            r = stats.loc[stats["ticker"]==etf]
            if not r.empty:
                rr = r.iloc[0]
                rows.append({"ticker": etf, "avg_vol_3m": rr["avg_vol_3m"], "last_close": rr["last_close"], "source": "ETF"})
            else:
                rows.append({"ticker": etf, "avg_vol_3m": None, "last_close": None, "source": "ETF"})
        return pd.DataFrame(rows)

    universe = get_us_stock_universe()
    extras = [e.strip().upper() for e in extras if _is_clean_us_symbol(e)]
    for e in extras:
        if e not in universe:
            universe.append(e)
    stock_universe = [t for t in universe if t not in ETF_SET]
    all_rows: List[pd.DataFrame] = []
    progress = st.progress(0.0, text="Computing 3M average volume across universe…")
    total = max(1, len(stock_universe))
    processed = 0
    for block in _chunks(stock_universe, BATCH_3MO):
        dfb = _batch_3mo_stats(block)
        if not dfb.empty:
            all_rows.append(dfb)
        processed += len(block)
        progress.progress(min(1.0, processed/total), text=f"3M avg vol calc: {processed}/{total} symbols")
    progress.empty()

    if not all_rows:
        return pd.DataFrame(columns=["ticker","avg_vol_3m","last_close","source"])

    vol_df = pd.concat(all_rows, ignore_index=True).dropna(subset=["avg_vol_3m"])

    # Ensure extras are present
    missing_extras = [e for e in extras if e not in vol_df["ticker"].unique().tolist()]
    if missing_extras:
        extra_fix = _batch_3mo_stats(missing_extras)
        if not extra_fix.empty:
            vol_df = pd.concat([vol_df, extra_fix], ignore_index=True)

    vol_df = vol_df.drop_duplicates(subset=["ticker"], keep="first")
    vol_df.sort_values("avg_vol_3m", ascending=False, inplace=True)

    top_stocks = vol_df.head(TOP_STOCKS).copy()
    top_stocks["source"] = "Stock"

    etf_rows = []
    for etf in FIXED_ETFS:
        etf_df = _batch_3mo_stats([etf])
        if not etf_df.empty:
            row = etf_df.iloc[0]
            etf_rows.append({"ticker": etf, "avg_vol_3m": float(row["avg_vol_3m"]), "last_close": row["last_close"], "source": "ETF"})
        else:
            etf_rows.append({"ticker": etf, "avg_vol_3m": None, "last_close": None, "source": "ETF"})

    full_final = pd.concat([top_stocks, pd.DataFrame(etf_rows)], ignore_index=True).reset_index(drop=True)
    return full_final

# =========================
# Sorting helpers
# =========================
def _build_sort_columns(series: pd.Series) -> pd.DataFrame:
    s = series.copy()
    s_str = s.astype(str).str.strip()
    is_blank = s.isna() | s_str.isin(BLANKS)

    pct_val = pd.to_numeric(s_str.str.rstrip("%").str.replace(",", "", regex=False), errors="coerce")
    pct_mask = s_str.str.endswith("%") & ~pd.isna(pct_val)

    cur_mask = s_str.str.match(r"^\(?\$?\s*[\d,]+(?:\.\d+)?\)?$")
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

st.title("📈 Forward Volatility Screener — Safe Mode")
st.caption("If your normal scan renders blank, turn on Safe Mode to test with known-good tickers and show live diagnostics.")

col_top = st.columns([1,1,2])
with col_top[0]:
    debug_mode = st.toggle("Safe Mode (Debug)", value=True, help="Scan AAPL/MSFT/NVDA/SPY only and show diagnostics.")
with col_top[1]:
    clear_cache = st.button("Clear Cache", help="Force-refresh all cached data functions.")
with col_top[2]:
    raw_extra = st.text_input("Optional: Add tickers (comma/space separated)", "", placeholder="e.g., NVDA, TSLA, OPEN, META")

if clear_cache:
    # Bust all caches so we don't keep empty results
    st.cache_data.clear()
    st.success("Cleared cache. Re-run your scan.")

def trigger_run():
    st.session_state.trigger_run = True

st.button("Build List & Run", type="primary", help="Rank list (or use Safe Mode list) and scan options", on_click=trigger_run)

# Auto-run once on first visit
if st.session_state.first_visit and not st.session_state.trigger_run:
    st.session_state.trigger_run = True
    st.session_state.first_visit = False

# ---------- MAIN EXECUTION ----------
if st.session_state.trigger_run:
    extras = _normalize_tickers(raw_extra)

    with st.spinner("Preparing ticker list…"):
        vol_top_df = rank_top27_by_3m_avg_with_etfs(extras=extras, debug_mode=debug_mode)

    st.session_state.vol_topN = vol_top_df
    st.session_state.tickers = vol_top_df["ticker"].tolist() if not vol_top_df.empty else (DEBUG_TICKERS if debug_mode else [])

    if st.session_state.tickers:
        with st.spinner("Scanning option chains…"):
            # reset diagnostics each run
            st.session_state["_debug_logs"] = []
            st.session_state["_diag_rows"] = []
            st.session_state["_diag_iv"] = []
            st.session_state.df = scan_many(st.session_state.tickers)
        st.success("✅ Scan complete! Results displayed below.")
    else:
        st.session_state.df = pd.DataFrame()
        st.error("No tickers to scan. If not in Safe Mode, your 3M-volume ranking may have returned empty. Toggle Safe Mode.")

    st.session_state.trigger_run = False

# --------- Selected list (stocks + ETFs) ---------
if st.session_state.vol_topN is not None and not st.session_state.vol_topN.empty:
    st.subheader("Selected Tickers")
    base = st.session_state.vol_topN.copy()
    # Simple table for clarity
    disp = pd.DataFrame({
        "Ticker": base["ticker"],
        "Avg Vol (3m)": base["avg_vol_3m"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else EM_DASH),
        "Last Close": base["last_close"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else EM_DASH),
        "Source": base["source"].astype(str),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

# --------- Forward-vol table ---------
df_current = st.session_state.df
if df_current is None or df_current.empty:
    st.warning("No rows to display. See diagnostics below.")
else:
    # Diagnostics summary
    st.write({
        "tickers_selected": len(st.session_state.tickers or []),
        "df_rows": len(df_current),
        "rows_with_iv": int(((df_current["iv1"] != EM_DASH) & (df_current["iv2"] != EM_DASH)).sum()) if "iv1" in df_current and "iv2" in df_current else 0
    })

    display_labels = [DISPLAY_MAP[k] for k in DISPLAY_KEYS if k in df_current.columns]
    default_label = DISPLAY_MAP.get("ff", display_labels[0])
    c1, c2 = st.columns([3, 1])
    with c1:
        sort_label = st.selectbox("Sort by", options=display_labels,
                                  index=display_labels.index(default_label), key="sort_col_label")
    with c2:
        sort_ascending = st.toggle("Ascending", value=False, key="sort_asc")

    sort_key = LABEL_TO_KEY.get(sort_label, "ff")
    df_sorted = sort_df(df_current, sort_key, sort_ascending)

    base_cols = [k for k in DISPLAY_KEYS if k in df_sorted.columns]
    df_display = df_sorted[base_cols].copy()
    df_display.rename(columns={k: DISPLAY_MAP.get(k, k) for k in base_cols}, inplace=True)

    st.subheader("Results")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# --------- Diagnostics ---------
st.divider()
st.subheader("Diagnostics")

# High-level data reachability per ticker
diag_rows = pd.DataFrame(st.session_state.get("_diag_rows", []))
if not diag_rows.empty:
    st.markdown("**Data reachability (spot & expiries)**")
    st.dataframe(diag_rows, use_container_width=True, hide_index=True)
else:
    st.caption("No reachability diagnostics gathered yet.")

# IV reachability per pair
diag_iv = pd.DataFrame(st.session_state.get("_diag_iv", []))
if not diag_iv.empty:
    st.markdown("**IV reachability (per pair)**")
    st.dataframe(diag_iv, use_container_width=True, hide_index=True)

# Raw logs
logs = st.session_state.get("_debug_logs", [])
if logs:
    st.markdown("**Debug logs**")
    for line in logs[:400]:
        st.text(f"• {line}")

st.markdown(
    "<p style='text-align:center; font-size:14px; color:#888;'>Built with ❤️ for Skyler — Safe Mode surfaces where the data breaks so you can fix it fast.</p>",
    unsafe_allow_html=True,
)
