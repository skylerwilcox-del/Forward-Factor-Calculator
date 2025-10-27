import math
from datetime import datetime, date, time, timedelta
from typing import Optional, Tuple, List, Dict
import re
import zoneinfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# App/session setup
# =========================
PACIFIC = zoneinfo.ZoneInfo("America/Los_Angeles")
EASTERN = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

st.set_page_config(page_title="Forward Vol Screener — Top 27 (3M Avg Vol, All US Stocks)", layout="wide")

# Persistent session state
st.session_state.setdefault("df", None)
st.session_state.setdefault("tickers", [])
st.session_state.setdefault("vol_topN", None)
st.session_state.setdefault("trigger_run", False)

# --- Config ---
TOP_STOCKS = 27
FIXED_ETFS = ["SPY", "QQQ", "VOO"]
ETF_SET = set(FIXED_ETFS)

# Download tuning
BATCH_3MO = 60
MAX_WORKERS = 12

# Large liquid fallback universe
FALLBACK_LIQUID = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","AMD","NFLX","CRM","COST","PEP","ADBE","INTC",
    "UBER","QCOM","AMAT","TSM","ORCL","PFE","MRK","JNJ","LLY","V","MA","JPM","BAC","WFC","C","GS","MS","SCHW","BLK",
    "GE","HON","CAT","BA","NKE","KO","PG","MCD","WMT","HD","LOW","TGT","DIS","CMCSA","T","VZ","TMUS","CSCO",
    "PANW","NOW","SNOW","DDOG","NET","MDB","ZS","PLTR","SHOP","ABNB","PYPL","SQ","ROKU","TTD","SPOT","DKNG","RBLX",
    "MU","NXPI","TXN","ADI","ON","LRCX","KLAC","ASML","BABA","BIDU","PDD","RIO","BHP","CVX","XOM","COP","OXY","SLB",
    "F","GM","RIVN","LCID","NIO","CCL","AAL","UAL","DAL","UPS","FDX","SOFI","COIN","HOOD","GME","OPEN"
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

def _market_close_et(d: date) -> datetime:
    return datetime.combine(d, time(16, 0, 0)).replace(tzinfo=EASTERN)

def _normalize_tickers(raw: str) -> List[str]:
    seen, out = set(), []
    for t in raw.replace(",", " ").split():
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _is_clean_us_symbol(sym: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?", sym))

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# yfinance (cached)
# =========================
def _cache_wrapper(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False)

@_cache_wrapper(1800)
def get_us_stock_universe() -> List[str]:
    candidates: List[str] = []
    for fn_name in ["tickers_nasdaq", "tickers_other", "tickers_sp500"]:
        try:
            fn = getattr(yf, fn_name, None)
            if callable(fn):
                vals = fn()
                if isinstance(vals, (list, tuple)):
                    candidates.extend([str(t).strip().upper() for t in vals if t])
        except Exception:
            continue
    candidates = [t for t in dict.fromkeys(candidates) if _is_clean_us_symbol(t)]
    if not candidates:
        candidates = FALLBACK_LIQUID.copy()
    for t in list(ETF_SET) + ["OPEN"]:
        if t not in candidates and _is_clean_us_symbol(t):
            candidates.append(t)
    return list(dict.fromkeys(candidates))

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

@_cache_wrapper(1200)
def _safe_multi_download(tickers, period, interval):
    if not tickers:
        return pd.DataFrame()
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

@_cache_wrapper(1200)
def _safe_single_download(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()

@_cache_wrapper(1800)
def get_spot(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    spot = tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice")
    if spot is None:
        px = tk.history(period="1d")
        if not px.empty:
            spot = float(px["Close"].iloc[-1])
    return _first_float(spot)

@_cache_wrapper(1200)
def get_options(ticker: str) -> List[str]:
    try:
        opts = yf.Ticker(ticker).options or []
        return [e for e in opts if isinstance(e, str) and len(e) == 10 and e[4] == "-" and e[7] == "-"]
    except Exception:
        return []

@_cache_wrapper(900)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oc = yf.Ticker(ticker).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

# =========================
# Earnings — robust next-event fetch and classification
# =========================
def _as_et(dt: pd.Timestamp | datetime | str | date) -> Optional[datetime]:
    """Convert to ET; if value appears date-only, return None (caller will expand proxies)."""
    try:
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return None
        if isinstance(dt, str):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dt):
                return None
            ts = pd.to_datetime(dt, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.tz_convert("America/New_York").to_pydatetime()
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None:
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return None
                dt = dt.tz_localize(UTC)
            else:
                dt = dt.tz_convert(UTC)
            return dt.tz_convert(EASTERN).to_pydatetime()
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return None
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(EASTERN)
    except Exception:
        return None
    return None

def _expand_proxies_for_date(d: date) -> List[datetime]:
    """BMO and AMC proxies for date-only values (ET)."""
    return [
        datetime.combine(d, time(8,30)).replace(tzinfo=EASTERN),   # BMO
        datetime.combine(d, time(16,1)).replace(tzinfo=EASTERN),   # AMC
    ]

def _extract_dates_from_get_earnings_dates(df: pd.DataFrame) -> List[datetime]:
    """
    yfinance may return:
    - DatetimeIndex named 'Earnings Date'
    - or a column 'Earnings Date'
    Normalize both to ET datetimes (with None meaning date-only → handled by caller).
    """
    out: List[datetime] = []
    if df is None or df.empty:
        return out

    # 1) Index path (most common)
    if isinstance(df.index, (pd.DatetimeIndex, pd.Index)) and str(getattr(df.index, "name", "")).lower().startswith("earnings"):
        for val in df.index.tolist():
            # Keep raw; caller will run _as_et (which returns None for date-only midnight)
            out.append(val)

    # 2) Column path (older variants / some forks)
    if "Earnings Date" in df.columns:
        out.extend(df["Earnings Date"].tolist())

    # Dedup while preserving order
    seen = set()
    uniq = []
    for v in out:
        key = str(v)
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq

@_cache_wrapper(900)
def get_next_earnings_event_et(ticker: str) -> Optional[datetime]:
    """
    Returns the earliest earnings datetime in ET that is >= (today ET - 1 day).
    Gathers from get_earnings_dates (index or column), calendar, and info/fast_info; robust to date-only.
    """
    tk = yf.Ticker(ticker)
    now_et = datetime.now(EASTERN)
    floor = (now_et - timedelta(days=1)).replace(microsecond=0)

    candidates: List[datetime] = []

    # 1) get_earnings_dates (index or column)
    try:
        df = tk.get_earnings_dates(limit=24)
        vals = _extract_dates_from_get_earnings_dates(df) if df is not None else []
        for val in vals:
            dt_et = _as_et(val)
            if dt_et is not None:
                candidates.append(dt_et.replace(microsecond=0))
            else:
                # date-only → add BMO/AMC proxies
                d = (val.date() if isinstance(val, (pd.Timestamp, datetime)) else
                     val if isinstance(val, date) else
                     pd.to_datetime(val, errors="coerce").date())
                if isinstance(d, date):
                    for p in _expand_proxies_for_date(d):
                        candidates.append(p.replace(microsecond=0))
    except Exception:
        pass

    # 2) calendar (dict or df-like)
    try:
        cal = tk.calendar
        candidate = None
        # Some yfinance builds put the date in index, others as dict-like
        if hasattr(cal, "index"):
            # Try index value by common keys
            for key in ["Earnings Date","EarningsDate","Earnings"]:
                if key in list(cal.index):
                    candidate = cal.loc[key].values[0]
                    break
        if candidate is None and isinstance(cal, dict):
            candidate = cal.get("Earnings Date") or cal.get("EarningsDate")
        if candidate is not None:
            dt_et = _as_et(candidate)
            if dt_et is not None:
                candidates.append(dt_et.replace(microsecond=0))
            else:
                d = (candidate.date() if isinstance(candidate, (pd.Timestamp, datetime)) else
                     candidate if isinstance(candidate, date) else
                     pd.to_datetime(candidate, errors="coerce").date())
                if isinstance(d, date):
                    for p in _expand_proxies_for_date(d):
                        candidates.append(p.replace(microsecond=0))
    except Exception:
        pass

    # 3) info/fast_info
    for src in [tk.info, getattr(tk, "fast_info", {})]:
        if not isinstance(src, dict):
            continue
        for key in ("Earnings Date", "earningsDate", "earnings_date", "nextEarningsDate"):
            try:
                val = src.get(key)
                if val is None:
                    continue
                if isinstance(val, (list, tuple, pd.Series, pd.Index)) and len(val) > 0:
                    val = val[0]
                dt_et = _as_et(val)
                if dt_et is not None:
                    candidates.append(dt_et.replace(microsecond=0))
                else:
                    d = (val.date() if isinstance(val, (pd.Timestamp, datetime)) else
                         val if isinstance(val, date) else
                         pd.to_datetime(val, errors="coerce").date())
                    if isinstance(d, date):
                        for p in _expand_proxies_for_date(d):
                            candidates.append(p.replace(microsecond=0))
            except Exception:
                continue

    if not candidates:
        return None

    fut = [c for c in candidates if c >= floor]
    if not fut:
        return None
    return sorted(set(fut))[0]

def earnings_label_for_pair(exp1: str, exp2: str, next_evt_et: Optional[datetime]) -> Optional[str]:
    """Return 'YYYY-MM-DD BMO/AMC/? ≤Exp1|Exp1–Exp2|>Exp2' or None."""
    if next_evt_et is None:
        return None
    d1 = _expiry_to_date(exp1)
    d2 = _expiry_to_date(exp2)
    if not d1 or not d2:
        return None

    # infer TOD label
    h = next_evt_et.hour + next_evt_et.minute/60
    tod = "BMO" if h < 9.5 else ("MID" if h < 16.0 else "AMC")
    base = f"{next_evt_et.date()} {tod}"

    if next_evt_et <= _market_close_et(d1):
        return f"{base} ≤Exp1"
    if next_evt_et <= _market_close_et(d2):
        return f"{base} Exp1–Exp2"
    return f"{base} >Exp2"

# =========================
# 3M Avg Volume ranking
# =========================
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
            except Exception:
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

@_cache_wrapper(1200)
def rank_top27_by_3m_avg_with_etfs(extras: List[str]) -> pd.DataFrame:
    universe = get_us_stock_universe()

    extras = [e.strip().upper() for e in extras if _is_clean_us_symbol(e)]
    for e in extras:
        if e not in universe:
            universe.append(e)

    stock_universe = [t for t in universe if t not in ETF_SET]

    all_rows: List[pd.DataFrame] = []
    progress = st.progress(0.0, text="Computing 3M average volume across entire universe…")
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
# IV/FF calculations
# =========================
@_cache_wrapper(900)
def atm_iv(ticker: str, expiry: str, spot: float) -> Optional[float]:
    calls, puts = get_chain(ticker, expiry)
    if calls.empty and puts.empty:
        return None
    strikes = pd.Index(sorted(set(calls["strike"]).union(set(puts["strike"])))) if not (calls.empty and puts.empty) else pd.Index([])
    if len(strikes) == 0:
        return None
    strikes = strikes.astype(float)
    atm = float(min(strikes, key=lambda s: abs(s - spot)))
    def iv_from(df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        row = df.loc[df["strike"].astype(float) == atm]
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
    s1 = set(map(float, pd.Index(sorted(set(c1["strike"]).union(set(p1["strike"])))))) if not (c1.empty and p1.empty) else set()
    s2 = set(map(float, pd.Index(sorted(set(c2["strike"]).union(set(p2["strike"])))))) if not (c2.empty and p2.empty) else set()
    inter = list(s1.intersection(s2))
    return float(min(inter, key=lambda s: abs(s - spot))) if inter else None

def call_mid_at(calls: pd.DataFrame, strike: float) -> Optional[float]:
    if calls is None or calls.empty:
        return None
    row = calls.loc[calls["strike"].astype(float) == strike]
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
# Screener
# =========================
def _nearest(ed, target):
    return min(ed, key=lambda x: (abs(x[1] - target), x[1])) if ed else None

@_cache_wrapper(900)
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

    # Anchors
    e7, e14, e30, e60, e90 = _nearest(ed, 7), _nearest(ed, 14), _nearest(ed, 30), _nearest(ed, 60), _nearest(ed, 90)

    pairs = []
    def add_pair(name, a, b):
        if a and b and b[1] > a[1]:
            pairs.append((name, a, b))
    add_pair("7–14", e7, e14)
    add_pair("7–30", e7, e30)
    add_pair("30–60", e30, e60)
    add_pair("30–90", e30, e90)
    add_pair("60–90", e60, e90)

    # Next earnings (always displayed)
    next_evt = get_next_earnings_event_et(ticker)

    rows = []
    for label, (exp1, dte1), (exp2, dte2) in pairs:
        iv1, iv2 = atm_iv(ticker, exp1, spot), atm_iv(ticker, exp2, spot)
        if iv1 is None or iv2 is None:
            continue
        s1, s2 = iv1/100.0, iv2/100.0
        T1, T2 = dte1/365.0, dte2/365.0
        fwd_sigma, ff = forward_and_ff(s1, T1, s2, T2)
        _, _, _, debit = calendar_debit(ticker, exp1, exp2, spot)

        earn_display = earnings_label_for_pair(exp1, exp2, next_evt)
        tags = []
        if earn_display and ("≤Exp1" in earn_display or "Exp1–Exp2" in earn_display):
            tags.append("earn")
        if ff is not None and ff >= 0.20:
            tags.append("hot")

        rows.append({
            "ticker": ticker, "pair": label,
            "exp1": exp1, "dte1": dte1, "iv1": f"{iv1:.2f}%",
            "exp2": exp2, "dte2": dte2, "iv2": f"{iv2:.2f}%",
            "fwd_vol": f"{(fwd_sigma*100):.2f}%" if fwd_sigma is not None else "—",
            "ff": f"{(ff*100):.2f}%" if ff is not None else "—",
            "cal_debit": f"{debit:.2f}" if debit is not None else "—",
            "earn_in_window": earn_display if earn_display else "—",
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
# Sorting helpers (stable & robust)
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
st.title("📈 Forward Volatility Screener (Top 27 by 3M Avg Volume — All US Stocks)")
st.markdown(
    "Evaluates **all NASDAQ + NYSE/AMEX** stocks by **average daily volume over the last 3 months** (no early gating), "
    f"selects **Top {TOP_STOCKS} stocks**, then adds **SPY, QQQ, VOO**."
)

refresh = st.checkbox("Force refresh data (bust caches)", value=False, help="Use if earnings dates look stale.")
if refresh:
    st.cache_data.clear()

raw_extra = st.text_input("Optional: Add tickers (comma/space separated)", "", placeholder="e.g., NVDA, TSLA, OPEN, META")

def trigger_run():
    st.session_state.trigger_run = True

colA, colB = st.columns([1, 3])
with colA:
    st.button("Build List & Run", type="primary", help="Rank by 3M avg volume, then scan options", on_click=trigger_run)
with colB:
    st.caption("Excludes today's partial volume until after ~4:05pm ET close.")

# ---------- MAIN EXECUTION ----------
if st.session_state.trigger_run:
    extras = _normalize_tickers(raw_extra)

    with st.spinner("Building ranked list (3M avg volume across entire universe)…"):
        vol_top_df = rank_top27_by_3m_avg_with_etfs(extras=extras)

    st.session_state.vol_topN = vol_top_df
    st.session_state.tickers = vol_top_df["ticker"].tolist() if not vol_top_df.empty else []

    if st.session_state.tickers:
        with st.spinner("Scanning option chains…"):
            st.session_state.df = scan_many(st.session_state.tickers)
        st.success("✅ Scan complete! Results displayed below.")
    else:
        st.session_state.df = pd.DataFrame()
        st.warning("No tickers selected by the ranking step.")

    st.session_state.trigger_run = False

# --------- Selected list (stocks + ETFs) ---------
if st.session_state.vol_topN is not None and not st.session_state.vol_topN.empty:
    st.subheader(f"Selected Tickers (Top {TOP_STOCKS} Stocks by 3M Avg Vol + ETFs)")
    base = st.session_state.vol_topN.copy()
    base["AvgVolNum"] = pd.to_numeric(base["avg_vol_3m"], errors="coerce")
    base = base.sort_values(by=["source", "AvgVolNum"], ascending=[True, False], kind="mergesort")
    disp = pd.DataFrame({
        "Ticker": base["ticker"],
        "Avg Vol (3m)": base["AvgVolNum"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—"),
        "Last Close": base["last_close"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Source": base["source"].astype(str),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

# --------- Forward-vol table ---------
df_current = st.session_state.df
if df_current is None or df_current.empty:
    st.info("Click **Build List & Run** to rank the entire US universe and scan options.")
else:
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
            color = "#ffe0b2"  # both
        elif earn:
            color = "#fff9c4"  # earnings ≤ Exp2
        elif hot:
            color = "#dcedc8"  # FF >= 0.20
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
    st.caption("Legend:  ≤Exp1 = before short leg;  Exp1–Exp2 = between legs;  >Exp2 = after long leg.  🟩 FF ≥ 0.20  🟨 Earnings ≤Exp2")

st.markdown(
    "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b> with GPT-5</p>",
    unsafe_allow_html=True,
)
