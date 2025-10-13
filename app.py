import math
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# =========================
# App/session setup
# =========================
PACIFIC = pytz.timezone("America/Los_Angeles")
EASTERN = pytz.timezone("America/New_York")
st.set_page_config(page_title="Forward Vol Screener — Top 27 (All US Stocks)", layout="wide")

# Persistent session state
st.session_state.setdefault("df", None)
st.session_state.setdefault("tickers", [])
st.session_state.setdefault("vol_topN", None)
st.session_state.setdefault("trigger_run", False)

# --- Config ---
TOP_STOCKS = 27
ADD_ETFS = True
FIXED_ETFS = ["VOO", "SPY", "QQQ"]
MAX_WORKERS = 12

# Download tuning
DL_CHUNK_STAGE1 = 80
DL_CHUNK_STAGE2 = 50
SHORTLIST_SIZE = 400

# A hard fallback list of liquid US tickers (ensures app never returns empty)
FALLBACK_LIQUID = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","AMD","NFLX","CRM","COST","PEP",
    "ADBE","INTC","UBER","JPM","BAC","XOM","CVX","KO","PFE","WMT","T","ORCL","QCOM","V","MA","PYPL",
    "WFC","GE","NKE","MRNA","DIS","BA","MU","PLTR","SOFI","CCL","NIO","RIVN","LCID","F","GM","SNAP",
    "SQ","SHOP","ABNB","MMM","IBM","VZ","CSCO","AMAT","TSM","BABA","BIDU","PDD","RIO","TQQQ","SQQQ"
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

def _normalize_tickers(raw: str) -> List[str]:
    seen, out = set(), []
    for t in raw.replace(",", " ").split():
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _is_clean_us_symbol(sym: str) -> bool:
    # Allow letters/digits and a single hyphen; exclude weird/OTC/indices/units, etc.
    # (yfinance can choke on many exotic/non-US formats)
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?", sym))

# =========================
# yfinance (cached data)
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
        # yfinance returns 'YYYY-MM-DD' strings
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
# Universe and downloads
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def get_us_stock_universe() -> List[str]:
    candidates: List[str] = []
    # Try yfinance helpers (when available in the runtime)
    for fn_name in ["tickers_nasdaq", "tickers_other", "tickers_sp500"]:
        try:
            fn = getattr(yf, fn_name, None)
            if callable(fn):
                vals = fn()
                if isinstance(vals, (list, tuple)):
                    candidates.extend([str(t).strip().upper() for t in vals if t])
        except Exception:
            continue

    # De-dup and sanitize
    candidates = [t for t in dict.fromkeys(candidates) if _is_clean_us_symbol(t)]

    # Hard fallback if nothing came back (or if environment blocks those helpers)
    if not candidates:
        candidates = FALLBACK_LIQUID.copy()

    # Make sure it’s not empty
    return list(dict.fromkeys(candidates))

def _exclude_today_if_open(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- Two-stage ranker ----------
def rank_top_stocks_all_us_pipeline(extras: List[str], top_n: int, add_etfs: bool,
                                    update_stage1=None, update_stage2=None) -> pd.DataFrame:
    universe = get_us_stock_universe()
    # Add any user extras
    for e in extras:
        e = e.strip().upper()
        if _is_clean_us_symbol(e) and e not in universe:
            universe.append(e)

    # Stage 1 — shortlist by last session volume
    stage1_rows = []
    processed = 0
    total = len(universe)
    for block in _chunks(universe, DL_CHUNK_STAGE1):
        data = _safe_multi_download(block, period="7d", interval="1d")
        got = [c[0] for c in getattr(data, "columns", pd.Index([])).unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
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
        if callable(update_stage1):
            update_stage1(processed, total)

    # If Stage 1 somehow empty, hard fallback to our liquid list and retry once
    if not stage1_rows:
        for block in _chunks(FALLBACK_LIQUID, DL_CHUNK_STAGE1):
            data = _safe_multi_download(block, period="7d", interval="1d")
            if data is None or data.empty:
                continue
            got = [c[0] for c in data.columns.unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
            for t in got:
                try:
                    sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                    sub = _exclude_today_if_open(sub)
                    if "Volume" not in sub.columns or sub["Volume"].dropna().empty:
                        continue
                    last_vol = float(sub["Volume"].dropna().iloc[-1])
                    last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
                    stage1_rows.append({"ticker": t, "last_vol": last_vol, "last_close": last_close})
                except Exception:
                    continue

    if not stage1_rows:
        # still empty; return ETFs only so the app shows *something*
        etf_rows = []
        for etf in FIXED_ETFS:
            try:
                hist = yf.download(etf, period="20d", interval="1d", auto_adjust=False, progress=False)
                hist = _exclude_today_if_open(hist)
                avgv = _avg5(hist["Volume"]) if "Volume" in hist.columns else None
                last_close = float(hist["Close"].dropna().iloc[-1]) if "Close" in hist.columns and not hist["Close"].dropna().empty else None
                etf_rows.append({"ticker": etf, "avg_week_volume": avgv, "last_close": last_close, "source": "ETF"})
            except Exception:
                continue
        return pd.DataFrame(etf_rows)

    stage1 = pd.DataFrame(stage1_rows).dropna(subset=["last_vol"]).sort_values("last_vol", ascending=False)
    shortlist = stage1["ticker"].head(SHORTLIST_SIZE).tolist()

    # Stage 2 — accurate 5-day average on shortlist
    stage2_rows = []
    processed = 0
    total2 = len(shortlist)
    for block in _chunks(shortlist, DL_CHUNK_STAGE2):
        data = _safe_multi_download(block, period="20d", interval="1d")
        got = [c[0] for c in getattr(data, "columns", pd.Index([])).unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
        for t in got:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if sub is None or sub.empty or "Volume" not in sub.columns:
                    continue
                sub = _exclude_today_if_open(sub)
                avgv = _avg5(sub["Volume"]) if "Volume" in sub.columns else None
                if avgv is None:
                    continue
                last_close = float(sub["Close"].dropna().iloc[-1]) if "Close" in sub.columns and not sub["Close"].dropna().empty else None
                stage2_rows.append({"ticker": t, "avg_week_volume": avgv, "last_close": last_close, "source": "Stock"})
            except Exception:
                continue
        processed += len(block)
        if callable(update_stage2):
            update_stage2(processed, total2)

    vol_df = pd.DataFrame(stage2_rows)
    if vol_df.empty:
        # as a fallback, compute 5d on FALLBACK_LIQUID
        for block in _chunks(FALLBACK_LIQUID, DL_CHUNK_STAGE2):
            data = _safe_multi_download(block, period="20d", interval="1d")
            if data is None or data.empty:
                continue
            got = [c[0] for c in data.columns.unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
            for t in got:
                try:
                    sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                    sub = _exclude_today_if_open(sub)
                    if "Volume" not in sub.columns:
                        continue
                    avgv = _avg5(sub["Volume"])
                    if avgv is None:
                        continue
                    last_close = float(sub["Close"].dropna().iloc[-1])
                    vol_df = pd.concat([vol_df, pd.DataFrame([{
                        "ticker": t, "avg_week_volume": avgv, "last_close": last_close, "source": "Stock"
                    }])], ignore_index=True)
                except Exception:
                    continue

    if vol_df.empty:
        return pd.DataFrame(columns=["ticker","avg_week_volume","last_close","source"])

    vol_df = vol_df.dropna(subset=["avg_week_volume"]).sort_values("avg_week_volume", ascending=False)
    top_stocks = vol_df.head(top_n).copy()

    if ADD_ETFS:
        etf_rows = []
        for etf in FIXED_ETFS:
            try:
                hist = yf.download(etf, period="20d", interval="1d", auto_adjust=False, progress=False)
                hist = _exclude_today_if_open(hist)
                avgv = _avg5(hist["Volume"]) if "Volume" in hist.columns else None
                last_close = float(hist["Close"].dropna().iloc[-1]) if "Close" in hist.columns and not hist["Close"].dropna().empty else None
                etf_rows.append({"ticker": etf, "avg_week_volume": avgv, "last_close": last_close, "source": "ETF"})
            except Exception:
                etf_rows.append({"ticker": etf, "avg_week_volume": None, "last_close": None, "source": "ETF"})
        top_full = pd.concat([top_stocks, pd.DataFrame(etf_rows)], ignore_index=True)
    else:
        top_full = top_stocks

    return top_full.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

# =========================
# IV/FF calculations
# =========================
def atm_iv(ticker: str, expiry: str, spot: float) -> Optional[float]:
    calls, puts = get_chain(ticker, expiry)
    if calls.empty and puts.empty:
        return None
    strikes = pd.Index(sorted(set(calls["strike"]).union(set(puts["strike"])))).astype(float)
    if len(strikes) == 0:
        return None
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
# Screener (per ticker)
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
        if ff is not None and ff >= 0.20:
            tags.append("hot")
        rows.append({
            "ticker": ticker, "pair": label,
            "exp1": exp1, "dte1": dte1, "iv1": f"{iv1:.2f}%",
            "exp2": exp2, "dte2": dte2, "iv2": f"{iv2:.2f}%",
            "fwd_vol": f"{(fwd_sigma*100):.2f}%" if fwd_sigma is not None else "—",
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
# Sorting helpers
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
    "Two-stage scan across **NASDAQ + NYSE/AMEX**. Ranks by **avg volume over the last 5 completed sessions**, "
    f"selects the **Top {TOP_STOCKS} stocks**, and {'adds **VOO, SPY, QQQ**' if ADD_ETFS else 'does not add ETFs'}."
)

raw_extra = st.text_input("Optional: Add tickers (comma/space separated)", "", placeholder="e.g., NVDA, TSLA, META")

def trigger_run():
    st.session_state.trigger_run = True

colA, colB = st.columns([1, 3])
with colA:
    st.button("Build List & Run", type="primary", help="Two-stage fast scan across the US universe", on_click=trigger_run)
with colB:
    st.caption("Excludes today's partial volume until after the 4:05pm ET close.")

# ---------- MAIN EXECUTION ----------
if st.session_state.trigger_run:
    extras = _normalize_tickers(raw_extra)

    prog1 = st.progress(0.0, text="Stage 1/2: Scanning universe (latest volumes)…")
    prog2 = st.progress(0.0, text="Stage 2/2: Computing 5-day average on shortlist…")

    def up1(done, total):
        prog1.progress(min(done/total, 1.0), text=f"Stage 1/2: {done}/{total} tickers processed…")
    def up2(done, total):
        prog2.progress(min(done/total, 1.0), text=f"Stage 2/2: {done}/{total} tickers processed…")

    with st.spinner("Building list…"):
        vol_top_df = rank_top_stocks_all_us_pipeline(
            extras=extras,
            top_n=TOP_STOCKS,
            add_etfs=ADD_ETFS,
            update_stage1=up1,
            update_stage2=up2,
        )

    prog1.empty()
    prog2.empty()

    st.session_state.vol_topN = vol_top_df
    st.session_state.tickers = vol_top_df["ticker"].tolist() if not vol_top_df.empty else []

    if st.session_state.tickers:
        with st.spinner("Scanning option chains…"):
            st.session_state.df = scan_many(st.session_state.tickers)
        st.success("✅ Scan complete! Results displayed below.")
    else:
        st.session_state.df = pd.DataFrame()
        st.warning("No tickers selected by the ranking step (API limits or market closed). Using the input box above can help.")

    st.session_state.trigger_run = False

# --------- Selected list (stocks + optional ETFs) ---------
if st.session_state.vol_topN is not None and not st.session_state.vol_topN.empty:
    st.subheader(f"Selected Tickers (Top {TOP_STOCKS} Stocks by 5-Day Avg Vol{' + ETFs' if ADD_ETFS else ''})")
    base = st.session_state.vol_topN.copy()
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

# Footer
st.markdown(
    "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b> with GPT-5</p>",
    unsafe_allow_html=True,
)
