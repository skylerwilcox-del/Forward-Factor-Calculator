import math
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict
import re

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
BATCH_3MO = 60          # batch size for 3m multi-download (balance speed & reliability)
MAX_WORKERS = 12        # for option-chain scanning only

# Large liquid fallback universe (used if ticker universe lookup fails)
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

def _normalize_tickers(raw: str) -> List[str]:
    seen, out = set(), []
    for t in raw.replace(",", " ").split():
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _is_clean_us_symbol(sym: str) -> bool:
    # Allow up to 5 letters/digits (+ optional hyphen suffix). Skip obvious non-US/OTC prefixes.
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?", sym))

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# yfinance (cached)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
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
    # sanitize
    candidates = [t for t in dict.fromkeys(candidates) if _is_clean_us_symbol(t)]
    if not candidates:
        candidates = FALLBACK_LIQUID.copy()
    # Always ensure FIXED_ETFS and a few known actives are present if the API misses them
    for t in list(ETF_SET) + ["OPEN"]:
        if t not in candidates and _is_clean_us_symbol(t):
            candidates.append(t)
    return list(dict.fromkeys(candidates))

def _exclude_today_if_open(df: pd.DataFrame) -> pd.DataFrame:
    """Remove today's partial bar until after ~4:05pm ET, so averages are comparable."""
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

@st.cache_data(ttl=1200, show_spinner=False)
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

@st.cache_data(ttl=1200, show_spinner=False)
def _safe_single_download(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_spot(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    spot = tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice")
    if spot is None:
        px = tk.history(period="1d")
        if not px.empty:
            spot = float(px["Close"].iloc[-1])
    return _first_float(spot)

@st.cache_data(ttl=1200, show_spinner=False)
def get_options(ticker: str) -> List[str]:
    try:
        opts = yf.Ticker(ticker).options or []
        return [e for e in opts if isinstance(e, str) and len(e) == 10 and e[4] == "-" and e[7] == "-"]
    except Exception:
        return []

@st.cache_data(ttl=900, show_spinner=False)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oc = yf.Ticker(ticker).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

@st.cache_data(ttl=3600, show_spinner=False)
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
# 3M Avg Volume ranking — NO GATING
# =========================
def _batch_3mo_stats(tickers: List[str]) -> pd.DataFrame:
    """
    Download 3mo daily data for a batch of tickers and compute:
      - avg_vol_3m (excluding today's partial bar)
      - last_close (from the same dataset)
    Returns rows only for tickers with valid volume history.
    """
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
        # Per-ticker fallback for those yfinance skipped in the batch response
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
        # Degenerate single-frame: try each individually
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

@st.cache_data(ttl=1200, show_spinner=False)
def rank_top27_by_3m_avg_with_etfs(extras: List[str]) -> pd.DataFrame:
    """
    Evaluate the ENTIRE US universe (NASDAQ + NYSE/AMEX) in 3M average volume
    using batched downloads — no Stage-1 gating. Extras are always included.
    """
    universe = get_us_stock_universe()

    # Add user extras explicitly (and sanitize)
    extras = [e.strip().upper() for e in extras if _is_clean_us_symbol(e)]
    for e in extras:
        if e not in universe:
            universe.append(e)

    # Remove fixed ETFs from the stock universe; they get appended later as ETFs
    stock_universe = [t for t in universe if t not in ETF_SET]

    # Iterate ENTIRE universe in batches; build a big 3m stats table
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
    # Guarantee extras are present even if yfinance missed a block (try single fetch)
    missing_extras = [e for e in extras if e not in vol_df["ticker"].unique().tolist()]
    if missing_extras:
        extra_fix = _batch_3mo_stats(missing_extras)
        if not extra_fix.empty:
            vol_df = pd.concat([vol_df, extra_fix], ignore_index=True)

    # Rank by 3m average volume (descending)
    vol_df = vol_df.drop_duplicates(subset=["ticker"], keep="first")
    vol_df.sort_values("avg_vol_3m", ascending=False, inplace=True)

    # Take top 27 stocks
    top_stocks = vol_df.head(TOP_STOCKS).copy()
    top_stocks["source"] = "Stock"

    # Append ETFs (always include)
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
# Screener
# =========================
@st.cache_data(ttl=900, show_spinner=False)
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

st.title("📈 Forward Volatility Screener (Top 27 by 3M Avg Volume — All US Stocks)")
st.markdown(
    "Evaluates **all NASDAQ + NYSE/AMEX** stocks by **average daily volume over the last 3 months** (no early gating), "
    f"selects **Top {TOP_STOCKS} stocks**, then adds **SPY, QQQ, VOO**."
)

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
    source_order = pd.api.types.CategoricalDtype(categories=["Stock", "ETF"], ordered=True)
    base["Source"] = base["source"].astype(source_order)
    base["SourceOrder"] = base["Source"].cat.codes
    base["AvgVolNum"] = pd.to_numeric(base["avg_vol_3m"], errors="coerce")
    base = base.sort_values(by=["SourceOrder", "AvgVolNum"], ascending=[True, False], kind="mergesort")
    disp = pd.DataFrame({
        "Ticker": base["ticker"],
        "Avg Vol (3m)": base["AvgVolNum"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—"),
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
            color = "#ffe0b2"  # both
        elif earn:
            color = "#fff9c4"  # earnings in window
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
    st.caption("🟩 FF ≥ 0.20 🟨 Earnings in window 🟧 Both conditions true")

st.markdown(
    "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b> with GPT-5</p>",
    unsafe_allow_html=True,
)
