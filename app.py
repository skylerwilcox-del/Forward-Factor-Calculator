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

# Persistent session state
if "df" not in st.session_state:
    st.session_state.df = None
if "tickers" not in st.session_state:
    st.session_state.tickers = []
if "vol_topN" not in st.session_state:
    st.session_state.vol_topN = None
if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False  # renamed to avoid conflicts

TOP_STOCKS = 27
ADD_ETFS = True
FIXED_ETFS = ["VOO", "SPY", "QQQ"]
MAX_WORKERS = 12
DL_CHUNK_STAGE1 = 80
DL_CHUNK_STAGE2 = 50
SHORTLIST_SIZE = 400

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
# Universe and ranking
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def get_us_stock_universe() -> List[str]:
    tickers = []
    try:
        nas = getattr(yf, "tickers_nasdaq", None)
        if callable(nas):
            r = nas()
            if isinstance(r, (list, tuple)) and r:
                tickers.extend([t.strip().upper() for t in r if t])
    except Exception:
        pass
    try:
        oth = getattr(yf, "tickers_other", None)
        if callable(oth):
            r = oth()
            if isinstance(r, (list, tuple)) and r:
                tickers.extend([t.strip().upper() for t in r if t])
    except Exception:
        pass
    if not tickers:
        try:
            spx = getattr(yf, "tickers_sp500", None)
            if callable(spx):
                r = spx()
                if isinstance(r, (list, tuple)) and r:
                    tickers.extend([t.strip().upper() for t in r if t])
        except Exception:
            pass
    return sorted(set(tickers))

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

def rank_top_stocks_all_us_pipeline(extras: List[str], top_n: int, add_etfs: bool,
                                    update_stage1=None, update_stage2=None) -> pd.DataFrame:
    universe = get_us_stock_universe()
    for e in extras:
        if e not in universe:
            universe.append(e)

    # Stage 1
    stage1_rows = []
    processed = 0
    total = len(universe)
    for block in _chunks(universe, DL_CHUNK_STAGE1):
        data = _safe_multi_download(block, period="7d", interval="1d")
        got = [c[0] for c in data.columns.unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
        for t in got:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                sub = _exclude_today_if_open(sub)
                if "Volume" not in sub.columns or sub["Volume"].dropna().empty:
                    continue
                last_vol = float(sub["Volume"].dropna().iloc[-1])
                last_close = float(sub["Close"].dropna().iloc[-1])
                stage1_rows.append({"ticker": t, "last_vol": last_vol, "last_close": last_close})
            except Exception:
                continue
        processed += len(block)
        if callable(update_stage1):
            update_stage1(processed, total)

    if not stage1_rows:
        return pd.DataFrame(columns=["ticker","avg_week_volume","last_close","source"])

    stage1 = pd.DataFrame(stage1_rows).sort_values("last_vol", ascending=False)
    shortlist = stage1["ticker"].head(SHORTLIST_SIZE).tolist()

    # Stage 2
    stage2_rows = []
    processed = 0
    total2 = len(shortlist)
    for block in _chunks(shortlist, DL_CHUNK_STAGE2):
        data = _safe_multi_download(block, period="20d", interval="1d")
        got = [c[0] for c in data.columns.unique(level=0)] if isinstance(data.columns, pd.MultiIndex) else block
        for t in got:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                sub = _exclude_today_if_open(sub)
                avgv = _avg5(sub["Volume"])
                last_close = float(sub["Close"].dropna().iloc[-1])
                stage2_rows.append({"ticker": t, "avg_week_volume": avgv, "last_close": last_close, "source": "Stock"})
            except Exception:
                continue
        processed += len(block)
        if callable(update_stage2):
            update_stage2(processed, total2)

    vol_df = pd.DataFrame(stage2_rows).dropna(subset=["avg_week_volume"])
    vol_df = vol_df.sort_values("avg_week_volume", ascending=False).head(top_n)

    if add_etfs:
        etf_rows = []
        for etf in FIXED_ETFS:
            try:
                hist = yf.download(etf, period="20d", interval="1d", progress=False)
                hist = _exclude_today_if_open(hist)
                avgv = _avg5(hist["Volume"])
                last_close = float(hist["Close"].dropna().iloc[-1])
                etf_rows.append({"ticker": etf, "avg_week_volume": avgv, "last_close": last_close, "source": "ETF"})
            except Exception:
                continue
        vol_df = pd.concat([vol_df, pd.DataFrame(etf_rows)], ignore_index=True)

    return vol_df.reset_index(drop=True)

# =========================
# UI
# =========================
st.title("📈 Forward Volatility Screener (Top 27 by 5-Day Avg Volume — All US Stocks)")
st.markdown("Ranks all U.S. stocks by **avg volume (5d)** and scans options.")

raw_extra = st.text_input("Optional: Add tickers (comma/space separated)", "", placeholder="e.g., NVDA, TSLA, META")

def trigger_run():
    st.session_state.trigger_run = True

colA, colB = st.columns([1, 3])
with colA:
    st.button("Build List & Run", type="primary", on_click=trigger_run)
with colB:
    st.caption("Excludes today’s volume until after 4:05pm ET close.")

# ---------- EXECUTION ----------
if st.session_state.trigger_run:
    extras = _normalize_tickers(raw_extra)
    prog1 = st.progress(0.0, text="Stage 1/2: Scanning universe…")
    prog2 = st.progress(0.0, text="Stage 2/2: Computing averages…")

    def up1(done, total):
        prog1.progress(done/total, text=f"Stage 1/2: {done}/{total}")
    def up2(done, total):
        prog2.progress(done/total, text=f"Stage 2/2: {done}/{total}")

    with st.spinner("Building list…"):
        vol_top_df = rank_top_stocks_all_us_pipeline(extras, TOP_STOCKS, ADD_ETFS, up1, up2)

    prog1.empty()
    prog2.empty()

    st.session_state.vol_topN = vol_top_df
    st.session_state.tickers = vol_top_df["ticker"].tolist() if not vol_top_df.empty else []
    st.session_state.df = pd.DataFrame()  # placeholder
    st.session_state.trigger_run = False

    st.success("✅ Scan complete! Displaying results below.")

# --------- Display results ---------
if st.session_state.vol_topN is not None and not st.session_state.vol_topN.empty:
    st.subheader(f"Selected Tickers (Top {TOP_STOCKS} Stocks{' + ETFs' if ADD_ETFS else ''})")
    base = st.session_state.vol_topN.copy()
    base["Avg Vol (5d)"] = base["avg_week_volume"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    base["Last Close"] = base["last_close"].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
    base.rename(columns={"ticker": "Ticker", "source": "Source"}, inplace=True)
    st.dataframe(base[["Ticker","Avg Vol (5d)","Last Close","Source"]], hide_index=True, use_container_width=True)

st.markdown(
    "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b> with GPT-5</p>",
    unsafe_allow_html=True,
)
