import math
from datetime import datetime, date
from typing import Optional, Tuple, List, Set, Dict

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

PACIFIC = pytz.timezone("America/Los_Angeles")

# ---------- helpers ----------
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

# ---------- yfinance ----------
@st.cache_data(ttl=600)
def get_spot(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    spot = tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice")
    if spot is None:
        px = tk.history(period="1d")
        if not px.empty:
            spot = float(px["Close"].iloc[-1])
    return _first_float(spot)

@st.cache_data(ttl=600)
def get_options(ticker: str) -> List[str]:
    return yf.Ticker(ticker).options or []

@st.cache_data(ttl=600)
def get_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oc = yf.Ticker(ticker).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

@st.cache_data(ttl=600)
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

# ---------- calculations ----------
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

# ---------- screener ----------
def screen_ticker(ticker: str) -> List[Dict]:
    spot = get_spot(ticker)
    if spot is None:
        return [{"ticker": ticker, "ff": "No spot"}]

    expiries = get_options(ticker)
    if not expiries:
        return [{"ticker": ticker, "ff": "No expirations"}]

    ed = [(e, _calc_dte(e)) for e in expiries]
    nearest = lambda t: min(ed, key=lambda x: abs(x[1] - t))
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
        s1, s2 = iv1 / 100.0, iv2 / 100.0
        T1, T2 = dte1 / 365.0, dte2 / 365.0
        fwd_sigma, ff = forward_and_ff(s1, T1, s2, T2)
        strike, c1, c2, debit = calendar_debit(ticker, exp1, exp2, spot)

        earn_txt, tags = "—", []
        if earn_dt:
            e1d, e2d = _expiry_to_date(exp1), _expiry_to_date(exp2)
            if e1d and e2d and min(e1d, e2d) <= earn_dt <= max(e1d, e2d):
                earn_txt, tags = earn_dt.strftime("%Y-%m-%d"), ["earn"]
        if ff and ff >= 0.20:
            tags.append("hot")

        rows.append({
            "ticker": ticker,
            "pair": label,
            "exp1": exp1, "dte1": dte1, "iv1": f"{iv1:.2f}%",
            "exp2": exp2, "dte2": dte2, "iv2": f"{iv2:.2f}%",
            "fwd_vol": f"{(fwd_sigma*100):.2f}%" if fwd_sigma else "—",
            "ff": f"{(ff*100):.2f}%" if ff else "—",
            "cal_debit": f"{debit:.2f}" if debit else "—",
            "earn_in_window": earn_txt,
            "_tags": tags,
        })
    return rows

# ---------- UI ----------
st.set_page_config(page_title="Forward Vol Screener", layout="wide")
st.title("📈 Forward Volatility Screener")

raw = st.text_area(
    "Tickers (comma / space separated):",
    "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, NFLX, AMD, AVGO",
    height=100,
)

if st.button("Run Screener"):
    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    progress = st.progress(0.0)
    all_rows = []
    for i, t in enumerate(tickers, 1):
        all_rows.extend(screen_ticker(t))
        progress.progress(i / len(tickers), text=f"Scanning {t}…")
    progress.empty()
    df = pd.DataFrame(all_rows)
    if "_tags" not in df.columns:
        df["_tags"] = [[] for _ in range(len(df))]

    def highlight(row):
        earn = "earn" in row["_tags"]
        hot = "hot" in row["_tags"]
        color = "#ffffff"
        if earn and hot:
            color = "#ffe0b2"
        elif earn:
            color = "#fff9c4"
        elif hot:
            color = "#dcedc8"
        return [f"background-color:{color}; color:#000000;"] * len(row)

    styled = (
        df.style
        .apply(highlight, axis=1)
        .set_properties(**{"border": "1px solid #bbb", "color": "#000", "font-size": "14px"})
    )

    st.markdown(
        """
        <style>
        table {border-collapse: collapse; width: 100%;}
        th {position: sticky; top: 0; background-color: #f0f0f0; color: #000; font-weight: bold;}
        tr:nth-child(even) {background-color: #fafafa;}
        tr:hover {background-color: #e0e0e0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(styled.to_html(), unsafe_allow_html=True)
    st.caption("🟩 FF ≥ 0.20 🟨 Earnings in window 🟧 Both conditions true")

    st.markdown(
        "<p style='text-align:center; font-size:14px; color:#888;'>Developed by <b>Skyler Wilcox</b> with GPT-5</p>",
        unsafe_allow_html=True,
    )
