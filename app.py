"""
Earnings Desk — an earnings calendar and results review app.

Two main views:
  • Calendar   — what is reporting, when, across your ticker list
  • Results    — how past quarters landed vs consensus, and how the stock moved

Data: Yahoo Finance via yfinance. Nothing is fetched until you press "Load data",
so the app stays friendly to Yahoo's rate limits.
"""

from __future__ import annotations

import calendar as pycalendar
import datetime as dt
import html
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    "MARA", "SPOT", "STX", "CRCL", "SMCI", "COIN", "NVDA", "SKHY", "RBLX", "MU",
    "MSTR", "SNDK", "LRCX", "KLAC", "CSCO", "AVGO", "QCOM", "IBM", "LITE", "INFQ",
    "NOK", "WDC", "UI", "RGTI", "TSM", "QBTS", "IONQ", "AMD", "TER", "COHR",
    "ASML", "ADI", "AMAT", "INTC", "NVTS", "GLW", "UMC", "CBRS", "CIEN", "MRVL",
    "WOLF", "Q", "NXPI", "CRDO", "ON", "DELL", "QRVO", "AEHR", "POET", "META",
    "SPCX", "HIMX", "HPE", "ENTG", "U", "AAPL", "UCTT", "TXN", "DOCN", "MCHP",
    "CEG", "GOOG", "HON", "QUBT", "TWLO", "SNOW", "ANET", "NET", "FTNT", "AI",
    "STM", "ESTC", "DT", "ORCL", "BOX", "CRM", "NEE", "TEAM", "CRWD", "PLTR",
    "DLR", "MSFT", "EQIX", "VEEV", "ETN", "WDAY", "INTU", "DUOL", "HOOD", "TEM",
    "IOT", "NOW", "INDI", "HUBS", "SNPS", "BBAI", "DDOG", "GEV", "PANW", "BABA",
    "ADBE", "VRT", "APPN", "MPWR", "ALAB", "CRWV", "SOUN", "APLD", "AMZN", "CDNS",
    "ADSK", "PATH", "AMBA", "NBIS", "MDB", "APH", "DRAM", "ARM",
]

# Small hand-made grouping so the calendar can be filtered by theme without an
# extra network call per ticker. Anything not listed falls into "Other".
THEMES: Dict[str, List[str]] = {
    "Semis & equipment": [
        "NVDA", "AMD", "INTC", "TSM", "UMC", "MU", "AVGO", "QCOM", "TXN", "ADI",
        "MRVL", "NXPI", "ON", "MCHP", "QRVO", "STM", "MPWR", "WOLF", "NVTS",
        "HIMX", "INDI", "AMBA", "ARM", "ALAB", "CRDO", "LRCX", "KLAC", "AMAT",
        "ASML", "TER", "AEHR", "ENTG", "UCTT", "SMCI", "DRAM",
    ],
    "Networking & optics": [
        "CSCO", "ANET", "CIEN", "COHR", "LITE", "GLW", "POET", "UI", "NOK", "APH",
    ],
    "Storage & hardware": ["STX", "WDC", "SNDK", "DELL", "HPE", "IBM", "AAPL"],
    "Software & data": [
        "MSFT", "ORCL", "CRM", "ADBE", "NOW", "SNOW", "MDB", "DDOG", "ESTC", "DT",
        "TEAM", "WDAY", "INTU", "HUBS", "VEEV", "APPN", "PATH", "BOX", "TWLO",
        "SNPS", "CDNS", "ADSK", "IOT", "U", "DUOL", "SPOT", "RBLX", "AI", "PLTR",
        "BBAI", "SOUN", "TEM", "META", "GOOG", "AMZN", "BABA", "NET",
    ],
    "Security": ["CRWD", "PANW", "FTNT"],
    "Crypto & fintech": ["COIN", "MSTR", "MARA", "HOOD", "CRCL", "APLD"],
    "Quantum": ["IONQ", "RGTI", "QBTS", "QUBT"],
    "Power & data centres": [
        "CEG", "NEE", "GEV", "VRT", "ETN", "HON", "EQIX", "DLR", "NBIS", "CRWV",
        "DOCN",
    ],
}

TICKER_THEME: Dict[str, str] = {t: theme for theme, names in THEMES.items() for t in names}


# --------------------------------------------------------------------------------------
# Page setup and styling
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Earnings Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#0F1620"
PANEL = "#17202C"
LINE = "#26313F"
TEXT = "#E6EAF0"
MUTED = "#8C99A9"
BEAT = "#3FB68B"
MISS = "#E2685E"
AMBER = "#E8A33D"
AZURE = "#5B8DEF"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      html, body, [class*="css"] {{ font-family: 'Inter Tight', system-ui, sans-serif; }}

      .masthead {{
        display: flex; align-items: center; gap: .85rem;
        margin: .25rem 0 .2rem;
      }}
      .masthead .mast-icon {{ font-size: 2.6rem; line-height: 1; }}
      .masthead h1 {{
        font-size: 2.75rem; font-weight: 700; letter-spacing: -.03em;
        margin: 0; line-height: 1.15;
      }}
      .mast-sub {{ color: {MUTED}; font-size: 1rem; margin: 0 0 1.3rem; }}

      .statbar {{ display: flex; flex-wrap: wrap; gap: .6rem; margin-bottom: 1rem; }}
      .stat {{
        flex: 1 1 150px; background: #ffffff; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: .7rem .85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
      }}
      .stat .k {{ color: #64748B; font-size: .74rem; }}
      .stat .v {{
        font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem;
        font-variant-numeric: tabular-nums; margin-top: .15rem; color: #0F172A;
      }}

      .cal-head, .cal-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; }}
      .cal-head div {{ color: {MUTED}; font-size: .76rem; padding: 0 .3rem .3rem; }}
      .cal-cell {{
        background: #ffffff; border: 1px solid #E2E8F0; border-radius: 9px;
        min-height: 96px; padding: .38rem .4rem; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
      }}
      .cal-cell.dim {{ opacity: .35; }}
      .cal-cell.today {{ border-color: {AMBER}; box-shadow: inset 0 0 0 1px {AMBER}55; }}
      .cal-date {{
        font-family: 'IBM Plex Mono', monospace; font-size: .78rem;
        color: #94A3B8; display: flex; justify-content: space-between;
      }}
      .cal-date b {{ color: #0F172A; font-weight: 600; }}
      .chip {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace;
        font-size: .7rem; padding: .1rem .34rem; margin: .16rem .16rem 0 0;
        border-radius: 5px; border: 1px solid transparent;
      }}
      .chip.bmo {{ background: #1D4ED8; border-color: #1E40AF; color: #EFF6FF; font-weight: 500; }}
      .chip.amc {{ background: #B45309; border-color: #92400E; color: #FEF3C7; font-weight: 500; }}
      .chip.tbd {{ background: #374151; border-color: #4B5563; color: #D1D5DB; font-weight: 500; }}
      .more {{ color: #64748B; font-size: .68rem; display: block; margin-top: .2rem; font-weight: 500; }}

      .legend {{ color: {MUTED}; font-size: .78rem; margin-top: .6rem; }}
      .legend b {{ color: {TEXT}; font-weight: 500; }}

      div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
      section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------------------

NUM_COLS = {"EPS Estimate": "eps_est", "Reported EPS": "eps_actual", "Surprise(%)": "surprise_yf"}


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "when", "date", "session", "eps_est", "eps_actual", "surprise_pct"]
    )


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_events(ticker: str, limit: int = 16) -> Tuple[pd.DataFrame, str]:
    """Return (events, error). One row per earnings date, past and future."""
    if yf is None:
        return _empty_events(), "yfinance is not installed"
    try:
        tk = yf.Ticker(ticker)
        raw = None
        for getter in ("get_earnings_dates", "earnings_dates"):
            try:
                attr = getattr(tk, getter)
                raw = attr(limit=limit) if callable(attr) else attr
            except Exception:
                raw = None
            if raw is not None and len(raw):
                break
        if raw is None or not len(raw):
            return _empty_events(), "no earnings dates returned"

        df = raw.reset_index()
        df = df.rename(columns={df.columns[0]: "when"})
        for src, dst in NUM_COLS.items():
            df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan

        when = pd.to_datetime(df["when"], errors="coerce", utc=True)
        try:
            local = pd.to_datetime(df["when"], errors="coerce").dt.tz_convert("America/New_York")
        except (TypeError, AttributeError):
            local = when.dt.tz_convert("America/New_York")
        df["when"] = local
        df["date"] = local.dt.date
        hour = local.dt.hour.fillna(-1)
        df["session"] = np.where(hour < 0, "TBD", np.where(hour < 12, "BMO", "AMC"))
        # Yahoo uses a placeholder time for unconfirmed dates; treat noon-ish as unknown.
        df.loc[(hour == 12) & (local.dt.minute == 0), "session"] = "TBD"

        est, act = df["eps_est"], df["eps_actual"]
        calc = (act - est) / est.abs().replace(0, np.nan) * 100
        df["surprise_pct"] = calc.where(calc.notna(), df["surprise_yf"])

        df["ticker"] = ticker
        out = df[["ticker", "when", "date", "session", "eps_est", "eps_actual", "surprise_pct"]]
        return out.dropna(subset=["date"]).sort_values("when", ascending=False), ""
    except Exception as exc:  # network, parsing, delisted tickers
        return _empty_events(), f"{type(exc).__name__}: {exc}"[:140]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_events_bulk(tickers: Tuple[str, ...], limit: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames, errors = [], {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_events, t, limit): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df, err = fut.result()
            except Exception as exc:
                df, err = _empty_events(), str(exc)[:140]
            if err:
                errors[t] = err
            if len(df):
                frames.append(df)
    events = pd.concat(frames, ignore_index=True) if frames else _empty_events()
    return events, errors


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_closes(tickers: Tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily closes, wide (dates x tickers). Empty frame if the download fails."""
    if yf is None or not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            list(tickers), start=start, end=end, auto_adjust=True,
            progress=False, threads=True, group_by="column",
        )
        if data is None or not len(data):
            return pd.DataFrame()
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        if len(tickers) == 1 and list(close.columns) == ["Close"]:
            close.columns = [tickers[0]]
        close.index = pd.to_datetime(close.index).date
        return close
    except Exception:
        return pd.DataFrame()


def add_price_reaction(events: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Percentage move on the session that first trades on the news."""
    events = events.copy()
    events["reaction_pct"] = np.nan
    if closes.empty:
        return events

    rets = closes.pct_change() * 100
    sessions = list(rets.index)

    for i, row in events.iterrows():
        tkr = row["ticker"]
        if tkr not in rets.columns or pd.isna(row["date"]):
            continue
        d = row["date"]
        # BMO prints move the same session; AMC (and unknown) move the next one.
        pos = np.searchsorted(sessions, d, side="left" if row["session"] == "BMO" else "right")
        if pos >= len(sessions):
            continue
        val = rets.iloc[pos][tkr]
        if pd.notna(val):
            events.at[i, "reaction_pct"] = float(val)
    return events


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def clean_tickers(text: str) -> List[str]:
    parts = [p.strip().upper() for p in text.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p and all(c.isalnum() or c in ".-" for c in p)]


def ics_export(upcoming: pd.DataFrame) -> str:
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Earnings Desk//EN", "CALSCALE:GREGORIAN"]
    for _, r in upcoming.iterrows():
        day = r["date"]
        lines += [
            "BEGIN:VEVENT",
            f"UID:{r['ticker']}-{day}@earnings-desk",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
            f"DTEND;VALUE=DATE:{day + dt.timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{r['ticker']} earnings ({r['session']})",
            f"DESCRIPTION:Consensus EPS {r['eps_est'] if pd.notna(r['eps_est']) else 'n/a'}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


def month_grid(events: pd.DataFrame, year: int, month: int, per_cell: int = 6) -> str:
    by_day: Dict[dt.date, List[Tuple[str, str]]] = {}
    for _, r in events.iterrows():
        by_day.setdefault(r["date"], []).append((r["ticker"], r["session"]))

    weeks = pycalendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    today = dt.date.today()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    head = "".join(f"<div>{n}</div>" for n in names)
    cells = []
    for week in weeks:
        for day in week:
            classes = ["cal-cell"]
            if day.month != month:
                classes.append("dim")
            if day == today:
                classes.append("today")
            items = sorted(by_day.get(day, []))
            chips = "".join(
                f'<span class="chip {sess.lower()}" title="{sess}">{html.escape(t)}</span>'
                for t, sess in items[:per_cell]
            )
            extra = f'<span class="more">+{len(items) - per_cell} more</span>' if len(items) > per_cell else ""
            count = f"<span>{len(items)}</span>" if items else ""
            cells.append(
                f'<div class="{" ".join(classes)}">'
                f'<div class="cal-date"><b>{day.day}</b>{count}</div>{chips}{extra}</div>'
            )
    return f'<div class="cal-head">{head}</div><div class="cal-grid">{"".join(cells)}</div>'


def stat_bar(items: List[Tuple[str, str]]) -> None:
    blocks = "".join(
        f'<div class="stat"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(str(v))}</div></div>'
        for k, v in items
    )
    st.markdown(f'<div class="statbar">{blocks}</div>', unsafe_allow_html=True)


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(strokeWidth=0).configure_axis(
        grid=True, gridColor=LINE, domainColor=LINE, tickColor=LINE,
        labelColor=MUTED, titleColor=MUTED, labelFont="Inter Tight", titleFont="Inter Tight",
    ).configure_legend(labelColor=MUTED, titleColor=MUTED)


# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------

if "universe" not in st.session_state:
    st.session_state.universe = sorted(set(DEFAULT_TICKERS))
if "selected" not in st.session_state:
    st.session_state.selected = sorted(set(DEFAULT_TICKERS))

with st.sidebar:
    st.markdown("### Watchlist")

    add_text = st.text_input("Add tickers", placeholder="e.g. ORCL, TSLA, ASTS")
    c1, c2 = st.columns(2)
    if c1.button("Add", use_container_width=True) and add_text:
        new = clean_tickers(add_text)
        st.session_state.universe = sorted(set(st.session_state.universe) | set(new))
        st.session_state.selected = sorted(set(st.session_state.selected) | set(new))
        st.rerun()
    if c2.button("Reset list", use_container_width=True):
        st.session_state.universe = sorted(set(DEFAULT_TICKERS))
        st.session_state.selected = sorted(set(DEFAULT_TICKERS))
        st.rerun()

    selected = st.multiselect(
        "Tickers in play",
        options=st.session_state.universe,
        key="selected",
        help="Remove any you don't follow, or add your own above.",
    )

    theme_pick = st.multiselect(
        "Limit to themes", options=sorted(THEMES.keys()),
        help="Optional. Filters the selection using a built-in grouping.",
    )
    if theme_pick:
        keep = {t for th in theme_pick for t in THEMES[th]}
        selected = [t for t in selected if t in keep]

    st.markdown("### Data")
    quarters = st.slider("Quarters of history per ticker", 4, 24, 12)
    want_reaction = st.toggle("Include price reaction", value=True,
                              help="Adds one batched price download for the move around each print.")
    st.caption(f"{len(selected)} tickers selected")

    run = st.button("Load data", type="primary", use_container_width=True, disabled=not selected)
    if st.button("Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("events", None)
        st.rerun()


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------

st.markdown(
    '<div class="masthead"><span class="mast-icon">📈</span>'
    "<h1>Earnings Desk</h1></div>"
    '<div class="mast-sub">Who reports when, and how the last few quarters landed</div>',
    unsafe_allow_html=True,
)

if run:
    with st.spinner(f"Pulling earnings dates for {len(selected)} tickers…"):
        events, errors = fetch_events_bulk(tuple(sorted(selected)), quarters)
    if want_reaction and len(events):
        past = events[events["date"] <= dt.date.today()]
        if len(past):
            start = min(past["date"]) - dt.timedelta(days=7)
            with st.spinner("Adding price reactions…"):
                closes = fetch_closes(tuple(sorted(selected)), start, dt.date.today() + dt.timedelta(days=1))
            events = add_price_reaction(events, closes)
    st.session_state.events = events
    st.session_state.errors = errors
    st.session_state.loaded_at = dt.datetime.now()

if "events" not in st.session_state:
    st.info(
        "Pick your tickers in the sidebar, then press **Load data**. "
        "Nothing is fetched until you do, which keeps Yahoo happy on a list this size."
    )
    st.stop()

events: pd.DataFrame = st.session_state.events
errors: Dict[str, str] = st.session_state.get("errors", {})

if events.empty:
    st.warning("No earnings dates came back for this selection. Try fewer tickers, or clear the cache and reload.")
    st.stop()

events = events[events["ticker"].isin(selected)].copy()
if "reaction_pct" not in events.columns:
    events["reaction_pct"] = np.nan
events["theme"] = events["ticker"].map(TICKER_THEME).fillna("Other")
today = dt.date.today()
upcoming = events[events["date"] >= today].sort_values("date").copy()
past = events[events["date"] < today].dropna(subset=["eps_actual"]).sort_values("date", ascending=False).copy()
upcoming["days_away"] = upcoming["date"].map(lambda d: (d - today).days)

next_up = upcoming.head(1)
busiest = upcoming[upcoming["days_away"] <= 30].groupby("date").size().sort_values(ascending=False)
beat_rate = (past["surprise_pct"] > 0).mean() * 100 if len(past) else np.nan

stat_bar([
    ("Tickers loaded", f"{events['ticker'].nunique()}"),
    ("Reporting next 7d", f"{(upcoming['days_away'] <= 7).sum()}"),
    ("Reporting next 30d", f"{(upcoming['days_away'] <= 30).sum()}"),
    ("Next up", f"{next_up.iloc[0]['ticker']} · {next_up.iloc[0]['days_away']}d" if len(next_up) else "—"),
    ("Busiest day (30d)", f"{busiest.index[0]:%d %b} · {busiest.iloc[0]}" if len(busiest) else "—"),
    ("Beat rate", f"{beat_rate:.0f}%" if pd.notna(beat_rate) else "—"),
])

tab_cal, tab_next, tab_results, tab_ticker, tab_health = st.tabs(
    ["Calendar", "Next 30 days", "Past results", "Single ticker", "Data check"]
)


# --------------------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------------------

with tab_cal:
    if "cal_offset" not in st.session_state:
        st.session_state.cal_offset = 0

    anchor = (dt.date(today.year, today.month, 1) +
              pd.DateOffset(months=st.session_state.cal_offset)).date()

    nav1, nav2, nav3, _ = st.columns([1, 1, 1, 6])
    if nav1.button("◀ Prev"):
        st.session_state.cal_offset -= 1
        st.rerun()
    if nav2.button("This month"):
        st.session_state.cal_offset = 0
        st.rerun()
    if nav3.button("Next ▶"):
        st.session_state.cal_offset += 1
        st.rerun()

    month_events = events[
        events["date"].map(lambda d: d.year == anchor.year and d.month == anchor.month)
    ]
    st.markdown(f"#### {anchor:%B %Y} · {len(month_events)} reports")
    st.markdown(month_grid(month_events, anchor.year, anchor.month), unsafe_allow_html=True)
    st.markdown(
        f'<div class="legend"><span class="chip bmo">BMO</span> before the open &nbsp;'
        f'<span class="chip amc">AMC</span> after the close &nbsp;'
        f'<span class="chip tbd">TBD</span> time unconfirmed. '
        f"Dates from Yahoo are estimates until a company confirms them.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Next 30 days
# --------------------------------------------------------------------------------------

with tab_next:
    horizon = st.slider("Horizon (days)", 7, 120, 30, step=7)
    window = upcoming[upcoming["days_away"] <= horizon].copy()

    if window.empty:
        st.info("Nothing scheduled in that window. Stretch the horizon or add tickers.")
    else:
        density = window.groupby("date").size().reset_index(name="reports")
        bars = alt.Chart(density).mark_bar(size=12, color=AZURE, cornerRadius=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("reports:Q", title="reports"),
            tooltip=["date:T", "reports:Q"],
        ).properties(height=140)
        st.altair_chart(chart_theme(bars), use_container_width=True)

        table = window[["date", "days_away", "ticker", "theme", "session", "eps_est"]].rename(
            columns={"date": "Date", "days_away": "In days", "ticker": "Ticker",
                     "theme": "Theme", "session": "Session", "eps_est": "Consensus EPS"}
        )
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "Consensus EPS": st.column_config.NumberColumn(format="%.2f"),
                "In days": st.column_config.NumberColumn(format="%d"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download CSV", table.to_csv(index=False).encode(),
            file_name=f"earnings_next_{horizon}d.csv", mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Download calendar (.ics)", ics_export(window).encode(),
            file_name="earnings.ics", mime="text/calendar", use_container_width=True,
        )


# --------------------------------------------------------------------------------------
# Past results
# --------------------------------------------------------------------------------------

with tab_results:
    if past.empty:
        st.info("No reported quarters came back for this selection.")
    else:
        f1, f2, f3 = st.columns([2, 1, 1])
        lookback = f1.slider("Quarters back", 1, 12, 4)
        outcome = f2.selectbox("Outcome", ["All", "Beats", "Misses", "In line"])
        min_abs = f3.number_input("Min |surprise| %", 0.0, 500.0, 0.0, step=1.0)

        ranked = past.copy()
        ranked["q_rank"] = ranked.groupby("ticker")["date"].rank(method="first", ascending=False)
        view = ranked[ranked["q_rank"] <= lookback].copy()

        if outcome == "Beats":
            view = view[view["surprise_pct"] > 0.5]
        elif outcome == "Misses":
            view = view[view["surprise_pct"] < -0.5]
        elif outcome == "In line":
            view = view[view["surprise_pct"].abs() <= 0.5]
        view = view[view["surprise_pct"].abs().fillna(0) >= min_abs]

        if view.empty:
            st.info("No quarters match those filters.")
        else:
            hit = (view["surprise_pct"] > 0).mean() * 100
            med_surprise = view["surprise_pct"].median()
            med_react = view["reaction_pct"].median()
            stat_bar([
                ("Quarters shown", f"{len(view)}"),
                ("Beat rate", f"{hit:.0f}%"),
                ("Median surprise", f"{med_surprise:+.1f}%" if pd.notna(med_surprise) else "—"),
                ("Median move", f"{med_react:+.1f}%" if pd.notna(med_react) else "—"),
            ])

            avg = (view.groupby("ticker")
                   .agg(surprise=("surprise_pct", "mean"), n=("surprise_pct", "size"))
                   .reset_index().dropna(subset=["surprise"]))
            extremes = pd.concat([avg.nlargest(12, "surprise"), avg.nsmallest(12, "surprise")]).drop_duplicates()

            left, right = st.columns(2)
            with left:
                st.markdown("**Average EPS surprise by ticker**")
                bar = alt.Chart(extremes).mark_bar(cornerRadius=2).encode(
                    y=alt.Y("ticker:N", sort="-x", title=None),
                    x=alt.X("surprise:Q", title="avg surprise %"),
                    color=alt.condition(alt.datum.surprise > 0, alt.value(BEAT), alt.value(MISS)),
                    tooltip=["ticker", alt.Tooltip("surprise:Q", format="+.1f"),
                             alt.Tooltip("n:Q", title="quarters")],
                ).properties(height=460)
                st.altair_chart(chart_theme(bar), use_container_width=True)

            with right:
                st.markdown("**Surprise vs the move that followed**")
                if view["reaction_pct"].notna().any():
                    sc = view.dropna(subset=["reaction_pct", "surprise_pct"])
                    sc = sc[sc["surprise_pct"].abs() < 300]
                    pts = alt.Chart(sc).mark_circle(size=70, opacity=.75).encode(
                        x=alt.X("surprise_pct:Q", title="EPS surprise %"),
                        y=alt.Y("reaction_pct:Q", title="price move %"),
                        color=alt.Color("theme:N", legend=alt.Legend(title=None, orient="bottom")),
                        tooltip=["ticker", "date:T",
                                 alt.Tooltip("surprise_pct:Q", format="+.1f"),
                                 alt.Tooltip("reaction_pct:Q", format="+.1f")],
                    ).properties(height=460)
                    st.altair_chart(chart_theme(pts), use_container_width=True)
                    corr = sc["surprise_pct"].corr(sc["reaction_pct"])
                    st.caption(
                        f"Correlation {corr:+.2f} across {len(sc)} quarters — "
                        "a reminder that guidance usually matters more than the EPS line."
                    )
                else:
                    st.info("Turn on price reaction in the sidebar and reload to see this.")

            st.markdown("**Every quarter in the window**")
            cols = ["date", "ticker", "theme", "session", "eps_est",
                    "eps_actual", "surprise_pct", "reaction_pct"]
            detail = view[cols].sort_values("date", ascending=False).rename(
                columns={"date": "Date", "ticker": "Ticker", "theme": "Theme", "session": "Session",
                         "eps_est": "Est EPS", "eps_actual": "Actual EPS",
                         "surprise_pct": "Surprise %", "reaction_pct": "Move %"}
            )
            st.dataframe(
                detail, use_container_width=True, hide_index=True, height=420,
                column_config={
                    "Est EPS": st.column_config.NumberColumn(format="%.2f"),
                    "Actual EPS": st.column_config.NumberColumn(format="%.2f"),
                    "Surprise %": st.column_config.NumberColumn(format="%+.1f"),
                    "Move %": st.column_config.NumberColumn(format="%+.1f"),
                },
            )
            st.download_button(
                "Download results CSV", detail.to_csv(index=False).encode(),
                file_name="earnings_results.csv", mime="text/csv",
            )


# --------------------------------------------------------------------------------------
# Single ticker
# --------------------------------------------------------------------------------------

with tab_ticker:
    pick = st.selectbox("Ticker", sorted(events["ticker"].unique()))
    hist = events[events["ticker"] == pick].sort_values("date")
    reported = hist.dropna(subset=["eps_actual"])
    nxt = hist[hist["date"] >= today].head(1)

    stat_bar([
        ("Next report", f"{nxt.iloc[0]['date']:%d %b %Y}" if len(nxt) else "not scheduled"),
        ("Session", nxt.iloc[0]["session"] if len(nxt) else "—"),
        ("Consensus EPS", f"{nxt.iloc[0]['eps_est']:.2f}"
         if len(nxt) and pd.notna(nxt.iloc[0]["eps_est"]) else "—"),
        ("Beats / quarters", f"{int((reported['surprise_pct'] > 0).sum())} of {len(reported)}"
         if len(reported) else "—"),
    ])

    if reported.empty:
        st.info(f"No reported quarters on file for {pick}.")
    else:
        melted = reported.melt(id_vars="date", value_vars=["eps_est", "eps_actual"],
                               var_name="series", value_name="eps").dropna()
        melted["series"] = melted["series"].map({"eps_est": "Consensus", "eps_actual": "Reported"})
        line = alt.Chart(melted).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("eps:Q", title="EPS"),
            color=alt.Color("series:N", scale=alt.Scale(range=[MUTED, AZURE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["date:T", "series:N", alt.Tooltip("eps:Q", format=".2f")],
        ).properties(height=260)
        st.altair_chart(chart_theme(line), use_container_width=True)

        if reported["reaction_pct"].notna().any():
            react = alt.Chart(reported.dropna(subset=["reaction_pct"])).mark_bar(cornerRadius=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("reaction_pct:Q", title="move on the print %"),
                color=alt.condition(alt.datum.reaction_pct > 0, alt.value(BEAT), alt.value(MISS)),
                tooltip=["date:T", alt.Tooltip("reaction_pct:Q", format="+.1f"),
                         alt.Tooltip("surprise_pct:Q", format="+.1f")],
            ).properties(height=200)
            st.altair_chart(chart_theme(react), use_container_width=True)

        st.dataframe(
            reported[["date", "session", "eps_est", "eps_actual", "surprise_pct", "reaction_pct"]]
            .sort_values("date", ascending=False)
            .rename(columns={"date": "Date", "session": "Session", "eps_est": "Est EPS",
                             "eps_actual": "Actual EPS", "surprise_pct": "Surprise %",
                             "reaction_pct": "Move %"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Est EPS": st.column_config.NumberColumn(format="%.2f"),
                "Actual EPS": st.column_config.NumberColumn(format="%.2f"),
                "Surprise %": st.column_config.NumberColumn(format="%+.1f"),
                "Move %": st.column_config.NumberColumn(format="%+.1f"),
            },
        )


# --------------------------------------------------------------------------------------
# Data check
# --------------------------------------------------------------------------------------

with tab_health:
    loaded_at = st.session_state.get("loaded_at")
    st.caption(f"Loaded {loaded_at:%d %b %Y, %H:%M}" if loaded_at else "")

    covered = set(events["ticker"].unique())
    missing = sorted(set(selected) - covered)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Returned data — {len(covered)}**")
        st.write(", ".join(sorted(covered)) or "none")
    with c2:
        st.markdown(f"**No earnings data — {len(missing)}**")
        st.write(", ".join(missing) or "none")
        if missing:
            st.caption(
                "Usually an ETF, an ADR, a recent listing, or a symbol Yahoo maps differently. "
                "Worth checking the symbol on finance.yahoo.com."
            )

    if errors:
        with st.expander(f"Fetch messages ({len(errors)})"):
            st.dataframe(
                pd.DataFrame(sorted(errors.items()), columns=["Ticker", "Message"]),
                use_container_width=True, hide_index=True,
            )

    buf = io.StringIO()
    events.sort_values(["ticker", "date"]).to_csv(buf, index=False)
    st.download_button("Download everything loaded (CSV)", buf.getvalue().encode(),
                       file_name="earnings_events_full.csv", mime="text/csv")

    st.caption(
        "Dates, consensus and reported EPS come from Yahoo Finance and can be revised or wrong. "
        "Treat scheduled dates as estimates until the company confirms. Not investment advice."
    )
