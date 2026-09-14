# Earnings Desk

A Streamlit earnings calendar for a tech/semis watchlist, plus a review of how the last
few quarters actually landed.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's in it

**Sidebar** — the default list of 122 tickers loads pre-selected. Remove any you don't
follow, or type your own into "Add tickers" (comma separated). An optional theme filter
(semis, software, quantum, power, crypto…) narrows the selection without extra calls.
Nothing is fetched until you press **Load data**, so a 120-ticker list doesn't hammer
Yahoo on every rerun. Results are cached for six hours; "Clear cache" forces a refresh.

**Calendar** — month grid with every scheduled report as a chip, colour-coded blue for
before the open and amber for after the close, grey when the time isn't confirmed. Today
is outlined. Prev / next month navigation.

**Next 30 days** — adjustable horizon (7–120 days), a density bar chart of how crowded
each day is, a sortable table with consensus EPS and days away, and downloads as CSV or
as an `.ics` file you can drop straight into a calendar app.

**Past results** — consensus vs reported EPS for the last N quarters, filtered by beat /
miss / in line and by minimum absolute surprise. Beat rate, median surprise, average
surprise by ticker, and a scatter of EPS surprise against the price move that followed
(with the correlation, which is usually weaker than people expect).

**Single ticker** — consensus vs reported line chart, the move on each print, and the
full quarter-by-quarter table.

**Data check** — which symbols returned nothing and why, plus a full CSV export.

## Notes on the data

- Source is Yahoo Finance via `yfinance`. Scheduled dates are estimates until the company
  confirms them, and reported figures are occasionally revised.
- Surprise % is recalculated as `(reported - estimate) / |estimate| * 100` rather than
  trusting Yahoo's own column, which changes units between versions.
- Price reaction uses the same session for a before-the-open print and the next session
  for after-the-close or unknown timing. It's a single batched download for the whole
  list, so it costs one request rather than one per ticker.
- Some symbols in the default list (ETFs, ADRs, recent listings) won't return earnings
  data at all — the Data check tab lists them.

## Deploying to Streamlit Community Cloud

Push `app.py`, `requirements.txt` and `.streamlit/config.toml` to the repo and point the
app at `app.py`. The config file carries the dark theme the custom CSS is built around.
