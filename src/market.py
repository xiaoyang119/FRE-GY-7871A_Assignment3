"""Market-data collectors: FRED (fredgraph.csv) and Yahoo (chart API).

Uses only the standard library + pandas/numpy so it runs without API keys.
"""
import io
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import config

UA = {"User-Agent": "Mozilla/5.0 (academic research script)"}


def _get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_fred(series_id):
    """Return a date-indexed Series for a FRED series (full history)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    txt = _get(url)
    df = pd.read_csv(io.StringIO(txt))
    df = df.dropna()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = df.set_index("observation_date")[series_id].astype(float)
    return s


def fetch_yahoo(ticker):
    """Return a date-indexed close-price Series for a Yahoo ticker."""
    q = urllib.parse.quote(ticker, safe="")
    p2 = int(time.time())
    p1 = int(pd.Timestamp(config.START).timestamp()) - 20 * 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    data = json.loads(_get(url))
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s").normalize()
    s = pd.Series(closes, index=idx).astype(float)
    s = s.dropna()  # drop weekend/holiday rows (Yahoo emits null closes for some tickers)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def compute_change(series, units):
    if units in ("pp", "usd"):
        return series.diff()
    if units == "pct":
        return 100.0 * np.log(series).diff()
    raise ValueError(f"unknown units: {units}")


def build_panel():
    """Fetch every variable and align everything to the anchor's trading days.

    Each level series is first restricted to the anchor's trading calendar and
    only *then* differenced, so a change is always trading-day-to-trading-day.
    (Some FRED series -- e.g. the HY OAS -- carry a few weekend/holiday rows;
    differencing those on their own calendar attributes the change to the wrong
    neighbour.)

    Returns (levels, changes) DataFrames indexed by date.
    """
    raw = {}
    for key, source, id_, units, name in config.VARIABLES:
        s = fetch_fred(id_) if source == "fred" else fetch_yahoo(id_)
        s = s[s.index >= pd.Timestamp(config.START)]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        raw[key] = s

    cal = raw[config.ANCHOR].index  # the anchor defines the trading calendar

    levels, changes = {}, {}
    for key, source, id_, units, name in config.VARIABLES:
        s = raw[key]
        s = s[s.index.isin(cal)]  # drop non-trading rows before differencing
        levels[key] = s.rename(key)
        changes[key] = compute_change(s, units).rename(key)

    levels = pd.DataFrame(levels).reindex(cal)
    changes = pd.DataFrame(changes).reindex(cal)
    changes = changes.dropna(subset=[config.ANCHOR])  # anchor defines trading days
    levels = levels.loc[changes.index]
    return levels, changes
