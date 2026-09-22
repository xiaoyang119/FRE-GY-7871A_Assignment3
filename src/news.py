"""War-news collector via the GDELT DOC 2.0 API (free, no key).

Two endpoints are used:
  * ``timelinevolraw``  -> daily article counts (used to flag high-news days)
  * ``artlist``         -> article titles/urls (used to run FinBERT + lexicon
                           and to split high-news days into bad/good war news)

GDELT is keyless but rate-limits by IP (HTTP 429).  The collector is therefore
**resilient and resumable**:
  * each query's series is cached to ``data/_vol_<query>.csv`` as soon as it
    succeeds, and skipped on re-run (so you never re-fetch what you already have);
  * a query that still 429s after backoff is *skipped*, the rest continue, and
    you just re-run later to fill in the gaps;
  * requests are spaced ~15s apart and 429 waits are long (25s, 50s, 75s...).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (academic research script)"}

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# War-focused search phrases for the 2026 Iran conflict.  Kept as phrases so
# that generic "Iran" news (sanctions, elections, sports) is down-weighted.
WAR_QUERIES = [
    "iran war",
    "iran israel",
    "iran strike",
    "iran missile",
    "iran nuclear",
    "strait of hormuz",
]


def _get(url, retries=4, sleep=25.0):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = sleep * (attempt + 1)
                print(f"    429 rate-limited -> wait {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("unreachable")


def gdelt_timeline(query, start="2026-02-28", end="2026-09-30"):
    """Daily article-count Series for a query over [start, end]."""
    q = urllib.parse.quote(query)
    url = (
        f"{BASE}?query={q}&mode=timelinevolraw&format=json"
        f"&STARTDATETIME={start.replace('-', '')}000000"
        f"&ENDDATETIME={end.replace('-', '')}000000"
    )
    data = json.loads(_get(url))
    rows = []
    for item in data.get("timeline", []):
        for d in item.get("data", []):
            rows.append((pd.Timestamp(d["date"]).date(), int(d["value"])))
    s = pd.Series({k: v for k, v in rows}, name=query).sort_index()
    return s


def gdelt_articles(query, start, end, maxrecords=200, retries=3, sleep=20):
    """List of {date, title, url, domain, sourcecountry} for a query/day."""
    q = urllib.parse.quote(query)
    url = (
        f"{BASE}?query={q}&mode=artlist&format=json"
        f"&STARTDATETIME={start.replace('-', '')}000000"
        f"&ENDDATETIME={end.replace('-', '')}235959"
        f"&maxrecords={maxrecords}"
    )
    data = json.loads(_get(url, retries=retries, sleep=sleep))
    out = []
    for a in data.get("articles", []):
        seen = a.get("seendate", "")[:8]
        date = pd.Timestamp(seen).date() if len(seen) == 8 else None
        out.append({
            "date": date,
            "title": (a.get("title") or "").strip(),
            "url": a.get("url", ""),
            "domain": a.get("domain", ""),
            "language": a.get("language", ""),
            "sourcecountry": a.get("sourcecountry", ""),
        })
    return out


def collect_volume(queries=WAR_QUERIES, start="2026-02-28", end="2026-09-30",
                   out_csv="data/war_news_volume.csv", delay=15.0):
    """Daily volume per query + a combined war-news intensity.

    Resumable: successful queries are cached to data/_vol_<query>.csv and
    skipped on the next run; failed queries are skipped and retried later.
    """
    os.makedirs("data", exist_ok=True)
    cols = {}
    for q in queries:
        cache = f"data/_vol_{q.replace(' ', '_')}.csv"
        if os.path.exists(cache):
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
            print(f"[cached] {q!r}: {len(s)} days")
        else:
            try:
                s = gdelt_timeline(q, start, end)
                s.to_csv(cache)
                print(f"[ok]     {q!r}: {len(s)} days")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL]   {q!r}: {type(e).__name__} {str(e)[:80]} "
                      f"(skipped; re-run later to retry)")
                continue
            time.sleep(delay)
        cols[q] = s

    if not cols:
        raise RuntimeError(
            "all queries failed (GDELT 429?); wait 10-15 min and re-run — "
            "already-cached queries are skipped, so progress is kept")

    vol = pd.DataFrame(cols).sort_index().fillna(0)
    vol["war_intensity"] = vol[list(cols.keys())].max(axis=1)  # max avoids double count
    vol["total"] = vol[list(cols.keys())].sum(axis=1)
    vol.index.name = "date"
    vol.to_csv(out_csv)
    print(f"\nsaved {out_csv} ({vol.shape[0]} days x {vol.shape[1]} cols)")
    return vol
