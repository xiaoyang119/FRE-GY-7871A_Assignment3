"""Configuration for the Iran war-risk replication (Rigobon & Sack 2003).

Applies heteroskedasticity-based identification to 2026 Iran war risk using
global market variables. Window: 2026-02-28 -> present (professor's spec).
"""
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# FinBERT: prefer a pre-downloaded local copy (skips a ~400 MB download), so we
# search a few candidate locations -- inside the project and up to two levels
# above it, which covers both <NLP>/.finbert-local and
# <NLP>/FRE-GY-7871A_Assignment3/iran_war_risk/ layouts.  Override with the
# FINBERT_MODEL environment variable.
FINBERT_MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")
if "FINBERT_MODEL" not in os.environ:
    _candidates = [
        ROOT / ".finbert-local",             # inside the project
        ROOT.parent / ".finbert-local",      # one level up
        ROOT.parent.parent / ".finbert-local",  # two levels up (original layout)
    ]
    for _cand in _candidates:
        if _cand.exists():
            FINBERT_MODEL = str(_cand)
            break

# Sample window start (professor: "Feb 28 to present").
START = "2026-02-28"

# Normalization anchor. The unobservable war-risk factor z1 is normalized to
# have a *unit* impact on this variable's daily change (as the paper normalizes
# on the 2-year Treasury yield).
ANCHOR = "DGS2"

# Global market variables.
#   key   : short column name
#   source: "fred" (fredgraph.csv, no key) or "yf" (Yahoo chart API)
#   id    : series id / ticker
#   units : how the daily change is computed
#           "pp"  -> level difference (percentage points: yields, spreads)
#           "pct" -> 100 * log-return (percent: equities, dollar index)
#           "usd" -> level difference (dollars: oil, gold)
#   name  : display name
VARIABLES = [
    ("DGS2",   "fred", "DGS2",         "pp",  "2-Year Treasury Yield"),
    ("DGS10",  "fred", "DGS10",        "pp",  "10-Year Treasury Yield"),
    ("T10YIE", "fred", "T10YIE",       "pp",  "10Y Breakeven Inflation"),
    ("SPX",    "yf",   "^GSPC",        "pct", "S&P 500"),
    ("SX5E",   "yf",   "^STOXX50E",    "pct", "Euro Stoxx 50"),
    ("BRENT",  "yf",   "BZ=F",         "usd", "Brent Crude Oil (front-month futures)"),
    ("GOLD",   "yf",   "GC=F",         "usd", "Gold"),
    ("DXY",    "yf",   "DX-Y.NYB",     "pct", "U.S. Dollar Index"),
    ("HYOAS",  "fred", "BAMLH0A0HYM2", "pp",  "US High-Yield OAS"),
    ("BBBOAS", "fred", "BAMLC0A4CBBB", "pp",  "US BBB (IG) OAS"),
]

# Magnitude of the war-risk shock used for reporting (moves the 2Y yield by
# -25 bp), as in the paper. All reported coefficients = d_j1 * SHOCK.
SHOCK = -0.25

def as_of():
    """Human-readable as-of stamp for outputs."""
    return datetime.now().strftime("%Y-%m-%d")
