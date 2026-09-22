"""Step 2: flag high/low war-news days from GDELT volume (professor's 1/0 flag).

Usage:
    python scripts/02_flag_days.py [--quantile 0.9]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import nlp  # noqa: E402


def main(quantile=0.9):
    vol = pd.read_csv("data/war_news_volume.csv", index_col=0, parse_dates=True)
    mkt = pd.read_csv("data/market_changes.csv", index_col=0, parse_dates=True)

    intensity = vol["war_intensity"].reindex(mkt.index).fillna(0)

    print("war_intensity over trading days (percentiles):")
    for p in [50, 70, 80, 85, 90, 95, 99]:
        print(f"  p{p}: {intensity.quantile(p / 100):8.0f}")
    print(f"  median = {intensity.median():.0f}   max = {intensity.max():.0f}")

    # Primary flag: top `quantile` of war-news intensity among trading days.
    flag = nlp.flag_high_days(intensity, quantile=quantile)

    h_dates = flag[flag == 1].index
    l_dates = flag[flag == 0].index
    print(f"\nH (high-news) days: {len(h_dates)}, L (low-news) days: {len(l_dates)}")
    print("H dates:", [d.strftime("%Y-%m-%d") for d in h_dates])

    out = pd.DataFrame({"war_intensity": intensity, "is_high": flag})
    out.index.name = "date"
    out.to_csv("data/day_flags.csv")
    print("\nsaved data/day_flags.csv")


if __name__ == "__main__":
    q = 0.9
    if "--quantile" in sys.argv:
        q = float(sys.argv[sys.argv.index("--quantile") + 1])
    main(quantile=q)
