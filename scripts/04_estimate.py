"""Step 4: heteroskedasticity-based estimation + variance decomposition.

Produces the paper's Tables 1, 2 and 3 for the 2026 Iran war risk.

The normalization anchor matters: the paper anchors on the 2-year Treasury
yield, but in 2026 the war-news days did NOT have elevated Treasury-yield
variance (ΔΩ11 < 0), which breaks identification.  Anchoring on Brent oil --
the war's primary channel, with a large positive variance shift -- restores it.

Usage:
    python scripts/04_estimate.py                              # anchor = DGS2
    python scripts/04_estimate.py --anchor BRENT --shock 10    # oil anchor
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config, het_id  # noqa: E402

# Reporting shock per anchor: the size of the anchor move that defines the
# "unit" war-risk increase.  DGS2: -25 bp (as in the paper).  BRENT: +$10.
DEFAULT_SHOCK = {
    "DGS2": -0.25, "DGS10": -0.25, "T10YIE": -0.10, "BRENT": 10.0,
    "SPX": -1.0, "SX5E": -1.0, "DXY": 1.0, "GOLD": 10.0,
    "HYOAS": 0.10, "BBBOAS": 0.05,
}


def main(anchor=None, shock=None):
    os.makedirs("outputs", exist_ok=True)

    anchor = anchor or config.ANCHOR
    shock = shock if shock is not None else DEFAULT_SHOCK.get(anchor, config.SHOCK)

    changes = pd.read_csv("data/market_changes.csv", index_col=0, parse_dates=True)
    flags = pd.read_csv("data/day_flags.csv", index_col=0, parse_dates=True)

    h_dates = flags[flags["is_high"] == 1].index
    l_dates = flags[flags["is_high"] == 0].index

    names = {k: n for k, _, _, _, n in config.VARIABLES}
    units = {k: u for k, _, _, u, _ in config.VARIABLES}

    # ---- sanity check: anchor variance MUST rise on H days ----
    x1 = changes[anchor]
    vh = (x1.loc[x1.index.isin(h_dates)] ** 2).mean()
    vl = (x1.loc[x1.index.isin(l_dates)] ** 2).mean()
    ok = (vh - vl) > 0
    print(f"anchor ({anchor}) variance: H={vh:.6f}  L={vl:.6f}  Δ={vh - vl:+.6f}  "
          f"--> {'OK' if ok else 'FAILS (variance must RISE on H days)'}")

    # ---- estimation ----
    est_full = het_id.estimate_table(changes, h_dates, l_dates,
                                     anchor=anchor, shock=shock)
    suffix = "" if anchor == config.ANCHOR else f"_{anchor}"

    # ---- Table 1: war-news days ----
    table1 = pd.DataFrame({"date": h_dates})
    if os.path.exists("data/day_regimes.csv"):
        reg = pd.read_csv("data/day_regimes.csv", parse_dates=["date"])
        reg["date"] = pd.to_datetime(reg["date"])
        table1 = table1.merge(reg, on="date", how="left")
    table1.to_csv("outputs/table1_war_news_days.csv", index=False)

    # ---- Table 2 ----
    t2 = est_full.copy()
    t2["name"] = t2["variable"].map(names)
    t2["units"] = t2["variable"].map(units)
    t2 = t2[["variable", "name", "units", "n_H", "n_L",
             "d_w1", "d_w2", "d_w3", "t_w1", "t_w2", "t_w3"]]
    t2.to_csv(f"outputs/table2_sensitivity{suffix}.csv", index=False)

    # ---- Table 3 ----
    t3 = het_id.variance_table(changes, h_dates, l_dates, est_full, anchor=anchor)
    t3["name"] = t3["variable"].map(names)
    t3 = t3[["variable", "name", "var_L", "var_H", "pred_change", "pct_H", "pct_all"]]
    t3.to_csv(f"outputs/table3_variance{suffix}.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\nWindow {changes.index.min().date()} -> {changes.index.max().date()}, "
          f"{len(changes)} days; H={len(h_dates)}, L={len(l_dates)}; "
          f"anchor={anchor}, shock={shock}\n")
    print("=== TABLE 2: impact of a war-risk shock ===")
    print(t2.round(3).to_string(index=False))
    print("\n=== TABLE 3: variance decomposition ===")
    print(t3.round(4).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--anchor") + 1].upper() if "--anchor" in sys.argv else None
    s = float(sys.argv[sys.argv.index("--shock") + 1]) if "--shock" in sys.argv else None
    main(anchor=a, shock=s)
