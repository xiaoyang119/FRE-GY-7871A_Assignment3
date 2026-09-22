"""Step 5: independent verification + robustness of the whole pipeline.

Walks the chain bottom-up and tests the *theory*, not just the code:
  1. data integrity   -- levels -> changes, anchor gaps
  2. rank-1 signature -- theory says DOmega21^2 = DOmega11*DOmega22
  3. recomputation    -- estimators from raw data vs the saved table
  4. placebo          -- is the real H split's anchor-variance shift special?
  5. quantile         -- does the flag threshold change the answer?
  6. variance algebra -- pred = d^2*DVar(anchor) reproduces Table 3
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, het_id, nlp  # noqa: E402

ANCHOR = "BRENT"


def mom(x1, x2, idx):
    a, b = x1[idx], x2[idx]
    return float((a * a).mean()), float((b * b).mean()), float((a * b).mean())


def main():
    levels = pd.read_csv("data/market_levels.csv", index_col=0, parse_dates=True)
    changes = pd.read_csv("data/market_changes.csv", index_col=0, parse_dates=True)
    flags = pd.read_csv("data/day_flags.csv", index_col=0, parse_dates=True)
    vol = pd.read_csv("data/war_news_volume.csv", index_col=0, parse_dates=True)

    h = flags[flags["is_high"] == 1].index
    l = flags[flags["is_high"] == 0].index
    h_pos = np.flatnonzero(changes.index.isin(h))
    l_pos = np.flatnonzero(changes.index.isin(l))

    units = {k: u for k, _, _, u, _ in config.VARIABLES}

    # ---------------------------------------------------------------- 1
    print("=" * 72)
    print("1) DATA INTEGRITY")
    print("=" * 72)
    print(f"levels {levels.shape} | changes {changes.shape} | "
          f"{changes.index.min().date()} .. {changes.index.max().date()}")
    for v in ["DGS2", "SPX", "BRENT", "HYOAS"]:
        manual = levels[v].diff() if units[v] in ("pp", "usd") else 100 * np.log(levels[v]).diff()
        a, b = changes[v], manual.reindex(changes.index)
        both = a.notna() & b.notna()   # the first saved row lacks its prior level
        ok = bool(np.allclose(a[both], b[both]))
        print(f"   {v:7s} units={units[v]:3s}  change==recomputed on {int(both.sum())} days? {ok}")
    print(f"   anchor={ANCHOR} NaN in changes: {int(changes[ANCHOR].isna().sum())}")
    print(f"   H={len(h_pos)} days, L={len(l_pos)} days")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 72)
    print("2) RANK-1 SIGNATURE  (theory: DOmega21^2 == DOmega11 * DOmega22)")
    print("=" * 72)
    print("   r = DOmega21^2 / (DOmega11*DOmega22);  r~1 => single-factor structure holds")
    xa = changes[ANCHOR].to_numpy()
    for v in changes.columns:
        if v == ANCHOR:
            continue
        x2 = changes[v].to_numpy()
        ok = ~(np.isnan(xa) | np.isnan(x2))
        hp = np.intersect1d(h_pos, np.flatnonzero(ok))
        lp = np.intersect1d(l_pos, np.flatnonzero(ok))
        d11 = mom(xa, xa, hp)[0] - mom(xa, xa, lp)[0]
        d21 = mom(xa, x2, hp)[2] - mom(xa, x2, lp)[2]
        d22 = mom(x2, x2, hp)[1] - mom(x2, x2, lp)[1]
        r = (d21 ** 2) / (d11 * d22) if d11 * d22 > 0 else np.nan
        print(f"   {v:7s}  DOmega11={d11:+9.5f}  DOmega21={d21:+9.5f}  "
              f"DOmega22={d22:+9.5f}  r={r:6.3f}")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 72)
    print("3) INDEPENDENT RECOMPUTATION vs saved table2_sensitivity_BRENT.csv")
    print("=" * 72)
    saved = pd.read_csv("outputs/table2_sensitivity_BRENT.csv").set_index("variable")
    print(f"   {'variable':8s} {'d_w1 hand':>10s} {'d_w1 file':>10s} "
          f"{'d_w2 hand':>10s} {'d_w2 file':>10s}")
    shock = 10.0
    for v in changes.columns:
        if v == ANCHOR:
            continue
        x2 = changes[v].to_numpy()
        ok = ~(np.isnan(xa) | np.isnan(x2))
        hp = np.intersect1d(h_pos, np.flatnonzero(ok))
        lp = np.intersect1d(l_pos, np.flatnonzero(ok))
        d11 = mom(xa, xa, hp)[0] - mom(xa, xa, lp)[0]
        d21 = mom(xa, x2, hp)[2] - mom(xa, x2, lp)[2]
        d22 = mom(x2, x2, hp)[1] - mom(x2, x2, lp)[1]
        w1, w2 = d21 / d11 * shock, d22 / d21 * shock
        print(f"   {v:8s} {w1:10.3f} {saved.loc[v,'d_w1']:10.3f} "
              f"{w2:10.3f} {saved.loc[v,'d_w2']:10.3f}")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 72)
    print("4) PLACEBO: is the real H split's anchor-variance shift special?")
    print("=" * 72)
    ok = ~np.isnan(xa)
    pos = np.flatnonzero(ok)
    nH = len(h_pos)
    real = mom(xa, xa, h_pos)[0] - mom(xa, xa, np.intersect1d(l_pos, pos))[0]
    rng = np.random.default_rng(0)
    draws = np.empty(5000)
    for i in range(5000):
        pick = np.sort(rng.choice(pos, nH, replace=False))
        rest = np.setdiff1d(pos, pick)
        draws[i] = (xa[pick] ** 2).mean() - (xa[rest] ** 2).mean()
    p = float((draws >= real).mean())
    print(f"   real DOmega11({ANCHOR})        = {real:+.4f}")
    print(f"   5000 random {nH}-day splits: mean={draws.mean():+.4f}  "
          f"sd={draws.std():.4f}  max={draws.max():+.4f}")
    print(f"   placebo p-value = P[random >= real] = {p:.4f}  "
          f"-> {'EXTREME (signal)' if p < 0.01 else 'NOT special'}")

    # ---------------------------------------------------------------- 5
    print()
    print("=" * 72)
    print("5) QUANTILE SENSITIVITY OF THE FLAG")
    print("=" * 72)
    intensity = vol["war_intensity"].reindex(changes.index).fillna(0)
    print(f"   {'q':>5s} {'nH':>4s} {'DOmega11':>10s} {'SX5E d_w3':>10s} "
          f"{'HYOAS d_w3':>11s} {'DXY d_w3':>9s}")
    for q in [0.80, 0.85, 0.90, 0.95]:
        f = nlp.flag_high_days(intensity, quantile=q)
        hd, ld = f[f == 1].index, f[f == 0].index
        hp = np.flatnonzero(changes.index.isin(hd))
        lp = np.flatnonzero(changes.index.isin(ld))
        d11 = mom(xa, xa, np.intersect1d(hp, pos))[0] - mom(xa, xa, np.intersect1d(lp, pos))[0]
        est = het_id.estimate_table(changes, hd, ld, anchor=ANCHOR, shock=shock).set_index("variable")
        print(f"   {q:5.2f} {len(hp):4d} {d11:10.3f} {est.loc['SX5E','d_w3']:10.3f} "
              f"{est.loc['HYOAS','d_w3']:11.3f} {est.loc['DXY','d_w3']:9.3f}")

    # ---------------------------------------------------------------- 6
    print()
    print("=" * 72)
    print("6) VARIANCE-DECOMPOSITION ALGEBRA  (Table 3)")
    print("=" * 72)
    t3 = pd.read_csv("outputs/table3_variance_BRENT.csv").set_index("variable")
    est = het_id.estimate_table(changes, h, l, anchor=ANCHOR, shock=shock).set_index("variable")
    dvar = mom(xa, xa, np.intersect1d(h_pos, pos))[0] - mom(xa, xa, np.intersect1d(l_pos, pos))[0]
    print(f"   DVar({ANCHOR}) = {dvar:.4f}   (war-risk variance increment)")
    for v in ["SX5E", "HYOAS", "DXY"]:
        d = est.loc[v, "raw_d_w3"]
        pred = d ** 2 * dvar
        vh = (changes[v].to_numpy() ** 2)[h_pos].mean()
        print(f"   {v:7s} d={d:+.4f}  pred=hand {pred:.4f} / file {t3.loc[v,'pred_change']:.4f}"
              f"   pct_H hand {pred/vh:.4f} / file {t3.loc[v,'pct_H']:.4f}")


if __name__ == "__main__":
    main()
