"""Heteroskedasticity-based identification (Rigobon & Sack 2003).

For each pair (anchor = 2Y yield, variable j) we split daily changes into
high-war-news days (H) and low-war-news days (L).  Under the identifying
assumptions (war-risk factor orthogonal to other factors; only its variance
shifts across H/L), the change in the variance-covariance matrix is

    ΔΩ = Ω_H − Ω_L = Δσ²(z1) · [[1, d],[d, d²]]

so the loading d of variable j on the war-risk factor is recovered two ways:

    d_w1 = ΔΩ21 / ΔΩ11          (instrument built from the anchor)
    d_w2 = ΔΩ22 / ΔΩ21          (instrument built from variable j)

and combined (w3) as an inverse-variance-weighted average.  Reported
coefficients are scaled by SHOCK (= −0.25), i.e. the impact of a war-risk
increase large enough to move the 2Y yield −25 bp.  Variances are mean squares
(mean of Δx², mean of Δx1·Δx2), matching the paper's footnote 12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _mom(x1, x2, idx):
    """Var(x1), Var(x2), Cov(x1,x2) over positional index `idx` (mean squares)."""
    a = x1[idx].astype(float)
    b = x2[idx].astype(float)
    return float((a * a).mean()), float((b * b).mean()), float((a * b).mean())


def pair_estimate(x1, x2, h_idx, l_idx, n_boot=3000, seed=0):
    """Estimate d (loading of var on war risk) + bootstrap standard errors.

    x1: anchor series (np.array).  x2: variable series (np.array).
    h_idx / l_idx: positional indices of high / low news days.
    """
    v1h, v2h, ch = _mom(x1, x2, h_idx)
    v1l, v2l, cl = _mom(x1, x2, l_idx)

    d11 = v1h - v1l
    d21 = ch - cl
    d22 = v2h - v2l

    d_w1 = d21 / d11 if d11 != 0 else np.nan
    d_w2 = d22 / d21 if d21 != 0 else np.nan

    # non-parametric bootstrap over H and L separately
    rng = np.random.default_rng(seed)
    h = np.asarray(h_idx)
    l = np.asarray(l_idx)
    b1 = np.empty(n_boot)
    b2 = np.empty(n_boot)
    for b in range(n_boot):
        hh = rng.choice(h, len(h), replace=True)
        ll = rng.choice(l, len(l), replace=True)
        v1h_, v2h_, ch_ = _mom(x1, x2, hh)
        v1l_, v2l_, cl_ = _mom(x1, x2, ll)
        d11_ = v1h_ - v1l_
        d21_ = ch_ - cl_
        d22_ = v2h_ - v2l_
        b1[b] = d21_ / d11_ if d11_ != 0 else np.nan
        b2[b] = d22_ / d21_ if d21_ != 0 else np.nan

    se1 = float(np.nanstd(b1))
    se2 = float(np.nanstd(b2))

    # inverse-variance combination (w3), guarding degenerate SEs
    w1 = 1.0 / se1 ** 2 if se1 > 0 else 1.0
    w2 = 1.0 / se2 ** 2 if se2 > 0 else 1.0
    d_w3 = (w1 * d_w1 + w2 * d_w2) / (w1 + w2)
    se3 = float(np.sqrt(1.0 / (w1 + w2)))

    return {
        "v1h": v1h, "v1l": v1l, "v2h": v2h, "v2l": v2l,
        "cov_h": ch, "cov_l": cl,
        "d_w1": d_w1, "d_w2": d_w2, "d_w3": d_w3,
        "se_w1": se1, "se_w2": se2, "se_w3": se3,
        "t_w1": abs(d_w1 / se1) if se1 > 0 else np.nan,
        "t_w2": abs(d_w2 / se2) if se2 > 0 else np.nan,
        "t_w3": abs(d_w3 / se3) if se3 > 0 else np.nan,
    }


def estimate_table(changes: pd.DataFrame, h_dates, l_dates,
                   anchor: str = None, shock: float = None):
    """Table 2 rows: war-risk impact on every non-anchor variable.

    Returns a DataFrame with one row per variable (coefficients already
    multiplied by `shock`, so they read as the effect of a −25 bp 2Y shock).
    """
    anchor = anchor or config.ANCHOR
    shock = shock if shock is not None else config.SHOCK

    all_dates = changes.index
    h_pos = np.flatnonzero(all_dates.isin(h_dates))
    l_pos = np.flatnonzero(all_dates.isin(l_dates))
    if len(h_pos) == 0 or len(l_pos) == 0:
        raise ValueError("empty H or L day set after alignment to trading days")

    x1 = changes[anchor].to_numpy()
    rows = []
    for var in changes.columns:
        if var == anchor:
            continue
        x2 = changes[var].to_numpy()
        # keep only days where both anchor and var are non-null
        ok = ~(np.isnan(x1) | np.isnan(x2))
        h_idx = np.intersect1d(h_pos, np.flatnonzero(ok))
        l_idx = np.intersect1d(l_pos, np.flatnonzero(ok))
        if len(h_idx) < 2 or len(l_idx) < 2:
            rows.append({"variable": var})
            continue
        e = pair_estimate(x1, x2, h_idx, l_idx)
        rows.append({
            "variable": var,
            "n_H": len(h_idx), "n_L": len(l_idx),
            "d_w1": e["d_w1"] * shock,
            "d_w2": e["d_w2"] * shock,
            "d_w3": e["d_w3"] * shock,
            "t_w1": e["t_w1"], "t_w2": e["t_w2"], "t_w3": e["t_w3"],
            "se_w3": e["se_w3"] * abs(shock),
            "raw_d_w3": e["d_w3"],
            "v2h": e["v2h"], "v2l": e["v2l"],
        })
    return pd.DataFrame(rows)


def variance_table(changes: pd.DataFrame, h_dates, l_dates, est_df,
                   anchor: str = None):
    """Table 3 rows: variance decomposition.

    Columns: Var on L days, Var on H days, predicted change in variance
    (d² × ΔVar(anchor)), % explained on H days, % explained on all days.
    """
    anchor = anchor or config.ANCHOR
    all_dates = changes.index
    h_pos = np.flatnonzero(all_dates.isin(h_dates))
    l_pos = np.flatnonzero(all_dates.isin(l_dates))

    x1 = changes[anchor].to_numpy()
    # the anchor itself can have gaps (e.g. Brent futures on a holiday), so
    # restrict its moments to non-NaN days or dvar_anchor silently becomes NaN
    ok1 = ~np.isnan(x1)
    h1 = np.intersect1d(h_pos, np.flatnonzero(ok1))
    l1 = np.intersect1d(l_pos, np.flatnonzero(ok1))
    v1h, _, _ = _mom(x1, x1, h1)
    v1l, _, _ = _mom(x1, x1, l1)
    dvar_anchor = v1h - v1l

    dmap = dict(zip(est_df["variable"], est_df["raw_d_w3"]))
    rows = []
    for var in changes.columns:
        x2 = changes[var].to_numpy()
        ok = ~np.isnan(x2)
        h_idx = np.intersect1d(h_pos, np.flatnonzero(ok))
        l_idx = np.intersect1d(l_pos, np.flatnonzero(ok))
        v2h = float((x2[h_idx] ** 2).mean())
        v2l = float((x2[l_idx] ** 2).mean())
        if var == anchor:
            rows.append({"variable": var, "var_L": v2l, "var_H": v2h,
                         "pred_change": np.nan, "pct_H": np.nan, "pct_all": np.nan})
            continue
        d = dmap.get(var, np.nan)
        pred = (d ** 2) * dvar_anchor
        pct_h = pred / v2h if v2h > 0 else np.nan
        total = len(h_idx) * v2h + len(l_idx) * v2l
        pct_all = (len(h_idx) * pred) / total if total > 0 else np.nan
        rows.append({"variable": var, "var_L": v2l, "var_H": v2h,
                     "pred_change": pred, "pct_H": pct_h, "pct_all": pct_all})
    return pd.DataFrame(rows)
