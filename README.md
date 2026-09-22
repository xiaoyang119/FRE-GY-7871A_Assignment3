# Iran War Risk 2026 — Heteroskedasticity-Based Identification (Assignment 3)

**FRE-GY 7871 A · NLP and the Investment Process · Fall 2026**

Replication of Rigobon & Sack (2003), *The Effects of War Risk on U.S. Financial
Markets*, applied to the 2026 Iran war. The unobservable "war-risk factor" is
recovered with identification-through-heteroskedasticity (Rigobon 2003), using
**NLP on war news** (GDELT volume + FinBERT + a war lexicon) to flag high/low
news days instead of the paper's hand-read newspaper list.

## Pipeline

```
scripts/
  00_collect_market.py    FRED + Yahoo daily series (2026-02-28 -> present)
  01_collect_news.py      GDELT daily war-news volume (6 war phrases)
  02_flag_days.py         flag high-news days (1) vs low (0)
  03_headlines.py         fetch H-day headlines + classify bad/good (FinBERT)
  04_estimate.py          het-ID estimators + variance decomposition
  05_verify.py            independent verification + robustness checks
```

## Setup (uv)

`pyproject.toml` + `.python-version` pin Python 3.13 and the dependencies.
Create the venv and sync the core deps (pandas + numpy):

```powershell
uv sync
```

To also install FinBERT (torch + transformers — used only by `03_headlines.py`):

```powershell
uv sync --extra finbert
```

Then run the pipeline in order (the venv is used automatically via `uv run`):

```powershell
uv run python scripts/00_collect_market.py
uv run python scripts/01_collect_news.py      # GDELT rate-limits: may retry a few minutes
uv run python scripts/02_flag_days.py         # add --quantile 0.85 to tune the H/L split
uv run python scripts/03_headlines.py         # needs the finbert extra; --no-finbert = lexicon only
uv run python scripts/04_estimate.py
```

(Equivalently, use `.venv\Scripts\python.exe` directly after `uv sync`.)
A local FinBERT copy at `../../.finbert-local` is auto-detected, so no model
download is needed.

## Method (one paragraph)

For each pair (anchor = 2Y Treasury yield, variable *j*), split daily changes
into high-war-news days **H** and low **L**.  Under the identifying assumptions
(war-risk factor orthogonal to other factors; only its variance shifts across
H/L), the change in the variance-covariance matrix is

```
ΔΩ = Ω_H − Ω_L = Δσ²(z₁) · [[1, d],[d, d²]]
```

so the loading `d` is recovered as `ΔΩ21/ΔΩ11` (instrument from the anchor) and
`ΔΩ22/ΔΩ21` (instrument from variable *j*), combined as an inverse-variance
weighted average (w₃).  Coefficients are reported for a shock that moves the 2Y
yield −25 bp (× −0.25).  Table 3 reports the variance decomposition, with the
share of variance attributable to war news as a lower bound.

## Outputs

- `outputs/table1_war_news_days.csv` — the flagged high-news days (events + regime)
- `outputs/table2_sensitivity.csv`  — war-risk impact per variable (3 estimators)
- `outputs/table3_variance.csv`     — variance decomposition

See `REPORT.md` for the full write-up, including the evaluation of whether
heteroskedasticity-based identification is the best approach and the
alternatives considered.
