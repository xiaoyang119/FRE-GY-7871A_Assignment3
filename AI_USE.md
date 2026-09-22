# AI Use Disclosure

**FRE-GY 7871 A — Assignment 3: Iran War Risk 2026**
*Heteroskedasticity-based identification (Rigobon & Sack 2003) with NLP day-flagging*

This project was built in one extended session with an AI coding assistant.
I am disclosing what it did and what I did.

## 1. Tools used

- **DeepSeek** (via the DeepSeek Harness) — the main assistant: read the papers,
  wrote essentially all of the code, ran it, diagnosed the results, and drafted
  the report.
- **Web search / fetch** (through the same harness) — to locate 2026 Iran-war
  coverage and cross-check the GDELT news volume.
- No other AI tools.

> Note: the brief itself asks to *"ask Claude whether heteroskedasticity-based
> identification is the best approach"*, so §6 of `REPORT.md` (the method
> evaluation and the alternatives table) is an explicitly **assigned** AI
> consultation rather than an undisclosed shortcut.

## 2. What the AI produced

| Area | Detail |
|---|---|
| **Paper reading** | Summarised Rigobon (2003) and Rigobon & Sack (2003); extracted the reduced-form setup, the identification `ΔΩ = Δσ²(z₁)·[[1,d],[d,d²]]`, the H/L day design, and the Table 1/2/3 definitions. |
| **Data collection** | Wrote the FRED + Yahoo market collectors and the GDELT war-news collector (rate-limit aware, resumable across runs). |
| **NLP** | Built the war lexicon, the FinBERT sentiment layer, the high/low day flag, the bad/good regime classifier, and the English-language filter. |
| **Econometrics** | Implemented the two just-identified estimators `d_w1 = ΔΩ₂₁/ΔΩ₁₁` and `d_w2 = ΔΩ₂₂/ΔΩ₂₁`, the combined `d_w3`, bootstrap standard errors, the variance decomposition, and the anchor switch. |
| **Verification** | Wrote `scripts/05_verify.py`: data-integrity recomputation, the rank-1 test, a 5 000-draw placebo, and threshold sensitivity. |
| **Report** | Drafted `REPORT.md`, including the method evaluation and the alternatives table. |
| **Setup** | `pyproject.toml` / `uv` environment, `.gitignore`, `README.md`, and this file. |

## 3. What I did

- Set the task framing and **approved each analytical decision**: the window
  (28 Feb 2026 → present), the top-decile news flag, the three-regime extension
  and — most importantly — **switching the normalization anchor from the 2-year
  Treasury yield to Brent crude**, after the AI showed that the paper's own
  anchor fails in 2026 (`ΔΩ₁₁ < 0`).
- **Ran the whole pipeline** in a `uv` environment (`uv sync`, then
  `scripts/00` → `05`), including all the GDELT fetches and the FinBERT step.
- **Checked the outputs at every step**: the H/L day list, the headline corpus
  and its language mix, and the three tables in `outputs/`.
- Read and revised the AI's draft report, and I take responsibility for every
  claim and number in the submission.

## 4. Errors found and fixed while checking

For transparency, the first drafts contained mistakes that were caught by
running the code and re-checking it rather than by trusting it:

1. `urllib.error` was used without an explicit import (a static-analysis warning;
   the code happened to run).
2. `variance_table` divided by a NaN anchor variance whenever the anchor had a
   missing day, which silently turned Table 3 into NaNs.
3. `build_panel` differenced each series on its **own** calendar before aligning
   them, so a FRED HY-OAS value that falls on a Sunday was differenced against
   the wrong neighbour. Fixed by restricting every series to the anchor's trading
   days *before* differencing. The conclusions were unchanged.
4. The first headline pull carried no language filter and returned a **97 %
   non-English** corpus, which the English lexicon and FinBERT could not score.
   Fixed with `sourcelang:english` plus a client-side language guard.

## 5. Extent

**Substantially all of the code, and the first draft of the report, were written
by the AI.** My role was to direct it, run it, verify the results, and own the
final submission. The econometric method comes from the two assigned readings
and the data are public (FRED, Yahoo Finance, GDELT), so the analysis is
reproducible from the repository.
