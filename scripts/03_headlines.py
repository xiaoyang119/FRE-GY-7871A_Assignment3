"""Step 3: fetch headlines for high-news days + classify (bad/good war news).

Resumable: each H day's headlines are cached to data/_hl_<YYYYMMDD>.csv and
skipped on re-run, so you can Ctrl-C and resume any time.

**English-only**: GDELT indexes 65+ languages, but our war lexicon and FinBERT
are English, so the artlist query carries ``sourcelang:english`` and results are
additionally filtered on the article's language field.  Use ``--refresh`` to
re-fetch days whose cache was built before the language filter existed.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import news, nlp  # noqa: E402

ARTLIST_QUERY = "iran war sourcelang:english"
MAXRECORDS = 25


def main(use_finbert=True, refresh=False):
    flags = pd.read_csv("data/day_flags.csv", index_col=0, parse_dates=True)
    h_dates = flags[flags["is_high"] == 1].index
    os.makedirs("data", exist_ok=True)

    all_rows = []
    for d in h_dates:
        ds = d.strftime("%Y-%m-%d")
        cache = f"data/_hl_{d.strftime('%Y%m%d')}.csv"
        arts = None
        if os.path.exists(cache):
            cached = pd.read_csv(cache)
            if refresh and "language" not in cached.columns:
                os.remove(cache)  # legacy multilingual cache -> re-fetch
                print(f"[stale]  {ds}: re-fetching (old cache has no language column)")
            else:
                arts = cached.to_dict("records")
                print(f"[cached] {ds}: {len(arts)}")
        if arts is None:
            try:
                arts = news.gdelt_articles(ARTLIST_QUERY, ds, ds,
                                           maxrecords=MAXRECORDS, retries=3, sleep=20)
                # client-side guard in case the query operator is not honoured
                kept = [a for a in arts
                        if str(a.get("language", "")).lower().startswith("eng")
                        or not a.get("language")]
                arts = kept
                pd.DataFrame(arts).to_csv(cache, index=False)  # cache only on success
                print(f"[ok]     {ds}: {len(arts)} English")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL]   {ds}: {type(e).__name__} (skipped; re-run later)")
                arts = []
            time.sleep(6.0)
        for a in arts:
            all_rows.append({"date": pd.Timestamp(ds), "title": a.get("title", ""),
                             "domain": a.get("domain", ""),
                             "language": a.get("language", ""),
                             "sourcecountry": a.get("sourcecountry", "")})

    hl = pd.DataFrame(all_rows)
    if hl.empty:
        print("no headlines yet; re-run later to retry the failed days")
        hl.to_csv("data/headlines.csv", index=False)
        return

    print(f"\nclassifying {len(hl)} headlines (lexicon + FinBERT)...")
    scored = None
    if use_finbert:
        try:
            scored = nlp.classify_headlines(hl["title"].tolist())
        except ImportError as e:  # transformers/torch not installed
            print(f"  ! FinBERT unavailable ({e}); falling back to lexicon-only.")
            print("    enable FinBERT later with:  uv sync --extra finbert")
            use_finbert = False
    if scored is None:
        scored = [{"lex_net": nlp.lexicon_direction(t), "fb_pos": 0.0,
                   "fb_neg": 0.0, "fb_neu": 0.0,
                   "regime": ("bad" if nlp.lexicon_direction(t) > 0 else
                              ("good" if nlp.lexicon_direction(t) < 0 else "unclear"))}
                  for t in hl["title"]]
    hl = pd.concat([hl, pd.DataFrame(scored)], axis=1)
    hl.to_csv("data/headlines.csv", index=False)

    g = hl.groupby("date").agg(
        n_headlines=("title", "size"),
        lex_net=("lex_net", "sum"),
        fb_pos=("fb_pos", "mean"),
        fb_neg=("fb_neg", "mean"),
        top_title=("title", "first"),
    ).reset_index()
    g["regime"] = g.apply(
        lambda r: "bad" if r["lex_net"] > 0 else ("good" if r["lex_net"] < 0 else "unclear"),
        axis=1,
    )
    g.to_csv("data/day_regimes.csv", index=False)
    print("\nsaved data/headlines.csv + data/day_regimes.csv")
    print(g.to_string())


if __name__ == "__main__":
    main(use_finbert="--no-finbert" not in sys.argv,
         refresh="--refresh" in sys.argv)
