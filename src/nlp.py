"""NLP for war news: war lexicon + FinBERT, high/low day flagging, 3-regime.

Core requirement (professor): flag high-news days 1/0 from daily war-news
volume.  The direction of each story is NOT needed for identification (the
effect comes from the *shift in variance*), but direction is used for the
"three-regime" extension (bad / good / no war news).

Two scoring layers, addressing the professor's "novel phrasing" concern:
  1. a transparent escalation/de-escalation lexicon (fast, interpretable), and
  2. FinBERT sentiment (ProsusAI/finbert, local) as a model-based signal that
     is not limited to dictionary entries.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import FINBERT_MODEL

# ---------------------------------------------------------------------------
# War-risk lexicon
# ---------------------------------------------------------------------------
# risk-UP  ("bad war news"): escalation, strikes, nuclear, closure, etc.
ESCALATION = [
    "strike", "strikes", "striking", "airstrike", "airstrikes", "air strike",
    "attack", "attacks", "attacked", "assault", "offensive", "invasion",
    "invade", "invaded", "missile", "missiles", "ballistic", "drone",
    "drones", "warhead", "warheads", "nuclear", "uranium", "enrichment",
    "enriched", "centrifuge", "centrifuges", "reactor", "reactors",
    "weapons-grade", "escalation", "escalate", "escalates", "escalating",
    "retaliation", "retaliate", "retaliatory", "tit-for-tat", "bomb",
    "bombing", "bombed", "bombard", "artillery", "shelling", "shelled",
    "barrage", "casualties", "killed", "kills", "death toll", "fatalities",
    "wounded", "blockade", "blockaded", "closure", "strait of hormuz",
    "hormuz", "shipping lane", "ultimatum", "mobilize", "mobilization",
    "maximum pressure", "red line", "red lines", "snapback", "snap-back",
]
# risk-DOWN ("good war news"): ceasefire, talks, deal, withdrawal, relief, ...
DEESCALATION = [
    "ceasefire", "cease-fire", "truce", "armistice", "stop the fighting",
    "talks", "negotiation", "negotiations", "negotiate", "negotiating",
    "diplomacy", "diplomatic", "diplomats", "peace talks", "deal",
    "agreement", "accord", "framework", "prisoner swap", "hostage deal",
    "de-escalation", "deescalation", "de-escalate", "deescalate",
    "de-escalating", "peace", "peaceful", "peace plan", "peacekeeping",
    "detente", "détente", "withdrawal", "withdraw", "withdrawing",
    "pull back", "pulls back", "pullback", "humanitarian", "aid corridor",
    "aid corridors", "corridor", "surrender", "surrender", "capitulate",
    "sanctions relief", "sanctions eased", "eased sanctions", "embargo lifted",
]

_ESC_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in ESCALATION) + r")\b", re.I)
_DEE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in DEESCALATION) + r")\b", re.I)


def lexicon_direction(text: str) -> int:
    """Net war-news direction: >0 escalation (risk up), <0 de-escalation."""
    text = text or ""
    up = len(_ESC_RE.findall(text))
    down = len(_DEE_RE.findall(text))
    return up - down


# ---------------------------------------------------------------------------
# FinBERT sentiment (lazy, cached)
# ---------------------------------------------------------------------------
_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline
        _PIPE = pipeline("sentiment-analysis", model=FINBERT_MODEL, top_k=None)
    return _PIPE


def finbert_sentiment(text: str) -> dict:
    """pos/neg/neu probabilities for a short text (one headline)."""
    text = (text or "")[:512]
    if not text.strip():
        return {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    out = _pipeline()(text)[0]  # list of {label, score}
    res = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    for r in out:
        if r["label"] == "positive":
            res["pos"] = r["score"]
        elif r["label"] == "negative":
            res["neg"] = r["score"]
        else:
            res["neu"] = r["score"]
    return res


def finbert_sentiment_batch(texts) -> list:
    """Batch FinBERT over headlines (faster than one-by-one)."""
    texts = [((t or "")[:512]).strip() for t in texts]
    rows = []
    for i in range(0, len(texts), 32):
        for out in _pipeline()(texts[i:i + 32]):
            res = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
            for r in out:
                if r["label"] == "positive":
                    res["pos"] = r["score"]
                elif r["label"] == "negative":
                    res["neg"] = r["score"]
                else:
                    res["neu"] = r["score"]
            rows.append(res)
    return rows


# ---------------------------------------------------------------------------
# High/low day flagging
# ---------------------------------------------------------------------------
def flag_high_days(intensity: pd.Series, quantile: float = 0.9) -> pd.Series:
    """Binary flag: 1 = high war-news day (top `quantile` of war intensity).

    `intensity` is a daily Series (calendar days).  Only days with a positive
    intensity are eligible so that long quiet stretches are not forced in.
    """
    positive = intensity[intensity > 0]
    if positive.empty:
        return pd.Series(0, index=intensity.index, name="is_high")
    thresh = positive.quantile(quantile)
    flag = (intensity >= thresh).astype(int)
    flag.name = "is_high"
    return flag


def classify_headline(title: str) -> dict:
    """Lexicon + FinBERT direction for one headline."""
    d = lexicon_direction(title)
    fb = finbert_sentiment(title)
    # combined: escalation lex -> bad; de-escalation lex -> good; else FinBERT sign
    if d > 0:
        regime = "bad"
    elif d < 0:
        regime = "good"
    else:
        regime = "bad" if fb["neg"] > fb["pos"] else "good"
    return {"lex_net": d, "fb_pos": fb["pos"], "fb_neg": fb["neg"],
            "fb_neu": fb["neu"], "regime": regime}


def classify_headlines(headlines: list[str]) -> list:
    """Batch-classify headlines (lexicon + FinBERT)."""
    fbs = finbert_sentiment_batch(headlines)
    out = []
    for title, fb in zip(headlines, fbs):
        d = lexicon_direction(title)
        if d > 0:
            regime = "bad"
        elif d < 0:
            regime = "good"
        else:
            regime = "bad" if fb["neg"] > fb["pos"] else "good"
        out.append({"lex_net": d, "fb_pos": fb["pos"], "fb_neg": fb["neg"],
                    "fb_neu": fb["neu"], "regime": regime})
    return out
