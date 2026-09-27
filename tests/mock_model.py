"""A tiny stand-in for hr_pipeline.sentiment.SentimentModel, used ONLY for
local smoke-testing the rest of the pipeline (schema/clean/dictionary/
aggregate) without network access to download the real HF model.

It approximates RoBERTa's *contextual* behavior well enough to sanity-check
wiring: it looks at a small set of strong positive/negative anchor words
(with simple negation handling) to produce soft pos/neu/neg probabilities.
This is NOT what ships in the app — hr_pipeline/sentiment.py's
SentimentModel (real transformers pipeline) is what app.py uses.
"""
import re
from typing import Dict, List

POS_WORDS = ["good", "great", "excellent", "best", "friendly", "flexible", "growth",
             "supportive", "positive", "happy", "love", "amazing", "nice", "helpful",
             "balance", "opportunities", "career", "learning"]
NEG_WORDS = ["bad", "poor", "low", "toxic", "biased", "rigid", "slow", "politics",
             "stress", "pressure", "worst", "hate", "unfair", "long hours", "no growth",
             "lack", "issue", "problem", "concern"]


class MockSentimentModel:
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        out = []
        for t in texts:
            tl = t.lower()
            pos_hits = sum(1 for w in POS_WORDS if w in tl)
            neg_hits = sum(1 for w in NEG_WORDS if w in tl)
            total = pos_hits + neg_hits
            if total == 0:
                out.append({"pos": 0.2, "neu": 0.6, "neg": 0.2})
                continue
            pos_share = pos_hits / total
            neg_share = neg_hits / total
            # squash toward a plausible confidence range
            pos = 0.05 + 0.85 * pos_share
            neg = 0.05 + 0.85 * neg_share
            neu = max(0.0, 1 - pos - neg)
            out.append({"pos": pos, "neu": neu, "neg": neg})
        return out
