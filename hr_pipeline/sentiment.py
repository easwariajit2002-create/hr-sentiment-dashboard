"""
Contextual sentiment analysis.

Design goals (per spec):
  * Sentiment Score = Positive probability - Negative probability, range -1..+1
  * Sentiment Magnitude = Positive probability + Negative probability, range 0..1
  * The star/overall rating must NEVER be used to determine sentiment.
  * Final label (Positive / Negative / Neutral / Mixed) is derived from the
    *contextual* model's per-field probabilities, not from a keyword count.
  * Mixed requires genuinely separate positive evidence (usually in `pros`)
    AND genuinely separate negative evidence (usually in `cons`) — a single
    ambivalent sentence is not enough, and negated-negative boilerplate like
    "no complaints" / "nothing negative" must never register as negative
    evidence.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest — a RoBERTa model
fine-tuned for contextual (not lexicon-based) 3-class sentiment
(negative / neutral / positive). This matches the "RoBERTa contextual
sentiment model" methodology used for the Bosch reference analysis.

The model is loaded once per process (Streamlit's @st.cache_resource wraps
`load_model` in app.py) and run in batches for speed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Boilerplate that LOOKS like it mentions something negative ("complaint",
# "cons", "bad") but is actually the employee saying there ISN'T anything
# negative. These must never count as negative evidence.
NEGATION_OF_NEGATIVE = [
    r"\bno\s+complaints?\b",
    r"\bnothing\s+(much\s+)?(to\s+)?(say\s+)?(that'?s?\s+)?negative\b",
    r"\bnothing\s+negative\b",
    r"\bnothing\s+much\s+to\s+say\b",
    r"\bnothing\s+to\s+(dislike|complain|hate)\b",
    r"\bno\s+cons?\b",
    r"\bcan'?t\s+think\s+of\s+any\b",
    r"\bnothing\s+comes?\s+to\s+mind\b",
    r"\bnothing\s+i\s+can\s+think\s+of\b",
    r"\bnone\s+so\s+far\b",
    r"\bnone\s+at\s+all\b",
    r"\ball\s+good\b",
    r"\beverything\s+is\s+fine\b",
    r"\bnot\s+much\b\s*$",
    r"^\s*(none|nothing|na|no)\s*$",
]
_NEGATION_RE = re.compile("|".join(NEGATION_OF_NEGATIVE), flags=re.IGNORECASE)

# Same idea in reverse, in case "pros" ever contains "nothing good" style text.
NEGATION_OF_POSITIVE = [
    r"\bnothing\s+(much\s+)?(to\s+)?(say\s+)?(that'?s?\s+)?(good|positive)\b",
    r"\bnone\s+that\s+i\s+can\s+see\b",
    r"^\s*(none|nothing|na|no)\s*$",
]
_NEGATION_POS_RE = re.compile("|".join(NEGATION_OF_POSITIVE), flags=re.IGNORECASE)

# Meaningfulness thresholds — a source only "counts" as positive/negative
# evidence if its dominant-class probability clears this bar.
MEANINGFUL_THRESHOLD = 0.40
NEUTRAL_MAGNITUDE_FLOOR = 0.15
NEUTRAL_SCORE_BAND = 0.15


@dataclass
class FieldSignal:
    text: Optional[str]
    pos: float
    neu: float
    neg: float
    negated: bool  # True if boilerplate negation stripped its polarity

    @property
    def present(self) -> bool:
        return bool(self.text)

    @property
    def dominant(self) -> str:
        if self.negated:
            return "neu"
        scores = {"pos": self.pos, "neu": self.neu, "neg": self.neg}
        return max(scores, key=scores.get)

    def is_meaningful(self, polarity: str) -> bool:
        if self.negated or not self.present:
            return False
        value = {"pos": self.pos, "neu": self.neu, "neg": self.neg}[polarity]
        return self.dominant == polarity and value >= MEANINGFUL_THRESHOLD


class SentimentModel:
    """Thin wrapper around a HF transformers text-classification pipeline.

    Kept separate from Streamlit so it's testable and cacheable independently.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            max_length=256,
            batch_size=batch_size,
        )
        # cardiffnlp label ids -> names differ by version; read from config to be safe
        self._id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Returns a list of {'pos':..,'neu':..,'neg':..} dicts, same order as input."""
        if not texts:
            return []
        raw = self._pipe(texts)
        out = []
        for scores in raw:
            probs = {"pos": 0.0, "neu": 0.0, "neg": 0.0}
            for item in scores:
                label = item["label"].lower()
                if "pos" in label:
                    probs["pos"] = item["score"]
                elif "neg" in label:
                    probs["neg"] = item["score"]
                else:
                    probs["neu"] = item["score"]
            out.append(probs)
        return out


def _clean_text(t) -> Optional[str]:
    """Normalize any incoming value to either a non-empty str or None.
    Guards against stray float NaNs (pandas string-dtype quirk) slipping
    through, since `bool(float('nan'))` is True and would otherwise pass
    truthiness checks downstream."""
    if t is None:
        return None
    if isinstance(t, float):
        return None  # NaN or any stray float means "missing"
    s = str(t).strip()
    return s if s else None


def _dedupe_texts(texts: List) -> Tuple[List[str], Dict[str, int]]:
    """Build the unique-text list to score once, even if the same phrase
    ("Not Provided", "Good work culture", etc.) repeats across thousands of rows."""
    unique = []
    index = {}
    for raw in texts:
        t = _clean_text(raw)
        if t and t not in index:
            index[t] = len(unique)
            unique.append(t)
    return unique, index


def score_field(model: SentimentModel, series: pd.Series, negation_re: re.Pattern) -> List[FieldSignal]:
    texts = series.tolist()
    unique_texts, lookup = _dedupe_texts(texts)
    predictions = model.predict_batch(unique_texts) if unique_texts else []

    signals = []
    for raw in texts:
        t = _clean_text(raw)
        if not t:
            signals.append(FieldSignal(text=None, pos=0.0, neu=1.0, neg=0.0, negated=False))
            continue
        probs = predictions[lookup[t]]
        negated = bool(negation_re.search(t))
        signals.append(FieldSignal(text=t, pos=probs["pos"], neu=probs["neu"], neg=probs["neg"], negated=negated))
    return signals


def combine_and_classify(pros_signals: List[FieldSignal],
                          cons_signals: List[FieldSignal],
                          title_signals: List[FieldSignal]) -> pd.DataFrame:
    n = len(pros_signals)
    rows = []
    for i in range(n):
        p, c, t = pros_signals[i], cons_signals[i], title_signals[i]

        sources = [(p, 1.0), (c, 1.0), (t, 0.3)]
        present_sources = [(s, w) for s, w in sources if s.present]

        if present_sources:
            total_w = sum(w for _, w in present_sources)
            pos_prob = sum((0.0 if s.negated else s.pos) * w for s, w in present_sources) / total_w
            neg_prob = sum((0.0 if s.negated else s.neg) * w for s, w in present_sources) / total_w
            neu_prob = max(0.0, 1.0 - pos_prob - neg_prob)
        else:
            pos_prob, neg_prob, neu_prob = 0.0, 1.0, 0.0

        score = pos_prob - neg_prob
        magnitude = pos_prob + neg_prob

        pros_positive = p.is_meaningful("pos")
        cons_negative = c.is_meaningful("neg")
        # a title can reinforce but is never enough on its own to flip the label
        title_positive = t.is_meaningful("pos")
        title_negative = t.is_meaningful("neg")

        reason_parts = []
        if pros_positive:
            reason_parts.append(f"Positive reason: {p.text}")
        elif title_positive and not pros_positive:
            reason_parts.append(f"Positive reason (title): {t.text}")
        if cons_negative:
            reason_parts.append(f"Negative reason: {c.text}")
        elif title_negative and not cons_negative:
            reason_parts.append(f"Negative reason (title): {t.text}")

        if pros_positive and cons_negative:
            label = "Mixed"
        elif pros_positive and not cons_negative:
            label = "Positive"
        elif cons_negative and not pros_positive:
            label = "Negative"
        else:
            # neither field cleared the meaningfulness bar — fall back to the
            # blended score/magnitude so we still classify obviously-neutral
            # or single-signal-but-below-threshold reviews sensibly.
            if magnitude < NEUTRAL_MAGNITUDE_FLOOR or abs(score) < NEUTRAL_SCORE_BAND:
                label = "Neutral"
            elif score > 0:
                label = "Positive"
            else:
                label = "Negative"

        if not reason_parts:
            if p.present:
                reason_parts.append(f"Review evidence: {p.text}")
            elif c.present:
                reason_parts.append(f"Review evidence: {c.text}")
            elif t.present:
                reason_parts.append(f"Review title/evidence: {t.text}")
            else:
                reason_parts.append("No review text available for sentiment evidence.")

        rows.append({
            "sentiment_score": round(float(score), 6),
            "sentiment_magnitude": round(float(magnitude), 6),
            "final_sentiment": label,
            "sentiment_reason": " | ".join(reason_parts),
        })
    return pd.DataFrame(rows)


def analyze_sentiment(df: pd.DataFrame, model: SentimentModel) -> pd.DataFrame:
    """Adds sentiment_score, sentiment_magnitude, final_sentiment, sentiment_reason.

    Works with whatever of pros/cons/review_title/review_text are present.
    The overall star rating is never read here — by construction.
    """
    n = len(df)
    empty_series = pd.Series([None] * n)

    pros_series = df["pros"] if "pros" in df.columns else (
        df["review_text"] if "review_text" in df.columns else empty_series
    )
    cons_series = df["cons"] if "cons" in df.columns else empty_series
    title_series = df["review_title"] if "review_title" in df.columns else empty_series

    pros_signals = score_field(model, pros_series, _NEGATION_POS_RE)
    cons_signals = score_field(model, cons_series, _NEGATION_RE)
    title_signals = score_field(model, title_series, _NEGATION_RE)

    result = combine_and_classify(pros_signals, cons_signals, title_signals)
    out = df.copy()
    for col in result.columns:
        out[col] = result[col].values
    return out
