"""
Aggregations for the dashboard tabs. Every number here is computed live
from whatever data was uploaded — nothing is hard-coded to Bosch's
977/471/837/85.67% reference figures. Those only show up if you happen to
re-run the Bosch file.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

RISK_LOW_VOLUME_N = 10          # below this, a concerning keyword is "low-volume", not a "concern"
RISK_NEGATIVE_PCT_THRESHOLD = 15.0
RISK_MIXED_PCT_THRESHOLD = 30.0
RISK_HIGH_VOLUME_N = 20


def hr_category_summary(category_long: pd.DataFrame) -> pd.DataFrame:
    if category_long.empty:
        return pd.DataFrame(columns=[
            "HR_Category", "Unique_Reviews", "Average_Sentiment_Score", "Average_Magnitude",
            "Positive", "Negative", "Neutral", "Mixed", "Total_Category_Reviews",
            "Positive_%", "Negative_%", "Neutral_%", "Mixed_%",
        ])

    rows = []
    for category, grp in category_long.groupby("HR_Category"):
        grp = grp.drop_duplicates(subset="review_id")
        n = len(grp)
        counts = grp["final_sentiment"].value_counts()
        rows.append({
            "HR_Category": category,
            "Unique_Reviews": n,
            "Average_Sentiment_Score": round(grp["sentiment_score"].mean(), 3),
            "Average_Magnitude": round(grp["sentiment_magnitude"].mean(), 3),
            "Positive": int(counts.get("Positive", 0)),
            "Negative": int(counts.get("Negative", 0)),
            "Neutral": int(counts.get("Neutral", 0)),
            "Mixed": int(counts.get("Mixed", 0)),
            "Total_Category_Reviews": n,
            "Positive_%": round(100 * counts.get("Positive", 0) / n, 1) if n else 0,
            "Negative_%": round(100 * counts.get("Negative", 0) / n, 1) if n else 0,
            "Neutral_%": round(100 * counts.get("Neutral", 0) / n, 1) if n else 0,
            "Mixed_%": round(100 * counts.get("Mixed", 0) / n, 1) if n else 0,
        })
    out = pd.DataFrame(rows).sort_values("Unique_Reviews", ascending=False).reset_index(drop=True)
    return out


def keyword_analysis(keyword_long: pd.DataFrame) -> pd.DataFrame:
    if keyword_long.empty:
        return pd.DataFrame(columns=[
            "HR_Category", "Keyword_or_Phrase", "Mentions", "Unique_Reviews",
            "Average_Score", "Average_Magnitude", "Positive_Mentions", "Negative_Mentions",
            "Neutral_Mentions", "Mixed_Mentions", "Total_Sentiment_Mentions", "Net_Sentiment_%",
        ])

    rows = []
    for (category, keyword), grp in keyword_long.groupby(["HR_Category", "Keyword_or_Phrase"]):
        n_mentions = len(grp)
        n_unique = grp["review_id"].nunique()
        counts = grp["final_sentiment"].value_counts()
        pos = int(counts.get("Positive", 0))
        neg = int(counts.get("Negative", 0))
        neu = int(counts.get("Neutral", 0))
        mix = int(counts.get("Mixed", 0))
        total = pos + neg + neu + mix
        net = round(100 * (pos - neg) / total, 1) if total else 0.0
        rows.append({
            "HR_Category": category,
            "Keyword_or_Phrase": keyword,
            "Mentions": n_mentions,
            "Unique_Reviews": n_unique,
            "Average_Score": round(grp["sentiment_score"].mean(), 3),
            "Average_Magnitude": round(grp["sentiment_magnitude"].mean(), 3),
            "Positive_Mentions": pos,
            "Negative_Mentions": neg,
            "Neutral_Mentions": neu,
            "Mixed_Mentions": mix,
            "Total_Sentiment_Mentions": total,
            "Net_Sentiment_%": net,
        })
    return pd.DataFrame(rows).sort_values("Mentions", ascending=False).reset_index(drop=True)


def positive_drivers(kw_analysis: pd.DataFrame, min_mentions: int = 5) -> pd.DataFrame:
    if kw_analysis.empty:
        return kw_analysis
    drivers = kw_analysis[(kw_analysis["Net_Sentiment_%"] > 0) & (kw_analysis["Unique_Reviews"] >= min_mentions)]
    drivers = drivers.sort_values(["Net_Sentiment_%", "Unique_Reviews"], ascending=[False, False])
    return drivers[[
        "HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Average_Score",
        "Average_Magnitude", "Positive_Mentions", "Negative_Mentions", "Net_Sentiment_%",
    ]].reset_index(drop=True)


def hr_risk_signals(kw_analysis: pd.DataFrame) -> pd.DataFrame:
    """Flags concerning keywords, but never over-claims: a keyword mentioned
    by a handful of people is a 'Low-volume signal', not a stated org-wide
    problem, regardless of how bad its negative/mixed percentage looks."""
    if kw_analysis.empty:
        return pd.DataFrame(columns=[
            "HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Average_Score",
            "Average_Magnitude", "Negative_%", "Mixed_%", "Signal_Type",
        ])

    df = kw_analysis.copy()
    df["Negative_%"] = np.where(df["Total_Sentiment_Mentions"] > 0,
                                 round(100 * df["Negative_Mentions"] / df["Total_Sentiment_Mentions"], 1), 0.0)
    df["Mixed_%"] = np.where(df["Total_Sentiment_Mentions"] > 0,
                              round(100 * df["Mixed_Mentions"] / df["Total_Sentiment_Mentions"], 1), 0.0)

    flagged = df[(df["Negative_%"] >= RISK_NEGATIVE_PCT_THRESHOLD) |
                 (df["Mixed_%"] >= RISK_MIXED_PCT_THRESHOLD)].copy()

    def signal_type(row):
        if row["Unique_Reviews"] < RISK_LOW_VOLUME_N:
            return "Low-volume signal"
        if row["Unique_Reviews"] >= RISK_HIGH_VOLUME_N:
            return "High-volume concern"
        return "Emerging concern"

    if flagged.empty:
        return flagged
    flagged["Signal_Type"] = flagged.apply(signal_type, axis=1)
    flagged = flagged.sort_values(["Unique_Reviews", "Negative_%"], ascending=[False, False])
    return flagged[[
        "HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Average_Score",
        "Average_Magnitude", "Negative_%", "Mixed_%", "Signal_Type",
    ]].reset_index(drop=True)


def dictionary_coverage(coverage: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(list(coverage.items()), columns=["Metric", "Value"])


def methodology_table(model_name: str, n_rows: int, n_dict_entries: int, coverage_pct: float,
                       n_categories: int, n_category_reviews: int) -> pd.DataFrame:
    rows = [
        ("Dataset", "Uploaded employee review export (company-agnostic pipeline)"),
        ("Number of Reviews", n_rows),
        ("Sentiment Model", f"{model_name} — contextual RoBERTa sentiment model"),
        ("Sentiment Classes", "Positive, Negative, Neutral, Mixed"),
        ("Sentiment Score", "Positive probability - Negative probability, from -1 to +1"),
        ("Sentiment Magnitude", "Positive probability + Negative probability, from 0 to 1"),
        ("Overall Rating Used For Sentiment?", "No — rating is shown for reference only, never used to derive sentiment"),
        ("HR Dictionary", "User-supplied (or bundled default) HR keyword/phrase dictionary"),
        ("Dictionary Entries", n_dict_entries),
        ("HR Dictionary Coverage", f"{coverage_pct}%"),
        ("HR Categories Detected", n_categories),
        ("Review x HR Category Observations", n_category_reviews),
        ("Employee-level Reason", "Generated from the review's own Pros/Cons/Title text and its contextual sentiment evidence"),
    ]
    rows = [(component, str(description)) for component, description in rows]
    return pd.DataFrame(rows, columns=["Component", "Description"])
