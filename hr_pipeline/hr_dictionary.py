"""
HR keyword/phrase dictionary matching.

Strictly a topic-tagger: it says WHICH HR category a review touches on,
using word-boundary matching against the dictionary's keyword/phrase/variant
list. It never overrides the contextual sentiment computed in sentiment.py —
the dictionary's own Sentiment/Sentiment_Strength columns are metadata only
and are not used to relabel a review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

NO_THEME = "No specific HR theme identified"
NO_KEYWORD = "No specific HR keyword identified"
NO_REASON = "No specific HR-related reason identified"

TEXT_FIELDS_FOR_MATCHING = ["pros", "cons", "review_title", "advice_to_mgmt", "review_text"]


@dataclass
class DictEntry:
    term: str
    category: str
    pattern: re.Pattern


def _compile_entries(dict_df: pd.DataFrame) -> List[DictEntry]:
    """Builds one compiled regex per keyword/phrase AND per variant.

    Longer terms are matched first at lookup time (by sorting), so e.g.
    "good work life balance" gets credit instead of only its substring
    "work life balance" — matching the granularity the reference analysis
    used (both were kept, but tests below prefer specificity when scoring).
    """
    entries: List[DictEntry] = []
    for _, row in dict_df.iterrows():
        category = str(row.get("hr_category", "")).strip()
        if not category:
            continue
        terms = set()
        main_term = row.get("keyword_or_phrase")
        if pd.notna(main_term) and str(main_term).strip():
            terms.add(str(main_term).strip().lower())
        variant = row.get("variants")
        if pd.notna(variant) and str(variant).strip():
            terms.add(str(variant).strip().lower())

        for term in terms:
            escaped = re.escape(term)
            # word-boundary aware even for multi-word phrases
            pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", flags=re.IGNORECASE)
            entries.append(DictEntry(term=term, category=category, pattern=pattern))

    # Sort longest-term-first so overlapping matches favor specificity
    entries.sort(key=lambda e: len(e.term), reverse=True)
    return entries


class HRDictionaryMatcher:
    def __init__(self, dict_df: pd.DataFrame):
        self.dict_df = dict_df
        self.entries = _compile_entries(dict_df)
        self.total_entries = len(dict_df)

    def _match_text(self, text: str) -> List[Tuple[str, str]]:
        """Returns list of (term, category) matches found in one text blob."""
        matches = []
        for entry in self.entries:
            if entry.pattern.search(text):
                matches.append((entry.term, entry.category))
        return matches

    def match_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns (df_with_hr_columns, category_long_table, keyword_long_table).

        category_long_table mirrors the reference "Review HR Mapping" sheet:
        one row per (review_id, HR_Category) with keyword count and the
        review's own sentiment carried over for later aggregation.

        keyword_long_table is one row per (review_id, HR_Category,
        Keyword_or_Phrase) — the finer granularity "Keyword Analysis" /
        "Positive Drivers" / "HR Risk Signals" need.
        """
        fields_present = [f for f in TEXT_FIELDS_FOR_MATCHING if f in df.columns]

        hr_categories_col, hr_keywords_col, hr_reason_col = [], [], []
        long_rows = []
        keyword_rows = []

        for _, row in df.iterrows():
            combined_text = " ".join(str(row[f]) for f in fields_present if pd.notna(row.get(f)))
            if not combined_text.strip():
                hr_categories_col.append(NO_THEME)
                hr_keywords_col.append(NO_KEYWORD)
                hr_reason_col.append(NO_REASON)
                continue

            matches = self._match_text(combined_text)
            if not matches:
                hr_categories_col.append(NO_THEME)
                hr_keywords_col.append(NO_KEYWORD)
                hr_reason_col.append(NO_REASON)
                continue

            # de-dup (term, category) pairs while preserving first-seen order
            seen = []
            for term, category in matches:
                if (term, category) not in seen:
                    seen.append((term, category))

            categories_in_order = []
            for _, category in seen:
                if category not in categories_in_order:
                    categories_in_order.append(category)

            hr_categories_col.append("; ".join(categories_in_order))
            hr_keywords_col.append("; ".join(term for term, _ in seen))
            hr_reason_col.append("; ".join(f"{term} ({category})" for term, category in seen))

            category_counts: Dict[str, int] = {}
            for term, category in seen:
                category_counts[category] = category_counts.get(category, 0) + 1

            for category, count in category_counts.items():
                long_rows.append({
                    "review_id": row["review_id"],
                    "HR_Category": category,
                    "Keyword_Count": count,
                    "sentiment_score": row.get("sentiment_score"),
                    "sentiment_magnitude": row.get("sentiment_magnitude"),
                    "final_sentiment": row.get("final_sentiment"),
                })

            for term, category in seen:
                keyword_rows.append({
                    "review_id": row["review_id"],
                    "HR_Category": category,
                    "Keyword_or_Phrase": term,
                    "sentiment_score": row.get("sentiment_score"),
                    "sentiment_magnitude": row.get("sentiment_magnitude"),
                    "final_sentiment": row.get("final_sentiment"),
                })

        out = df.copy()
        out["HR_Categories"] = hr_categories_col
        out["HR_Keywords"] = hr_keywords_col
        out["HR_Reason"] = hr_reason_col

        long_df = pd.DataFrame(long_rows, columns=[
            "review_id", "HR_Category", "Keyword_Count",
            "sentiment_score", "sentiment_magnitude", "final_sentiment",
        ])
        keyword_df = pd.DataFrame(keyword_rows, columns=[
            "review_id", "HR_Category", "Keyword_or_Phrase",
            "sentiment_score", "sentiment_magnitude", "final_sentiment",
        ])
        return out, long_df, keyword_df

    def coverage_report(self, df: pd.DataFrame) -> Dict[str, float]:
        total = len(df)
        with_theme = int((df["HR_Categories"] != NO_THEME).sum())
        without_theme = total - with_theme
        pct = round(100 * with_theme / total, 2) if total else 0.0
        return {
            "Total Reviews": total,
            "Reviews with HR Theme": with_theme,
            "Reviews without HR Theme": without_theme,
            "HR Dictionary Coverage (%)": pct,
            "Dictionary Entries": self.total_entries,
        }
