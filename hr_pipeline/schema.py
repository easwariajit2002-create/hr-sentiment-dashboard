"""
Column schema detection & validation.

The whole point of this module is reusability: it never assumes the
uploaded file is Bosch. It looks at the column names the user actually
uploaded, matches them against known aliases for each canonical field,
and reports back what it found, what's missing, and what got disabled.

Canonical fields used everywhere downstream:
    review_id, date, employee_title, location, employee_status,
    employment_type, tenure, gender, review_title, rating_overall,
    pros, cons, advice_to_mgmt, review_text
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

# --- Canonical field definitions -------------------------------------------------

# At least ONE of these text fields must exist, or there is nothing to analyze.
TEXT_FIELDS_REQUIRE_ONE = ["pros", "cons", "review_text", "review_title"]

# Fields that are nice-to-have; if missing we just disable the related
# filter/analysis rather than failing.
OPTIONAL_FIELDS = [
    "date", "employee_title", "location", "employee_status",
    "employment_type", "tenure", "gender", "review_title",
    "rating_overall", "advice_to_mgmt", "review_id",
]

ALL_CANONICAL_FIELDS = list(dict.fromkeys(TEXT_FIELDS_REQUIRE_ONE + OPTIONAL_FIELDS + ["pros", "cons"]))

# Aliases are matched against a normalized column name (lowercase, spaces/
# underscores/dashes stripped). Order doesn't matter within a list.
ALIASES: Dict[str, List[str]] = {
    "review_id": ["reviewid", "id", "review#", "recordid"],
    "date": ["date", "reviewdate", "createdat", "postdate", "submissiondate"],
    "employee_title": ["employeetitle", "jobtitle", "title", "role", "position", "designation"],
    "location": ["location", "city", "office", "worklocation", "site"],
    "employee_status": ["employeestatus", "status", "currentformer"],
    "employment_type": ["employmenttype", "emptype", "currentemployee"],
    "tenure": ["tenure", "yearsofservice", "yearsatcompany", "experience", "lengthofservice"],
    "gender": ["gender", "sex"],
    "review_title": ["reviewtitle", "headline", "summary", "title"],
    "rating_overall": ["ratingoverall", "rating", "overallrating", "starrating", "score"],
    "pros": ["pros", "positives", "whatyouliked", "prosreview", "likes"],
    "cons": ["cons", "negatives", "whatyoudislike", "consreview", "dislikes", "improvementareas"],
    "advice_to_mgmt": ["advicetomgmt", "advicetomanagement", "advice", "suggestions"],
    "review_text": ["reviewtext", "review", "fulltext", "comments", "feedback", "body"],
}

# Dictionary file column aliases (HR keyword/sentiment dictionary)
DICT_ALIASES: Dict[str, List[str]] = {
    "keyword_or_phrase": ["keywordorphrase", "keyword", "phrase", "term"],
    "hr_category": ["hrcategory", "category", "theme"],
    "sentiment": ["sentiment", "polarity"],
    "sentiment_strength": ["sentimentstrength", "strength", "weight"],
    "keyword_type": ["keywordtype", "type"],
    "variants": ["variants", "variant", "synonyms", "aliases"],
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _best_match(normalized_col: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    for canonical, aliases in alias_map.items():
        candidates = [canonical] + aliases
        if normalized_col in {_normalize(c) for c in candidates}:
            return canonical
    return None


def _fuzzy_match(normalized_columns: Dict[str, str], alias_map: Dict[str, List[str]],
                  used: set) -> Dict[str, str]:
    """Fallback fuzzy pass for columns that didn't exact-match any alias."""
    mapping = {}
    all_alias_terms = []
    term_to_canonical = {}
    for canonical, aliases in alias_map.items():
        for a in [canonical] + aliases:
            na = _normalize(a)
            all_alias_terms.append(na)
            term_to_canonical[na] = canonical

    for orig_col, norm_col in normalized_columns.items():
        if orig_col in used:
            continue
        close = difflib.get_close_matches(norm_col, all_alias_terms, n=1, cutoff=0.82)
        if close:
            canonical = term_to_canonical[close[0]]
            if canonical not in mapping.values():
                mapping[orig_col] = canonical
    return mapping


@dataclass
class SchemaResult:
    column_mapping: Dict[str, str]  # original_column -> canonical_field
    reverse_mapping: Dict[str, str] = field(default_factory=dict)  # canonical_field -> original_column
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    unmapped_columns: List[str] = field(default_factory=list)
    ok: bool = True
    messages: List[str] = field(default_factory=list)

    def has(self, canonical_field: str) -> bool:
        return canonical_field in self.reverse_mapping


def detect_schema(df: pd.DataFrame) -> SchemaResult:
    normalized_columns = {c: _normalize(c) for c in df.columns}
    mapping: Dict[str, str] = {}
    used_canonical = set()

    # Exact/alias pass
    for orig_col, norm_col in normalized_columns.items():
        canonical = _best_match(norm_col, ALIASES)
        if canonical and canonical not in used_canonical:
            mapping[orig_col] = canonical
            used_canonical.add(canonical)

    # Fuzzy pass for leftovers
    remaining_cols = {c: n for c, n in normalized_columns.items() if c not in mapping}
    fuzzy = _fuzzy_match(remaining_cols, ALIASES, used=set(mapping.keys()))
    for orig_col, canonical in fuzzy.items():
        if canonical not in used_canonical:
            mapping[orig_col] = canonical
            used_canonical.add(canonical)

    reverse_mapping = {v: k for k, v in mapping.items()}
    unmapped = [c for c in df.columns if c not in mapping]

    result = SchemaResult(column_mapping=mapping, reverse_mapping=reverse_mapping,
                           unmapped_columns=unmapped)

    # At least one text field required
    has_text = any(result.has(f) for f in TEXT_FIELDS_REQUIRE_ONE)
    if not has_text:
        result.ok = False
        result.missing_required.append(
            "No review text column found (need at least one of: pros, cons, review_title, review_text)."
        )

    for f in OPTIONAL_FIELDS:
        if not result.has(f):
            result.missing_optional.append(f)

    if result.missing_optional:
        result.messages.append(
            "The following optional fields were not found and their filters/analyses will be "
            "disabled: " + ", ".join(result.missing_optional)
        )
    return result


def detect_dictionary_schema(df: pd.DataFrame) -> SchemaResult:
    normalized_columns = {c: _normalize(c) for c in df.columns}
    mapping: Dict[str, str] = {}
    used_canonical = set()
    for orig_col, norm_col in normalized_columns.items():
        canonical = _best_match(norm_col, DICT_ALIASES)
        if canonical and canonical not in used_canonical:
            mapping[orig_col] = canonical
            used_canonical.add(canonical)

    reverse_mapping = {v: k for k, v in mapping.items()}
    result = SchemaResult(column_mapping=mapping, reverse_mapping=reverse_mapping,
                           unmapped_columns=[c for c in df.columns if c not in mapping])

    required = ["keyword_or_phrase", "hr_category"]
    for r in required:
        if not result.has(r):
            result.ok = False
            result.missing_required.append(r)
    return result


def apply_mapping(df: pd.DataFrame, schema: SchemaResult) -> pd.DataFrame:
    """Return a copy of df renamed to canonical field names (unmapped columns kept as-is)."""
    return df.rename(columns=schema.column_mapping)
