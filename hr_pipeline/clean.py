"""
Data cleaning.

Everything here is generic — it operates on whatever canonical columns
happen to be present after schema mapping, and never assumes a specific
company's data shape.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

# Tokens that mean "this field is genuinely empty", regardless of company.
NULL_TOKENS = {
    "not provided", "n/a", "na", "none", "nil", "no comment", "no comments",
    "-", ".", "null", "unknown", "not applicable", "not disclosed", "",
}

TEXT_CANDIDATE_COLS = ["pros", "cons", "advice_to_mgmt", "review_title", "review_text"]


def _normalize_null(value) -> object:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.lower() in NULL_TOKENS:
        return None
    return s


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean a schema-mapped dataframe. Returns (clean_df, report)."""
    df = df.copy()
    report = {"rows_in": len(df), "duplicates_removed": 0, "missing_field_counts": {}}

    # Normalize obvious "empty" tokens to real nulls in text columns.
    # Cast to plain object dtype first: pandas' newer StringDtype silently
    # turns a mapped `None` back into a float NaN, which then slips past
    # truthiness checks downstream (NaN is truthy in Python).
    for col in TEXT_CANDIDATE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object).map(_normalize_null)

    # Track missingness before dropping anything (needed for the Methodology tab)
    for col in TEXT_CANDIDATE_COLS + ["rating_overall", "date", "employee_title", "location",
                                       "employee_status", "employment_type", "tenure", "gender"]:
        if col in df.columns:
            report["missing_field_counts"][col] = int(df[col].isna().sum())

    # Parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Ensure a stable review_id exists (drill-down and joins rely on it)
    if "review_id" not in df.columns:
        df.insert(0, "review_id", range(len(df)))
    else:
        # coerce to a clean int/str id, fill any missing with a running index
        df["review_id"] = df["review_id"].where(df["review_id"].notna(), pd.Series(range(len(df))))

    # Duplicate detection: same pros+cons+title+date+employee_title looks like a true duplicate
    dedup_cols = [c for c in ["pros", "cons", "review_title", "date", "employee_title"] if c in df.columns]
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        report["duplicates_removed"] = before - len(df)

    # Rows with genuinely no text anywhere are useless for sentiment — flag, don't silently vanish
    text_cols_present = [c for c in ["pros", "cons", "review_title", "review_text"] if c in df.columns]
    if text_cols_present:
        has_any_text = df[text_cols_present].notna().any(axis=1)
        report["rows_without_any_text"] = int((~has_any_text).sum())
        df["_has_text"] = has_any_text
    else:
        df["_has_text"] = False

    # Clean rating to numeric if present
    if "rating_overall" in df.columns:
        df["rating_overall"] = pd.to_numeric(df["rating_overall"], errors="coerce")

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report
