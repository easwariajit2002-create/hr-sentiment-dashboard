"""
Benchmark the live pipeline's output against a reference employee-level
sentiment analysis (e.g. an existing Colab/notebook export).

This exists specifically so thresholds are only ever changed on evidence:
run this against Bosch_Final_HR_Sentiment_Analysis_FINAL.xlsx (or any
similarly-shaped reference) after deploying with real model access, look at
the confusion matrix and the sampled mismatches, and only then decide
whether a genuine methodological gap exists.

Generic by design — works for any company's reference file, not just Bosch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

SENTIMENT_COL_ALIASES = ["final_sentiment", "sentiment", "sentiment_label", "final sentiment"]
ID_COL_ALIASES = ["review_id", "reviewid", "id", "review#"]


def _find_col(columns, aliases) -> Optional[str]:
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in columns}
    for orig, n in norm.items():
        if n in aliases:
            return orig
    return None


def load_reference_sentiment(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Reads a reference file (xlsx or csv). For multi-sheet Excel exports
    (like the Bosch FINAL.xlsx workbook), scans sheets for one that has a
    recognizable per-review sentiment label column and uses that one."""
    import io
    buf = io.BytesIO(file_bytes)

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(buf)
        sentiment_col = _find_col(df.columns, SENTIMENT_COL_ALIASES)
        if not sentiment_col:
            raise ValueError("No recognizable sentiment column found in the reference CSV.")
        return df.rename(columns={sentiment_col: "reference_sentiment"})

    xl = pd.ExcelFile(buf)
    best_sheet, best_df, best_col = None, None, None
    for sheet in xl.sheet_names:
        sheet_df = xl.parse(sheet)
        col = _find_col(sheet_df.columns, SENTIMENT_COL_ALIASES)
        if col and (best_df is None or len(sheet_df) > len(best_df)):
            best_sheet, best_df, best_col = sheet, sheet_df, col
    if best_df is None:
        raise ValueError(
            "No sheet with a recognizable sentiment column was found in this workbook. "
            f"Sheets checked: {xl.sheet_names}"
        )
    return best_df.rename(columns={best_col: "reference_sentiment"})


def align(reviews: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Aligns our computed reviews with the reference by review_id if a
    shared id column exists, else falls back to row position (valid only
    when both files preserve the same original row order, as Bosch's
    Employee Analysis sheet does relative to bosch_final_cleaned.csv)."""
    ref_id_col = _find_col(reference.columns, ID_COL_ALIASES)
    if ref_id_col and "review_id" in reviews.columns:
        merged = reviews[["review_id", "final_sentiment"]].merge(
            reference[[ref_id_col, "reference_sentiment"]].rename(columns={ref_id_col: "review_id"}),
            on="review_id", how="inner",
        )
        if len(merged) >= min(len(reviews), len(reference)) * 0.5:
            return merged

    # Positional fallback
    n = min(len(reviews), len(reference))
    merged = pd.DataFrame({
        "review_id": reviews["review_id"].iloc[:n].values,
        "final_sentiment": reviews["final_sentiment"].iloc[:n].values,
        "reference_sentiment": reference["reference_sentiment"].iloc[:n].values,
    })
    return merged


def compare(reviews: pd.DataFrame, reference: pd.DataFrame) -> Dict:
    merged = align(reviews, reference)
    merged["reference_sentiment"] = merged["reference_sentiment"].astype(str).str.strip().str.title()
    merged["final_sentiment"] = merged["final_sentiment"].astype(str).str.strip().str.title()

    agreement = (merged["final_sentiment"] == merged["reference_sentiment"]).mean()
    confusion = pd.crosstab(merged["reference_sentiment"], merged["final_sentiment"],
                             rownames=["Reference"], colnames=["App"])
    mismatches = merged[merged["final_sentiment"] != merged["reference_sentiment"]]

    return {
        "n_compared": len(merged),
        "agreement_rate": round(float(agreement), 4),
        "confusion_matrix": confusion,
        "mismatches": mismatches,
    }
