"""
HR Employee Sentiment Intelligence — reusable dashboard.

Deploy on Streamlit Community Cloud (or run locally with
`streamlit run app.py`). Upload any company's review export (Bosch today,
Dell tomorrow, anything with a broadly similar column shape) plus an HR
keyword/sentiment dictionary (a sensible default is bundled), and get:
  Executive Overview | Workforce Analysis | HR Themes | Positive Drivers |
  HR Risk / Mixed Signals | Employee Reviews | Methodology / Data Quality

Nothing here is Bosch-specific: every column reference goes through the
schema-detection layer in hr_pipeline/schema.py.
"""
import io
import os

# Keep the downloaded model weights in a stable, explicit location for the
# life of this container. This doesn't survive a Streamlit Community Cloud
# redeploy (the filesystem is rebuilt), but it does mean the ~500MB download
# happens once per running instance, not once per user session — the
# combination of this + @st.cache_resource below is what actually matters:
# cache_resource keeps the loaded model in memory for every visitor after
# the first, for as long as this instance stays up.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))

import pandas as pd
import streamlit as st

from hr_pipeline.pipeline import run_pipeline
from hr_pipeline.sentiment import MODEL_NAME, SentimentModel
from hr_pipeline import benchmark

st.set_page_config(page_title="HR Employee Sentiment Intelligence", layout="wide")

DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), "assets", "default_hr_dictionary.xlsx")


# --------------------------------------------------------------------------------
# Cached, expensive resources
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the contextual sentiment model (first run only)...")
def load_model() -> SentimentModel:
    return SentimentModel(MODEL_NAME)


@st.cache_data(show_spinner=False)
def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)


@st.cache_data(show_spinner="Running the sentiment + HR theme pipeline (first run per file)...")
def cached_pipeline(review_bytes: bytes, review_name: str, dict_bytes: bytes, dict_name: str):
    review_df = read_table(review_bytes, review_name)
    dict_df = read_table(dict_bytes, dict_name)
    model = load_model()
    result = run_pipeline(review_df, dict_df, model, model_name=MODEL_NAME)
    return result


# --------------------------------------------------------------------------------
# Sidebar: uploads
# --------------------------------------------------------------------------------
st.sidebar.title("📥 Data")
review_file = st.sidebar.file_uploader("Employee review export (CSV or Excel)", type=["csv", "xlsx", "xls"])
dict_file = st.sidebar.file_uploader(
    "HR keyword/sentiment dictionary (optional — bundled default used otherwise)",
    type=["xlsx", "xls", "csv"],
)

st.title("🧭 HR Employee Sentiment Intelligence")
st.caption(
    "Upload any company's employee-review export. The pipeline validates columns, cleans the data, "
    "runs contextual sentiment analysis, tags HR themes, and builds the dashboard below — automatically."
)

if not review_file:
    st.info("⬅️ Upload an employee review export to get started (CSV or Excel).")
    st.stop()

review_bytes = review_file.getvalue()
if dict_file:
    dict_bytes, dict_name = dict_file.getvalue(), dict_file.name
else:
    with open(DEFAULT_DICT_PATH, "rb") as f:
        dict_bytes = f.read()
    dict_name = "default_hr_dictionary.xlsx"
    st.sidebar.caption("Using the bundled default HR dictionary.")

try:
    result = cached_pipeline(review_bytes, review_file.name, dict_bytes, dict_name)
except ValueError as e:
    st.error(f"❌ Could not process this file: {e}")
    st.stop()

reviews = result.reviews

if result.data_schema.missing_optional:
    st.warning(
        "Some optional fields weren't found in this file, so the related filters/analyses are disabled: "
        + ", ".join(result.data_schema.missing_optional)
    )

# --------------------------------------------------------------------------------
# Sidebar: cross-filters (apply everywhere)
# --------------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")


def multiselect_if_available(label, col):
    if col in reviews.columns and reviews[col].notna().any():
        options = sorted(reviews[col].dropna().unique().tolist())
        return st.sidebar.multiselect(label, options)
    st.sidebar.caption(f"{label}: not available in this dataset")
    return []


loc_sel = multiselect_if_available("Location", "location")
title_sel = multiselect_if_available("Employee Title", "employee_title")
status_sel = multiselect_if_available("Employee Status", "employee_status")
emp_type_sel = multiselect_if_available("Employment Type", "employment_type")
tenure_sel = multiselect_if_available("Tenure", "tenure")
gender_sel = multiselect_if_available("Gender", "gender")

date_range = None
if "date" in reviews.columns and reviews["date"].notna().any():
    min_d, max_d = reviews["date"].min(), reviews["date"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))

category_options = sorted(result.category_summary["HR_Category"].tolist()) if not result.category_summary.empty else []
category_sel = st.sidebar.multiselect("HR Category", category_options)

sentiment_sel = st.sidebar.multiselect("Sentiment", ["Positive", "Negative", "Neutral", "Mixed"])


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if loc_sel:
        out = out[out["location"].isin(loc_sel)]
    if title_sel:
        out = out[out["employee_title"].isin(title_sel)]
    if status_sel:
        out = out[out["employee_status"].isin(status_sel)]
    if emp_type_sel:
        out = out[out["employment_type"].isin(emp_type_sel)]
    if tenure_sel:
        out = out[out["tenure"].isin(tenure_sel)]
    if gender_sel:
        out = out[out["gender"].isin(gender_sel)]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["date"].isna()) | ((out["date"] >= start) & (out["date"] <= end))]
    if category_sel:
        out = out[out["HR_Categories"].apply(lambda cats: any(c in str(cats) for c in category_sel))]
    if sentiment_sel:
        out = out[out["final_sentiment"].isin(sentiment_sel)]
    return out


filtered = apply_filters(reviews)
filtered_ids = set(filtered["review_id"])
filtered_category_long = result.category_long[result.category_long["review_id"].isin(filtered_ids)]
filtered_keyword_long = result.keyword_long[result.keyword_long["review_id"].isin(filtered_ids)]

if filtered.empty:
    st.warning("No reviews match the current filter combination.")
    st.stop()

# recompute category/keyword-level tables against the filtered slice so every
# tab and every chart honors all active filters together
from hr_pipeline import aggregate  # noqa: E402  (after st.stop() guards above)

f_cat_summary = aggregate.hr_category_summary(filtered_category_long)
f_kw_analysis = aggregate.keyword_analysis(filtered_keyword_long)
f_pos_drivers = aggregate.positive_drivers(f_kw_analysis)
f_risk = aggregate.hr_risk_signals(f_kw_analysis)

tabs = st.tabs([
    "Executive Overview", "Workforce Analysis", "HR Themes", "Positive Drivers",
    "HR Risk / Mixed Signals", "Employee Reviews", "Methodology / Data Quality",
])

# --------------------------------------------------------------------------------
# 1. Executive Overview
# --------------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(filtered):,}")
    c2.metric("Avg. Rating", f"{filtered['rating_overall'].mean():.2f}" if "rating_overall" in filtered.columns else "N/A")
    c3.metric("Avg. Sentiment Score", f"{filtered['sentiment_score'].mean():+.2f}")
    c4.metric("Avg. Magnitude", f"{filtered['sentiment_magnitude'].mean():.2f}")

    counts = filtered["final_sentiment"].value_counts()
    pct = (100 * counts / len(filtered)).round(1)
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Positive %", f"{pct.get('Positive', 0)}%")
    c6.metric("Negative %", f"{pct.get('Negative', 0)}%")
    c7.metric("Neutral %", f"{pct.get('Neutral', 0)}%")
    c8.metric("Mixed %", f"{pct.get('Mixed', 0)}%")

    st.subheader("Top HR Themes")
    if not f_cat_summary.empty:
        st.dataframe(f_cat_summary.head(10)[[
            "HR_Category", "Unique_Reviews", "Average_Sentiment_Score", "Positive_%", "Negative_%", "Mixed_%",
        ]], use_container_width=True, hide_index=True)
    else:
        st.caption("No HR themes detected in the current filter selection.")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top Positive Drivers")
        if not f_pos_drivers.empty:
            st.dataframe(f_pos_drivers.head(8)[["HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Net_Sentiment_%"]],
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Not enough data for positive drivers yet.")
    with colB:
        st.subheader("Risk / Mixed Signals")
        if not f_risk.empty:
            st.dataframe(f_risk.head(8)[["HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Negative_%", "Mixed_%", "Signal_Type"]],
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No flagged risk signals in the current filter selection.")

# --------------------------------------------------------------------------------
# 2. Workforce Analysis
# --------------------------------------------------------------------------------
with tabs[1]:
    st.caption("All filters in the sidebar apply here (and everywhere else in this dashboard).")
    dims = [d for d in ["location", "employee_title", "employee_status", "employment_type", "tenure", "gender"]
            if d in filtered.columns]
    if not dims:
        st.info("No workforce dimension columns were found in this dataset.")
    else:
        dim = st.selectbox("Break down by", dims, format_func=lambda x: x.replace("_", " ").title())
        breakdown = (filtered.groupby(dim)
                     .agg(Reviews=("review_id", "count"),
                          Avg_Rating=("rating_overall", "mean") if "rating_overall" in filtered.columns else ("review_id", "count"),
                          Avg_Sentiment_Score=("sentiment_score", "mean"),
                          Avg_Magnitude=("sentiment_magnitude", "mean"))
                     .sort_values("Reviews", ascending=False).reset_index())
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.bar_chart(breakdown.set_index(dim)["Avg_Sentiment_Score"])

# --------------------------------------------------------------------------------
# 3. HR Themes (+ drill-down: category -> keyword -> reviews)
# --------------------------------------------------------------------------------
with tabs[2]:
    if f_cat_summary.empty:
        st.info("No HR themes detected in the current filter selection.")
    else:
        st.dataframe(f_cat_summary, use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Drill down: category → keyword → reviews")
        chosen_cat = st.selectbox("HR Category", f_cat_summary["HR_Category"].tolist())
        cat_keywords = f_kw_analysis[f_kw_analysis["HR_Category"] == chosen_cat].sort_values("Mentions", ascending=False)
        st.dataframe(cat_keywords, use_container_width=True, hide_index=True)
        if not cat_keywords.empty:
            chosen_kw = st.selectbox("Keyword / phrase", cat_keywords["Keyword_or_Phrase"].tolist())
            kw_review_ids = filtered_keyword_long[
                (filtered_keyword_long["HR_Category"] == chosen_cat) &
                (filtered_keyword_long["Keyword_or_Phrase"] == chosen_kw)
            ]["review_id"].unique()
            drill_reviews = filtered[filtered["review_id"].isin(kw_review_ids)]
            st.caption(f"{len(drill_reviews)} underlying review(s)")
            show_cols = [c for c in ["review_id", "date", "employee_title", "location", "pros", "cons",
                                      "final_sentiment", "sentiment_score", "sentiment_magnitude", "sentiment_reason"]
                         if c in drill_reviews.columns]
            st.dataframe(drill_reviews[show_cols], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------
# 4. Positive Drivers
# --------------------------------------------------------------------------------
with tabs[3]:
    if f_pos_drivers.empty:
        st.info("Not enough data for positive drivers in the current filter selection.")
    else:
        st.dataframe(f_pos_drivers, use_container_width=True, hide_index=True)
        st.bar_chart(f_pos_drivers.set_index("Keyword_or_Phrase")["Net_Sentiment_%"].head(15))

# --------------------------------------------------------------------------------
# 5. HR Risk / Mixed Signals
# --------------------------------------------------------------------------------
with tabs[4]:
    st.caption(
        "A keyword mentioned by only a handful of reviewers is labeled a 'Low-volume signal', "
        "never stated as a major organizational problem, regardless of its negative/mixed %."
    )
    if f_risk.empty:
        st.info("No flagged risk signals in the current filter selection.")
    else:
        st.dataframe(f_risk, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------
# 6. Employee Reviews (search/filter individual reviews)
# --------------------------------------------------------------------------------
with tabs[5]:
    search = st.text_input("Search review text (pros / cons / title)")
    display_df = filtered
    if search:
        text_cols = [c for c in ["pros", "cons", "review_title", "advice_to_mgmt"] if c in filtered.columns]
        mask = False
        for c in text_cols:
            mask = mask | filtered[c].astype(str).str.contains(search, case=False, na=False)
        display_df = filtered[mask]
    st.caption(f"{len(display_df)} review(s)")
    show_cols = [c for c in [
        "review_id", "date", "employee_title", "location", "employee_status", "employment_type", "tenure",
        "review_title", "rating_overall", "pros", "cons", "advice_to_mgmt",
        "final_sentiment", "sentiment_score", "sentiment_magnitude", "sentiment_reason",
        "HR_Categories", "HR_Keywords", "HR_Reason",
    ] if c in display_df.columns]
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True, height=500)

# --------------------------------------------------------------------------------
# 7. Methodology / Data Quality
# --------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Methodology")
    st.dataframe(result.methodology, use_container_width=True, hide_index=True)

    st.subheader("Data Quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows analysed", result.clean_report["rows_out"])
    c2.metric("Duplicates removed", result.clean_report["duplicates_removed"])
    c3.metric("Rows without any review text", result.clean_report.get("rows_without_any_text", 0))

    st.write("Missing-field counts (before dropping anything):")
    missing_df = pd.DataFrame(list(result.clean_report["missing_field_counts"].items()),
                               columns=["Field", "Missing Count"])
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.subheader("HR Dictionary Coverage")
    st.dataframe(result.dictionary_coverage, use_container_width=True, hide_index=True)

    if result.data_schema.missing_optional:
        st.warning("Disabled (not found in this file): " + ", ".join(result.data_schema.missing_optional))

    st.subheader("Detected Column Mapping")
    mapping_df = pd.DataFrame(list(result.data_schema.reverse_mapping.items()),
                               columns=["Canonical Field", "Original Column"])
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧪 Benchmark against a reference analysis (optional)")
    st.caption(
        "Upload a reference employee-level sentiment file (for example, an existing "
        "Bosch_Final_HR_Sentiment_Analysis_FINAL.xlsx-style export) to compare this app's "
        "sentiment labels against it, review by review. Use this before changing any "
        "threshold — only genuine, evidenced mismatches should drive a change."
    )
    ref_file = st.file_uploader("Reference file (xlsx or csv)", type=["xlsx", "xls", "csv"], key="benchmark_upload")
    if ref_file:
        try:
            reference_df = benchmark.load_reference_sentiment(ref_file.getvalue(), ref_file.name)
            report = benchmark.compare(reviews, reference_df)
            st.metric("Agreement with reference", f"{report['agreement_rate'] * 100:.1f}%",
                       help=f"Compared on {report['n_compared']} reviews")
            st.write("Confusion matrix (rows = reference label, columns = this app's label):")
            st.dataframe(report["confusion_matrix"], use_container_width=True)
            st.write(f"Mismatched reviews ({len(report['mismatches'])}):")
            st.dataframe(report["mismatches"], use_container_width=True, hide_index=True, height=300)
        except ValueError as e:
            st.error(str(e))
