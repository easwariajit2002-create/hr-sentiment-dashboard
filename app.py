"""
HR Employee Sentiment Intelligence — reusable dashboard.

Deploy on Streamlit Community Cloud (or run locally with
`streamlit run app.py`). Upload a RAW company review export (Bosch today,
Dell tomorrow, anything with a broadly similar column shape) plus an HR
keyword/sentiment dictionary (a sensible default is bundled), and the
backend automatically cleans it, scores contextual sentiment, tags HR
themes, and renders a senior-leadership-ready dashboard.

Nothing here is company-specific: every column reference goes through the
schema-detection layer in hr_pipeline/schema.py. The underlying pipeline
(schema, cleaning, sentiment model, HR dictionary matching, aggregation,
benchmark tool) is unchanged from the working version — this file only
upgrades the UI layer around it.
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hr_pipeline.pipeline import run_pipeline
from hr_pipeline.sentiment import MODEL_NAME, SentimentModel
from hr_pipeline import benchmark

st.set_page_config(page_title="HR Employee Sentiment Intelligence", layout="wide",
                    initial_sidebar_state="expanded")

DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), "assets", "default_hr_dictionary.xlsx")
SMALL_SAMPLE_N = 10  # below this many reviews, a theme is flagged as a small sample
CORPORATE_COLORS = {
    "Positive": "#2E7D32", "Negative": "#C62828", "Neutral": "#78909C", "Mixed": "#F9A825",
}

# --------------------------------------------------------------------------------
# Corporate styling — clean, minimal, presentation-friendly
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3 { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; letter-spacing: -0.01em; }
    h1 { color: #1a2b40; font-weight: 700; }
    h2, h3 { color: #24344a; }
    [data-testid="stMetric"] {
        background: #f7f9fb; border: 1px solid #e6eaee; border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    [data-testid="stMetricLabel"] { color: #5b6b7c; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #1a2b40; font-weight: 700; }
    div[data-testid="stExpander"] { border: 1px solid #e6eaee; border-radius: 10px; }
    .insight-card {
        background: #f7f9fb; border-left: 4px solid #1a2b40; border-radius: 6px;
        padding: 10px 14px; margin-bottom: 8px; font-size: 0.95rem; color: #24344a;
    }
    .small-sample-tag {
        display: inline-block; background: #fff3e0; color: #8a5a00; font-size: 0.75rem;
        padding: 2px 8px; border-radius: 10px; margin-left: 6px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# Cached, expensive resources (unchanged pipeline underneath)
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


@st.cache_data(show_spinner="Cleaning data, running sentiment analysis, and tagging HR themes...")
def cached_pipeline(review_bytes: bytes, review_name: str, dict_bytes: bytes, dict_name: str):
    review_df = read_table(review_bytes, review_name)
    dict_df = read_table(dict_bytes, dict_name)
    model = load_model()
    result = run_pipeline(review_df, dict_df, model, model_name=MODEL_NAME)
    return result


def small_sample_tag(n: int) -> str:
    return " <span class='small-sample-tag'>Small sample — interpret cautiously</span>" if n < SMALL_SAMPLE_N else ""


# --------------------------------------------------------------------------------
# Sidebar: uploads
# --------------------------------------------------------------------------------
st.sidebar.title("📥 Data")
review_file = st.sidebar.file_uploader("Employee review export — RAW CSV or Excel, any company", type=["csv", "xlsx", "xls"])
dict_file = st.sidebar.file_uploader(
    "HR keyword/sentiment dictionary (optional — bundled default used otherwise)",
    type=["xlsx", "xls", "csv"],
)

st.title("🧭 HR Employee Sentiment Intelligence")
st.caption(
    "Upload a raw employee-review export from any company. The backend detects columns, cleans the "
    "data, removes duplicates, runs contextual sentiment analysis, tags HR themes, and builds the "
    "dashboard below automatically — no separate cleaning step required."
)

if not review_file:
    st.info("⬅️ Upload a raw employee review export (CSV or Excel) to get started.")
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

# --------------------------------------------------------------------------------
# Data Quality summary — always visible, right after a successful run
# --------------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### 📊 Data Quality")
    cr = result.clean_report
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw rows", f"{cr['rows_in']:,}")
    c2.metric("Duplicates removed", f"{cr['duplicates_removed']:,}")
    c3.metric("Final rows analysed", f"{cr['rows_out']:,}")
    c4.metric("Rows without any text", f"{cr.get('rows_without_any_text', 0):,}")
    cov_row = result.dictionary_coverage
    coverage_pct = cov_row.loc[cov_row["Metric"] == "HR Dictionary Coverage (%)", "Value"]
    c5.metric("HR dictionary coverage", f"{coverage_pct.iloc[0]}%" if len(coverage_pct) else "N/A")

    detected = sorted(result.data_schema.reverse_mapping.keys())
    missing = result.data_schema.missing_optional
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.caption(f"**Detected fields ({len(detected)}):** " + ", ".join(detected))
    with dcol2:
        if missing:
            st.caption(f"**Missing / unavailable fields ({len(missing)}) — filters disabled, never inferred:** "
                       + ", ".join(missing))
        else:
            st.caption("**Missing / unavailable fields:** none — every optional field was detected.")

# --------------------------------------------------------------------------------
# Sidebar: cross-filters (apply everywhere) — keyed to a version counter so
# "Reset Filters" can clear every widget back to its default in one click.
# --------------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")
if "filter_version" not in st.session_state:
    st.session_state["filter_version"] = 0
if st.sidebar.button("↺ Reset Filters", use_container_width=True):
    st.session_state["filter_version"] += 1
    st.rerun()
v = st.session_state["filter_version"]


def multiselect_if_available(label, col):
    if col in reviews.columns and reviews[col].notna().any():
        options = sorted(reviews[col].dropna().unique().tolist())
        return st.sidebar.multiselect(label, options, key=f"{col}_{v}")
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
    date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()), key=f"date_{v}")
else:
    st.sidebar.caption("Date: not available in this dataset")

category_options = sorted(result.category_summary["HR_Category"].tolist()) if not result.category_summary.empty else []
category_sel = st.sidebar.multiselect("HR Category", category_options, key=f"cat_{v}")
sentiment_sel = st.sidebar.multiselect("Sentiment", ["Positive", "Negative", "Neutral", "Mixed"], key=f"sent_{v}")


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

if filtered.empty:
    st.warning("No reviews match the current filter combination. Try Reset Filters in the sidebar.")
    st.stop()

filtered_ids = set(filtered["review_id"])
filtered_category_long = result.category_long[result.category_long["review_id"].isin(filtered_ids)]
filtered_keyword_long = result.keyword_long[result.keyword_long["review_id"].isin(filtered_ids)]

# recompute category/keyword-level tables against the filtered slice so every
# tab and every chart honors all active filters together
from hr_pipeline import aggregate  # noqa: E402  (after st.stop() guards above)

f_cat_summary = aggregate.hr_category_summary(filtered_category_long)
f_kw_analysis = aggregate.keyword_analysis(filtered_keyword_long)
f_pos_drivers = aggregate.positive_drivers(f_kw_analysis)
f_risk = aggregate.hr_risk_signals(f_kw_analysis)


def representative_reviews(category: str, keyword: str, n: int = 3) -> pd.DataFrame:
    ids = filtered_keyword_long[(filtered_keyword_long["HR_Category"] == category) &
                                 (filtered_keyword_long["Keyword_or_Phrase"] == keyword)]["review_id"].unique()
    sub = filtered[filtered["review_id"].isin(ids)]
    show_cols = [c for c in ["date", "employee_title", "location", "pros", "cons",
                              "final_sentiment", "sentiment_score", "sentiment_reason"] if c in sub.columns]
    return sub[show_cols].head(n)


def build_insights() -> list:
    insights = []
    n_total = len(filtered)

    # Largest review concentration
    for dim, label in [("location", "location"), ("employee_title", "role")]:
        if dim in filtered.columns and filtered[dim].notna().any():
            top_val = filtered[dim].value_counts().idxmax()
            top_n = filtered[dim].value_counts().max()
            insights.append(
                f"Largest review concentration: **{top_val}** ({label}) accounts for "
                f"{top_n} of {n_total} reviews ({100 * top_n / n_total:.0f}%)."
            )
            break

    # Strongest positive theme (meaningful sample size)
    if not f_cat_summary.empty:
        pos_candidates = f_cat_summary[f_cat_summary["Unique_Reviews"] >= 5].sort_values("Positive_%", ascending=False)
        if not pos_candidates.empty:
            row = pos_candidates.iloc[0]
            insights.append(
                f"Strongest positive theme: **{row['HR_Category']}** — {row['Positive_%']}% positive "
                f"across {row['Unique_Reviews']} reviews."
            )
        # Strongest negative theme
        neg_candidates = f_cat_summary[f_cat_summary["Unique_Reviews"] >= 5].sort_values("Negative_%", ascending=False)
        if not neg_candidates.empty and neg_candidates.iloc[0]["Negative_%"] > 0:
            row = neg_candidates.iloc[0]
            insights.append(
                f"Strongest negative theme: **{row['HR_Category']}** — {row['Negative_%']}% negative "
                f"across {row['Unique_Reviews']} reviews."
            )
        # Strongest mixed theme
        mix_candidates = f_cat_summary[f_cat_summary["Unique_Reviews"] >= 5].sort_values("Mixed_%", ascending=False)
        if not mix_candidates.empty and mix_candidates.iloc[0]["Mixed_%"] > 0:
            row = mix_candidates.iloc[0]
            insights.append(
                f"Most polarizing theme: **{row['HR_Category']}** shows {row['Mixed_%']}% mixed sentiment "
                f"across {row['Unique_Reviews']} reviews — genuine praise and criticism both present."
            )

    # Overall tone
    counts = filtered["final_sentiment"].value_counts()
    pos_pct = 100 * counts.get("Positive", 0) / n_total
    neg_pct = 100 * counts.get("Negative", 0) / n_total
    if pos_pct >= neg_pct * 2 and pos_pct > 40:
        insights.append(f"Overall tone skews positive: {pos_pct:.0f}% positive vs {neg_pct:.0f}% negative.")
    elif neg_pct >= pos_pct and neg_pct > 20:
        insights.append(f"Overall tone shows meaningful concern: {neg_pct:.0f}% negative vs {pos_pct:.0f}% positive.")

    # Rating vs sentiment alignment
    if "rating_overall" in filtered.columns and filtered["rating_overall"].notna().sum() > 10:
        corr = filtered[["rating_overall", "sentiment_score"]].dropna().corr().iloc[0, 1]
        if pd.notna(corr):
            direction = "aligns closely with" if corr > 0.5 else ("only loosely tracks" if corr > 0.2 else "diverges from")
            insights.append(f"Contextual sentiment {direction} the star rating (correlation = {corr:.2f}).")

    return insights


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

    st.markdown("#### 🔑 Key Insights")
    insights = build_insights()
    if insights:
        for ins in insights:
            st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)
    else:
        st.caption("Not enough data in the current filter selection to generate insights.")

    st.divider()
    st.markdown("#### 📈 Charts")
    chart_row1_a, chart_row1_b = st.columns(2)

    with chart_row1_a:
        dist_df = counts.rename_axis("Sentiment").reset_index(name="Reviews")
        fig = px.bar(dist_df, x="Sentiment", y="Reviews", color="Sentiment",
                     color_discrete_map=CORPORATE_COLORS, title="Sentiment Distribution", text="Reviews")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with chart_row1_b:
        if "date" in filtered.columns and filtered["date"].notna().sum() > 3:
            trend = filtered.dropna(subset=["date"]).copy()
            trend["period"] = trend["date"].dt.to_period("M").dt.to_timestamp()
            trend_agg = trend.groupby("period")["sentiment_score"].mean().reset_index()
            fig = px.line(trend_agg, x="period", y="sentiment_score", markers=True,
                          title="Average Sentiment Score Over Time")
            fig.update_layout(margin=dict(t=40, b=10), yaxis_title="Avg. Sentiment Score", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sentiment trend over time: date field not available or too sparse in this dataset.")

    chart_row2_a, chart_row2_b = st.columns(2)
    with chart_row2_a:
        if "location" in filtered.columns and filtered["location"].notna().any():
            top_locations = filtered["location"].value_counts().head(10).index.tolist()
            loc_df = filtered[filtered["location"].isin(top_locations)]
            ct = pd.crosstab(loc_df["location"], loc_df["final_sentiment"])
            ct = ct.reindex(top_locations)
            fig = go.Figure()
            for sentiment in ["Positive", "Negative", "Neutral", "Mixed"]:
                if sentiment in ct.columns:
                    fig.add_trace(go.Bar(name=sentiment, x=ct.index, y=ct[sentiment],
                                          marker_color=CORPORATE_COLORS[sentiment]))
            fig.update_layout(barmode="stack", title="Sentiment by Location (top 10)", margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sentiment by location: location field not available in this dataset.")

    with chart_row2_b:
        if not f_cat_summary.empty:
            top_cat = f_cat_summary.sort_values("Unique_Reviews", ascending=False).head(10)
            fig = px.bar(top_cat.sort_values("Unique_Reviews"), x="Unique_Reviews", y="HR_Category",
                         orientation="h", title="Top HR Categories by Review Volume", text="Unique_Reviews")
            fig.update_layout(margin=dict(t=40, b=10), yaxis_title="", xaxis_title="Reviews")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No HR themes detected in the current filter selection.")

    chart_row3_a, chart_row3_b = st.columns(2)
    with chart_row3_a:
        if not f_cat_summary.empty:
            top_cat = f_cat_summary.sort_values("Unique_Reviews", ascending=False).head(10)
            melted = top_cat.melt(id_vars=["HR_Category"],
                                   value_vars=["Positive_%", "Negative_%", "Neutral_%", "Mixed_%"],
                                   var_name="Sentiment", value_name="Percent")
            melted["Sentiment"] = melted["Sentiment"].str.replace("_%", "")
            fig = px.bar(melted, x="Percent", y="HR_Category", color="Sentiment", orientation="h",
                         color_discrete_map=CORPORATE_COLORS, title="HR Category Sentiment Breakdown")
            fig.update_layout(margin=dict(t=40, b=10), yaxis_title="", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No HR themes detected in the current filter selection.")

    with chart_row3_b:
        if "rating_overall" in filtered.columns and filtered["rating_overall"].notna().sum() > 5:
            rating_df = filtered.dropna(subset=["rating_overall"]).copy()
            fig = px.box(rating_df, x="rating_overall", y="sentiment_score",
                         title="Sentiment Score by Star Rating", points=False)
            fig.update_layout(margin=dict(t=40, b=10), xaxis_title="Star Rating", yaxis_title="Sentiment Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Rating vs. sentiment: rating field not available or too sparse in this dataset.")

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
        agg_kwargs = dict(Reviews=("review_id", "count"),
                           Avg_Sentiment_Score=("sentiment_score", "mean"),
                           Avg_Magnitude=("sentiment_magnitude", "mean"))
        if "rating_overall" in filtered.columns:
            agg_kwargs["Avg_Rating"] = ("rating_overall", "mean")
        breakdown = (filtered.groupby(dim).agg(**agg_kwargs)
                     .sort_values("Reviews", ascending=False).reset_index())
        st.dataframe(breakdown.round(2), use_container_width=True, hide_index=True)
        fig = px.bar(breakdown.sort_values("Avg_Sentiment_Score"), x="Avg_Sentiment_Score", y=dim,
                     orientation="h", title=f"Average Sentiment Score by {dim.replace('_', ' ').title()}")
        fig.update_layout(margin=dict(t=40, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------
# 3. HR Themes (+ drill-down: category -> keyword -> reviews)
# --------------------------------------------------------------------------------
with tabs[2]:
    if f_cat_summary.empty:
        st.info("No HR themes detected in the current filter selection.")
    else:
        st.dataframe(f_cat_summary, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("#### 🔍 Drill down: category → keyword → reviews")
        chosen_cat = st.selectbox("HR Category", f_cat_summary["HR_Category"].tolist())
        cat_keywords = f_kw_analysis[f_kw_analysis["HR_Category"] == chosen_cat].sort_values("Mentions", ascending=False)

        cat_row = f_cat_summary[f_cat_summary["HR_Category"] == chosen_cat].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Reviews touching this category", int(cat_row["Unique_Reviews"]))
        m2.metric("Avg. sentiment score", f"{cat_row['Average_Sentiment_Score']:+.2f}")
        m3.metric("Positive / Negative / Mixed", f"{cat_row['Positive_%']}% / {cat_row['Negative_%']}% / {cat_row['Mixed_%']}%")

        st.dataframe(cat_keywords, use_container_width=True, hide_index=True)
        if not cat_keywords.empty:
            chosen_kw = st.selectbox("Keyword / phrase", cat_keywords["Keyword_or_Phrase"].tolist())
            kw_review_ids = filtered_keyword_long[
                (filtered_keyword_long["HR_Category"] == chosen_cat) &
                (filtered_keyword_long["Keyword_or_Phrase"] == chosen_kw)
            ]["review_id"].unique()
            drill_reviews = filtered[filtered["review_id"].isin(kw_review_ids)]
            st.caption(f"{len(drill_reviews)} underlying review(s)" +
                       (" — small sample, interpret cautiously" if len(drill_reviews) < SMALL_SAMPLE_N else ""))
            show_cols = [c for c in ["review_id", "date", "employee_title", "location", "pros", "cons",
                                      "final_sentiment", "sentiment_score", "sentiment_magnitude", "sentiment_reason"]
                         if c in drill_reviews.columns]
            st.dataframe(drill_reviews[show_cols], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------
# 4. Positive Drivers
# --------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### ✅ Major Positive HR Themes")
    if f_pos_drivers.empty:
        st.info("Not enough data for positive drivers in the current filter selection.")
    else:
        display_df = f_pos_drivers.copy()
        display_df["Sample"] = display_df["Unique_Reviews"].apply(
            lambda n: "Small sample — interpret cautiously" if n < SMALL_SAMPLE_N else "")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        fig = px.bar(f_pos_drivers.sort_values("Net_Sentiment_%").head(15), x="Net_Sentiment_%", y="Keyword_or_Phrase",
                     orientation="h", title="Top Positive Drivers (net sentiment %)", color_discrete_sequence=["#2E7D32"])
        fig.update_layout(margin=dict(t=40, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Representative reviews")
        top_row = f_pos_drivers.iloc[0]
        st.caption(f"Showing sample reviews for the strongest driver: **{top_row['Keyword_or_Phrase']}** "
                   f"({top_row['HR_Category']})")
        st.dataframe(representative_reviews(top_row["HR_Category"], top_row["Keyword_or_Phrase"]),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------
# 5. HR Risk / Mixed Signals
# --------------------------------------------------------------------------------
with tabs[4]:
    st.caption(
        "A keyword mentioned by only a handful of reviewers is labeled a 'Low-volume signal', "
        "never stated as a major organizational problem, regardless of its negative/mixed %."
    )
    st.markdown("#### ⚠️ Negative / High-Risk Themes")
    if f_risk.empty:
        st.info("No flagged risk signals in the current filter selection.")
    else:
        neg_focus = f_risk[f_risk["Negative_%"] >= f_risk["Mixed_%"]].copy()
        if neg_focus.empty:
            st.caption("No themes are predominantly negative in the current filter selection.")
        else:
            neg_focus["Sample"] = neg_focus["Unique_Reviews"].apply(
                lambda n: "Small sample — interpret cautiously" if n < SMALL_SAMPLE_N else "")
            st.dataframe(neg_focus, use_container_width=True, hide_index=True)
            top_row = neg_focus.iloc[0]
            st.caption(f"Representative reviews for **{top_row['Keyword_or_Phrase']}** ({top_row['HR_Category']})"
                       + small_sample_tag(top_row["Unique_Reviews"]), unsafe_allow_html=True)
            st.dataframe(representative_reviews(top_row["HR_Category"], top_row["Keyword_or_Phrase"]),
                         use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🔀 Mixed-Sentiment Themes")
    if f_kw_analysis.empty:
        st.info("No HR themes detected in the current filter selection.")
    else:
        mixed_df = f_kw_analysis.copy()
        mixed_df["Mixed_%"] = (100 * mixed_df["Mixed_Mentions"] / mixed_df["Total_Sentiment_Mentions"]
                                .replace(0, pd.NA)).fillna(0).round(1)
        mixed_df = mixed_df[(mixed_df["Mixed_%"] > 0) & (mixed_df["Unique_Reviews"] >= 3)]
        mixed_df = mixed_df.sort_values(["Mixed_%", "Unique_Reviews"], ascending=[False, False])
        if mixed_df.empty:
            st.caption("No genuinely mixed-sentiment themes in the current filter selection.")
        else:
            mixed_df["Sample"] = mixed_df["Unique_Reviews"].apply(
                lambda n: "Small sample — interpret cautiously" if n < SMALL_SAMPLE_N else "")
            show_cols = ["HR_Category", "Keyword_or_Phrase", "Unique_Reviews", "Average_Score",
                         "Mixed_%", "Positive_Mentions", "Negative_Mentions", "Sample"]
            st.dataframe(mixed_df[show_cols].head(15), use_container_width=True, hide_index=True)
            top_row = mixed_df.iloc[0]
            st.caption(f"Representative reviews for **{top_row['Keyword_or_Phrase']}** ({top_row['HR_Category']})"
                       + small_sample_tag(top_row["Unique_Reviews"]), unsafe_allow_html=True)
            st.dataframe(representative_reviews(top_row["HR_Category"], top_row["Keyword_or_Phrase"]),
                         use_container_width=True, hide_index=True)

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

    st.subheader("Data Quality (detail)")
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
