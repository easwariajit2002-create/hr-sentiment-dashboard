from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import aggregate, clean, schema
from .hr_dictionary import HRDictionaryMatcher
from .sentiment import SentimentModel, analyze_sentiment


@dataclass
class PipelineResult:
    reviews: pd.DataFrame
    category_summary: pd.DataFrame
    keyword_analysis: pd.DataFrame
    positive_drivers: pd.DataFrame
    risk_signals: pd.DataFrame
    dictionary_coverage: pd.DataFrame
    methodology: pd.DataFrame
    category_long: pd.DataFrame
    keyword_long: pd.DataFrame
    data_schema: schema.SchemaResult
    clean_report: dict
    dict_schema: schema.SchemaResult


def run_pipeline(raw_df: pd.DataFrame, raw_dict_df: pd.DataFrame, model: SentimentModel,
                  model_name: str) -> PipelineResult:
    # 1. Schema detection + mapping
    data_schema = schema.detect_schema(raw_df)
    if not data_schema.ok:
        raise ValueError("; ".join(data_schema.missing_required))
    mapped_df = schema.apply_mapping(raw_df, data_schema)

    dict_schema = schema.detect_dictionary_schema(raw_dict_df)
    if not dict_schema.ok:
        raise ValueError(
            "HR dictionary is missing required columns: " + ", ".join(dict_schema.missing_required)
        )
    mapped_dict_df = schema.apply_mapping(raw_dict_df, dict_schema)

    # 2. Clean
    clean_df, clean_report = clean.clean_dataframe(mapped_df)

    # 3. Sentiment (contextual, rating never touched)
    sentiment_df = analyze_sentiment(clean_df, model)

    # 4. HR dictionary matching (topic only, does not touch sentiment)
    matcher = HRDictionaryMatcher(mapped_dict_df)
    reviews, category_long, keyword_long = matcher.match_dataframe(sentiment_df)
    coverage = matcher.coverage_report(reviews)

    # 5. Aggregates
    cat_summary = aggregate.hr_category_summary(category_long)
    kw_analysis = aggregate.keyword_analysis(keyword_long)
    pos_drivers = aggregate.positive_drivers(kw_analysis)
    risk = aggregate.hr_risk_signals(kw_analysis)
    cov_df = aggregate.dictionary_coverage(coverage)
    methodology = aggregate.methodology_table(
        model_name=model_name,
        n_rows=len(reviews),
        n_dict_entries=coverage["Dictionary Entries"],
        coverage_pct=coverage["HR Dictionary Coverage (%)"],
        n_categories=cat_summary["HR_Category"].nunique() if not cat_summary.empty else 0,
        n_category_reviews=len(category_long),
    )

    return PipelineResult(
        reviews=reviews,
        category_summary=cat_summary,
        keyword_analysis=kw_analysis,
        positive_drivers=pos_drivers,
        risk_signals=risk,
        dictionary_coverage=cov_df,
        methodology=methodology,
        category_long=category_long,
        keyword_long=keyword_long,
        data_schema=data_schema,
        clean_report=clean_report,
        dict_schema=dict_schema,
    )
