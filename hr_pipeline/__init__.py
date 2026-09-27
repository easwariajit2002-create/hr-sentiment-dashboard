"""Reusable HR Employee Sentiment Intelligence pipeline.

This package is intentionally company-agnostic: every module works off a
detected column mapping rather than hard-coded column names, so the same
code path handles Bosch today and Dell (or any other company with a
similarly-shaped export) tomorrow without edits.
"""
