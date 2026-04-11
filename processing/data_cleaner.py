"""
processing/data_cleaner.py
==========================
Handles all preprocessing of job data before NLP and ML analysis.

Two main pipelines:
  1. clean_data()     — basic cleaning for display and skill analysis
  2. prepare_for_ml() — deeper cleaning specifically for ML model input

Pipeline flow:
  MongoDB / CSV -> clean_data() -> prepare_for_ml() -> ML-ready DataFrame
"""

import re
import pandas as pd
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS     = ["title", "company", "location", "description"]
TEXT_COLUMNS         = ["title", "company", "location", "description"]
URL_PATTERN          = re.compile(r"http\S+|www\.\S+")
WHITESPACE_PATTERN   = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")


# ---------------------------------------------------------------------------
# Stage 1 — Basic cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the standard cleaning pipeline on raw scraped job data.

    Steps:
      1. Add any missing columns with empty string defaults
      2. Remove exact duplicate rows
      3. Normalize text: lowercase, strip whitespace, remove URLs
      4. Drop rows with no usable title or company
      5. Reset index

    Args:
        df: Raw job data from scraper or MongoDB load.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for display and skill analysis.
    """
    if df is None or df.empty:
        log.warning("clean_data received an empty DataFrame")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = df.copy()
    log.info("clean_data started: %d rows", len(df))

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            log.debug("Added missing column: %s", col)

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.info("Removed %d exact duplicate rows", removed)

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().apply(_clean_text)

    before = len(df)
    df = df[df["title"].str.strip().ne("") & df["title"].ne("nan")]
    df = df[df["company"].str.strip().ne("") & df["company"].ne("nan")]
    df = df.dropna(subset=["title", "company"])
    removed = before - len(df)
    if removed:
        log.info("Dropped %d rows with missing title or company", removed)

    df = df.reset_index(drop=True)
    log.info("clean_data complete: %d records", len(df))
    return df


# ---------------------------------------------------------------------------
# Stage 2 — ML preparation
# ---------------------------------------------------------------------------

def prepare_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deeper text normalization to prepare data for ML models.

    Goes further than clean_data() by:
      - Removing special characters and punctuation
      - Creating a combined 'text' feature column (title + description)
      - Dropping rows where description is empty
      - Final deduplication on combined text

    Args:
        df: Output of clean_data().

    Returns:
        pd.DataFrame: ML-ready DataFrame with an added 'text' column.
    """
    if df.empty:
        log.warning("prepare_for_ml received an empty DataFrame")
        return df

    df = df.copy()
    log.info("prepare_for_ml started: %d rows", len(df))

    for col in ["title", "description"]:
        if col in df.columns:
            df[col] = df[col].apply(_deep_clean_text)

    before = len(df)
    df = df[df["description"].str.strip().ne("")]
    removed = before - len(df)
    if removed:
        log.info("Dropped %d rows with empty description", removed)

    df["text"] = (df["title"] + " " + df["description"].fillna("")).str.strip()
    log.debug("Created combined 'text' feature column")

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    removed = before - len(df)
    if removed:
        log.info("Removed %d near-duplicate records based on text column", removed)

    df = df.reset_index(drop=True)
    log.info("prepare_for_ml complete: %d records with 'text' feature", len(df))
    return df


def _clean_text(text: str) -> str:
    """Remove URLs and collapse whitespace."""
    text = URL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _deep_clean_text(text: str) -> str:
    """Remove URLs, special characters, and collapse whitespace."""
    text = URL_PATTERN.sub(" ", text)
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()