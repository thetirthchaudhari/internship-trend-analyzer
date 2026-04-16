"""
RAG-style salary prediction for pasted job descriptions.

Design:
  1. Retrieve the closest scraped job descriptions to understand the wording
     and role framing used in LinkedIn / Naukri postings.
  2. Retrieve salary-bearing examples from the Kaggle salary dataset
     (plus any local salary rows) using the enriched query.
  3. Produce a salary prediction from the retrieved evidence.
  4. If CEREBRAS_API_KEY is configured, ask Cerebras to refine the final answer.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any
from urllib import error, request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None

from logger import get_logger
from settings import (
    CATEGORY_LABELS,
    CEREBRAS_API_KEY,
    CEREBRAS_MODEL,
    KAGGLE_SALARY_DATA_PATH,
    PREDICTION_STATUS_OPTIONS,
    SCRAPED_DATA_PATHS,
)

log = get_logger(__name__)

SCRAPED_TEXT_FIELDS = (
    "title",
    "company",
    "location",
    "role_category",
    "search_query",
    "description",
)

SALARY_TEXT_FIELDS = (
    "title",
    "company",
    "location",
    "role_category",
    "search_query",
    "employment_status",
    "description_hint",
)

INTERNSHIP_MARKERS = ("intern", "internship", "trainee")
KNOWN_EMPLOYMENT_STATUSES = tuple(
    value for value, _label in PREDICTION_STATUS_OPTIONS
)
UNPAID_MARKERS = (
    "unpaid",
    "without stipend",
    "no stipend",
    "not paid",
    "volunteer",
)


def predict_salary(
    job_description: str,
    job_title: str = "",
    job_location: str = "",
    employment_mode: str = "intern",
    allowed_employment_statuses: list[str] | None = None,
    allowed_scraped_categories: list[str] | None = None,
    top_k_salary: int = 5,
    top_k_descriptions: int = 3,
) -> dict[str, Any]:
    """
    Predict salary from a pasted job description using a hybrid RAG flow.
    """
    base_query = _join_text_parts(job_title, job_location, job_description)
    if not base_query:
        raise ValueError("Job description is required for prediction.")

    scraped_df = build_scraped_description_corpus(load_scraped_jobs())
    selected_scraped_categories = normalize_scraped_categories(
        allowed_scraped_categories
    )
    description_examples = retrieve_description_examples(
        scraped_df=scraped_df,
        query_text=base_query,
        allowed_scraped_categories=selected_scraped_categories,
        top_k=top_k_descriptions,
    )

    salary_df = build_salary_corpus(load_salary_jobs())
    if salary_df.empty:
        return {
            "prediction": "Salary prediction unavailable",
            "confidence": "low",
            "method": "unavailable",
            "summary": (
                "No salary-bearing records were found in the Kaggle dataset or the local scraped data."
            ),
            "retrieved_examples": [],
            "description_examples": description_examples,
            "salary_corpus_size": 0,
            "used_cerebras": False,
            "cerebras_status": "not_applicable",
        }

    enriched_query = build_enriched_query(
        base_query=base_query,
        description_examples=description_examples,
    )
    employment_mode = normalize_employment_mode(employment_mode)
    selected_statuses = normalize_employment_statuses(
        allowed_employment_statuses,
        employment_mode=employment_mode,
    )
    prefer_internship = resolve_internship_preference(
        employment_mode=employment_mode,
        query_text=base_query,
    )
    salary_examples = retrieve_salary_examples(
        salary_df=salary_df,
        query_text=enriched_query,
        top_k=top_k_salary,
        prefer_internship=prefer_internship,
        allowed_employment_statuses=selected_statuses,
        job_title=job_title,
        job_location=job_location,
    )

    heuristic = build_heuristic_prediction(
        salary_examples=salary_examples,
        description_examples=description_examples,
        prefer_internship=prefer_internship,
    )
    cerebras_result, cerebras_status, cerebras_error = generate_with_cerebras(
        job_title=job_title,
        job_location=job_location,
        job_description=job_description,
        salary_examples=salary_examples,
        description_examples=description_examples,
        heuristic_prediction=heuristic["prediction"],
    )

    if cerebras_result:
        result = {
            "prediction": cerebras_result.get("prediction") or heuristic["prediction"],
            "confidence": cerebras_result.get("confidence") or heuristic["confidence"],
            "method": "cerebras_rag",
            "summary": cerebras_result.get("summary") or heuristic["summary"],
            "retrieved_examples": salary_examples,
            "description_examples": description_examples,
            "salary_corpus_size": len(salary_df),
            "employment_mode": employment_mode,
            "selected_employment_statuses": selected_statuses,
            "selected_scraped_categories": selected_scraped_categories,
            "salary_examples_used": len(salary_examples),
            "used_cerebras": True,
            "cerebras_status": cerebras_status,
            "cerebras_error": None,
        }
    else:
        result = {
            "prediction": heuristic["prediction"],
            "confidence": heuristic["confidence"],
            "method": "hybrid_rag_heuristic",
            "summary": heuristic["summary"],
            "retrieved_examples": salary_examples,
            "description_examples": description_examples,
            "salary_corpus_size": len(salary_df),
            "employment_mode": employment_mode,
            "selected_employment_statuses": selected_statuses,
            "selected_scraped_categories": selected_scraped_categories,
            "salary_examples_used": len(salary_examples),
            "used_cerebras": False,
            "cerebras_status": cerebras_status,
            "cerebras_error": cerebras_error,
        }

    internship_rows = int(salary_df["is_internship_like"].fillna(False).sum())
    if internship_rows < 200:
        result["warning"] = (
            "The internship-specific salary slice is still small, so this estimate may be broad."
        )
    elif not description_examples:
        result["warning"] = (
            "No close scraped descriptions were found, so the prediction relied more heavily on role title and salary metadata."
        )

    return result


@lru_cache(maxsize=1)
def load_scraped_jobs() -> pd.DataFrame:
    """
    Load scraped jobs from MongoDB and CSVs. These rows are used to learn
    how live job descriptions are phrased.
    """
    frames: list[pd.DataFrame] = []

    try:
        from database.mongo_client import get_collection, load_jobs_to_dataframe

        collection = get_collection()
        mongo_df = load_jobs_to_dataframe(collection)
        if not mongo_df.empty:
            frames.append(mongo_df)
    except Exception as exc:
        log.warning("MongoDB description source unavailable: %s", exc)

    for path in SCRAPED_DATA_PATHS:
        if not os.path.exists(path):
            continue
        try:
            csv_df = pd.read_csv(path)
        except Exception as exc:
            log.warning("Could not load scraped source %s: %s", path, exc)
            continue
        if not csv_df.empty:
            frames.append(csv_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates().reset_index(drop=True)


@lru_cache(maxsize=1)
def load_salary_jobs() -> pd.DataFrame:
    """
    Load salary-bearing rows from the Kaggle dataset and supplement them with
    any local salary rows from the scraper outputs.
    """
    frames: list[pd.DataFrame] = []

    kaggle_df = load_kaggle_salary_dataset()
    if not kaggle_df.empty:
        frames.append(kaggle_df)

    local_salary_df = load_local_salary_rows()
    if not local_salary_df.empty:
        frames.append(local_salary_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates().reset_index(drop=True)


def load_kaggle_salary_dataset() -> pd.DataFrame:
    if not os.path.exists(KAGGLE_SALARY_DATA_PATH):
        return pd.DataFrame()

    raw_df = pd.read_csv(KAGGLE_SALARY_DATA_PATH)
    if raw_df.empty:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "title": _series_from_df(raw_df, "Job Title").astype(str).str.strip(),
            "company": _series_from_df(raw_df, "Company Name").astype(str).str.strip(),
            "location": _series_from_df(raw_df, "Location").astype(str).str.strip(),
            "role_category": _series_from_df(raw_df, "Job Roles").astype(str).str.strip(),
            "employment_status": _series_from_df(raw_df, "Employment Status").astype(str).str.strip(),
            "source": "kaggle_salary_dataset",
            "job_url": "",
            "search_query": "",
            "description": "",
            "duration": "",
            "salary_report_count": pd.to_numeric(
                _series_from_df(raw_df, "Salaries Reported", 0),
                errors="coerce",
            ).fillna(0),
            "rating": pd.to_numeric(_series_from_df(raw_df, "Rating", 0), errors="coerce"),
            "salary_annual": pd.to_numeric(_series_from_df(raw_df, "Salary"), errors="coerce"),
        }
    )

    df = df[df["salary_annual"].notna() & df["salary_annual"].gt(0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["salary_monthly_min"] = df["salary_annual"] / 12.0
    df["salary_monthly_max"] = df["salary_annual"] / 12.0
    df["salary_monthly_mid"] = df["salary_annual"] / 12.0
    df["salary_value"] = df["salary_monthly_mid"].apply(
        lambda value: f"Approx. {format_single_monthly(value)}"
    )
    df["is_internship_like"] = (
        df["employment_status"].str.contains("intern|trainee", case=False, na=False)
        | df["title"].str.contains("intern|trainee", case=False, na=False)
    )
    df["description_hint"] = (
        "role="
        + df["role_category"].fillna("")
        + " | employment="
        + df["employment_status"].fillna("")
    )

    return df.reset_index(drop=True)


def load_local_salary_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in SCRAPED_DATA_PATHS:
        if not os.path.exists(path):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            log.warning("Could not load local salary file %s: %s", path, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        return pd.DataFrame()

    salary_raw = pd.Series("", index=df.index, dtype="object")
    for field in ("salary", "Salary", "stipend"):
        if field in df.columns:
            values = df[field].fillna("").astype(str).str.strip()
            salary_raw = salary_raw.mask(salary_raw.eq(""), values)

    normalized = pd.DataFrame(
        {
            "title": _series_from_df(df, "title").astype(str).str.strip(),
            "company": _series_from_df(df, "company").astype(str).str.strip(),
            "location": _series_from_df(df, "location").astype(str).str.strip(),
            "role_category": _series_from_df(df, "role_category").astype(str).str.strip(),
            "employment_status": "Intern",
            "source": _series_from_df(df, "source", "scraped_jobs").fillna("scraped_jobs").astype(str).str.strip(),
            "job_url": _series_from_df(df, "job_url").astype(str).str.strip(),
            "search_query": _series_from_df(df, "search_query").astype(str).str.strip(),
            "description": _series_from_df(df, "description").astype(str).str.strip(),
            "duration": _series_from_df(df, "duration").astype(str).str.strip(),
            "salary_value": salary_raw.fillna("").astype(str).str.strip(),
            "salary_report_count": 1.0,
        }
    )

    normalized = normalized[
        normalized["salary_value"].ne("")
        & normalized["salary_value"].str.lower().ne("nan")
    ].copy()

    if normalized.empty:
        return pd.DataFrame()

    parsed = normalized["salary_value"].apply(parse_salary)
    normalized["salary_monthly_min"] = parsed.apply(
        lambda value: value["minimum"] if value and value["kind"] == "monthly_range" else None
    )
    normalized["salary_monthly_max"] = parsed.apply(
        lambda value: value["maximum"] if value and value["kind"] == "monthly_range" else None
    )
    normalized["salary_monthly_mid"] = parsed.apply(
        lambda value: (value["minimum"] + value["maximum"]) / 2
        if value and value["kind"] == "monthly_range"
        else None
    )
    normalized["salary_annual"] = normalized["salary_monthly_mid"].apply(
        lambda value: value * 12 if pd.notna(value) else None
    )
    normalized["is_internship_like"] = True
    normalized["description_hint"] = normalized["description"].fillna("").astype(str).str.slice(0, 220)

    return normalized.reset_index(drop=True)


def build_scraped_description_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    corpus = df.copy()
    for field in SCRAPED_TEXT_FIELDS:
        if field not in corpus.columns:
            corpus[field] = ""

    corpus["description"] = corpus["description"].fillna("").astype(str).str.strip()
    corpus = corpus[
        corpus["description"].ne("")
        & corpus["description"].str.lower().ne("nan")
    ].copy()
    if corpus.empty:
        return pd.DataFrame()

    corpus["document_text"] = corpus.apply(_build_scraped_text, axis=1)
    corpus["description_excerpt"] = corpus["description"].str.slice(0, 240)
    corpus["job_url"] = corpus["job_url"].fillna("").astype(str).str.strip() if "job_url" in corpus.columns else ""
    corpus["source"] = corpus["source"].fillna("").astype(str).str.strip() if "source" in corpus.columns else ""

    return corpus.drop_duplicates(subset=["title", "company", "description"]).reset_index(drop=True)


def build_salary_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    corpus = df.copy()
    for field in SALARY_TEXT_FIELDS:
        if field not in corpus.columns:
            corpus[field] = ""

    corpus["document_text"] = corpus.apply(_build_salary_text, axis=1)
    corpus["description_excerpt"] = corpus["description_hint"].fillna("").astype(str).str.slice(0, 220)
    corpus["duration_value"] = corpus["duration"].fillna("").astype(str).str.strip() if "duration" in corpus.columns else ""
    corpus["job_url"] = corpus["job_url"].fillna("").astype(str).str.strip() if "job_url" in corpus.columns else ""
    corpus["source"] = corpus["source"].fillna("").astype(str).str.strip() if "source" in corpus.columns else ""

    return corpus.drop_duplicates(subset=["title", "company", "location", "salary_value"]).reset_index(drop=True)


def retrieve_description_examples(
    scraped_df: pd.DataFrame,
    query_text: str,
    allowed_scraped_categories: list[str] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if scraped_df.empty:
        return []

    if allowed_scraped_categories:
        filtered_df = scraped_df[
            scraped_df["role_category"].fillna("").astype(str).isin(allowed_scraped_categories)
        ].copy()
        if filtered_df.empty:
            return []
        scraped_df = filtered_df

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(scraped_df["document_text"].tolist())
    query_vector = vectorizer.transform([query_text])
    similarities = linear_kernel(query_vector, matrix).flatten()
    ranked_indexes = similarities.argsort()[::-1]

    examples = []
    for idx in ranked_indexes:
        similarity = float(similarities[idx])
        if len(examples) >= top_k:
            break
        if similarity <= 0 and examples:
            break

        row = scraped_df.iloc[idx]
        examples.append(
            {
                "title": str(row.get("title", "")).strip(),
                "company": str(row.get("company", "")).strip(),
                "location": str(row.get("location", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "role_category": str(row.get("role_category", "")).strip(),
                "search_query": str(row.get("search_query", "")).strip(),
                "description_excerpt": str(row.get("description_excerpt", "")).strip(),
                "job_url": str(row.get("job_url", "")).strip(),
                "similarity": round(similarity, 4),
            }
        )

    return examples


def build_enriched_query(base_query: str, description_examples: list[dict[str, Any]]) -> str:
    enrichment_parts = [base_query]

    for example in description_examples:
        enrichment_parts.append(
            _join_text_parts(
                example.get("title", ""),
                example.get("role_category", ""),
                example.get("search_query", ""),
                example.get("description_excerpt", ""),
            )
        )

    return "\n".join(part for part in enrichment_parts if part).strip()


def retrieve_salary_examples(
    salary_df: pd.DataFrame,
    query_text: str,
    top_k: int = 5,
    prefer_internship: bool = True,
    allowed_employment_statuses: list[str] | None = None,
    job_title: str = "",
    job_location: str = "",
) -> list[dict[str, Any]]:
    candidates = salary_df.copy()

    if allowed_employment_statuses:
        allowed_lookup = {status.lower() for status in allowed_employment_statuses}
        candidates = candidates[
            candidates["employment_status"].fillna("").astype(str).str.lower().isin(allowed_lookup)
        ].copy()
        if candidates.empty:
            return []

    if prefer_internship:
        internship_df = candidates[candidates["is_internship_like"].fillna(False)].copy()
        if not internship_df.empty:
            candidates = internship_df

    if candidates.empty:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(candidates["document_text"].tolist())
    query_vector = vectorizer.transform([query_text])
    similarities = linear_kernel(query_vector, matrix).flatten()

    query_title_tokens = _tokenize_text(job_title)
    query_location_tokens = _tokenize_text(job_location)

    salary_midpoints = pd.to_numeric(
        candidates.get("salary_monthly_mid", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    upper_outlier_cap = (
        float(salary_midpoints.quantile(0.97))
        if len(salary_midpoints) >= 20
        else None
    )

    ranking_rows = []
    for idx, similarity in enumerate(similarities):
        row = candidates.iloc[idx]
        title_overlap = _token_overlap_score(query_title_tokens, row.get("title", ""))
        category_overlap = _token_overlap_score(
            query_title_tokens,
            row.get("role_category", ""),
        )
        search_query_overlap = _token_overlap_score(
            query_title_tokens,
            row.get("search_query", ""),
        )
        location_overlap = _token_overlap_score(
            query_location_tokens,
            row.get("location", ""),
        )

        report_count = _coerce_positive_number(
            row.get("salary_report_count"),
            default=1.0,
        )
        report_bonus = min(report_count, 6.0) / 60.0

        local_source_bonus = 0.0
        if (
            str(row.get("source", "")).strip().lower() != "kaggle_salary_dataset"
            and (title_overlap > 0 or search_query_overlap > 0)
        ):
            local_source_bonus = 0.08

        midpoint = _coerce_positive_number(row.get("salary_monthly_mid"))
        outlier_penalty = 0.0
        if (
            midpoint is not None
            and upper_outlier_cap is not None
            and midpoint > upper_outlier_cap
        ):
            ratio = midpoint / max(upper_outlier_cap, 1.0)
            outlier_penalty = min(0.18, 0.05 + (ratio - 1.0) * 0.08)

        ranking_score = (
            float(similarity)
            + 0.30 * title_overlap
            + 0.16 * category_overlap
            + 0.12 * search_query_overlap
            + 0.08 * location_overlap
            + report_bonus
            + local_source_bonus
            - outlier_penalty
        )

        ranking_rows.append(
            (
                idx,
                ranking_score,
                float(similarity),
                title_overlap,
                category_overlap,
                search_query_overlap,
            )
        )

    ranked_indexes = [
        idx
        for idx, *_ in sorted(
            ranking_rows,
            key=lambda item: (item[1], item[2], item[3], item[4], item[5]),
            reverse=True,
        )
    ]
    ranking_lookup = {idx: score for idx, score, *_rest in ranking_rows}

    examples = []
    for idx in ranked_indexes:
        similarity = float(similarities[idx])
        if len(examples) >= top_k:
            break
        if similarity <= 0 and examples:
            break

        row = candidates.iloc[idx]
        examples.append(
            {
                "title": str(row.get("title", "")).strip(),
                "company": str(row.get("company", "")).strip(),
                "location": str(row.get("location", "")).strip(),
                "salary": str(row.get("salary_value", "")).strip(),
                "duration": str(row.get("duration_value", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "job_url": str(row.get("job_url", "")).strip(),
                "role_category": str(row.get("role_category", "")).strip(),
                "search_query": str(row.get("search_query", "")).strip(),
                "description_excerpt": str(row.get("description_excerpt", "")).strip(),
                "employment_status": str(row.get("employment_status", "")).strip(),
                "salary_annual": row.get("salary_annual"),
                "salary_monthly_min": row.get("salary_monthly_min"),
                "salary_monthly_max": row.get("salary_monthly_max"),
                "salary_report_count": row.get("salary_report_count", 0),
                "similarity": round(similarity, 4),
                "ranking_score": round(float(ranking_lookup.get(idx, similarity)), 4),
            }
        )

    return examples


def build_heuristic_prediction(
    salary_examples: list[dict[str, Any]],
    description_examples: list[dict[str, Any]],
    prefer_internship: bool,
) -> dict[str, str]:
    if not salary_examples:
        return {
            "prediction": "Salary prediction unavailable",
            "confidence": "low",
            "summary": "No relevant salary examples were retrieved from the current salary corpus.",
        }

    if all(_is_unpaid(example["salary"]) for example in salary_examples[:3]):
        return {
            "prediction": "Likely unpaid or stipend not clearly specified",
            "confidence": "medium",
            "summary": (
                "The closest salary examples were all unpaid or stipend-free roles, "
                "so the safest estimate is that this posting may not include a fixed stipend."
            ),
        }

    numeric_examples = []

    for example in salary_examples:
        minimum = _coerce_positive_number(example.get("salary_monthly_min"))
        maximum = _coerce_positive_number(example.get("salary_monthly_max"))
        if minimum is None or maximum is None:
            continue

        similarity_weight = max(float(example.get("similarity", 0.0)), 0.05)
        report_weight = _coerce_positive_number(
            example.get("salary_report_count"),
            default=1.0,
        )
        weight = similarity_weight * min(report_weight, 8.0)

        midpoint = (minimum + maximum) / 2.0
        numeric_examples.append(
            {
                "minimum": minimum,
                "maximum": maximum,
                "midpoint": midpoint,
                "weight": weight,
            }
        )

    if not numeric_examples:
        return {
            "prediction": "Salary varies in retrieved examples",
            "confidence": "low",
            "summary": (
                "Salary evidence was retrieved, but it was not numeric enough to turn into a monthly range automatically."
            ),
        }

    minima = [example["minimum"] for example in numeric_examples]
    maxima = [example["maximum"] for example in numeric_examples]
    midpoints = [example["midpoint"] for example in numeric_examples]
    weights = [example["weight"] for example in numeric_examples]

    midpoint = _weighted_quantile(midpoints, weights, 0.5)

    if len(numeric_examples) >= 4 and midpoint > 0:
        filtered = [
            example
            for example in numeric_examples
            if midpoint / 2.5 <= example["midpoint"] <= midpoint * 2.5
        ]
        if len(filtered) >= 2:
            minima = [example["minimum"] for example in filtered]
            maxima = [example["maximum"] for example in filtered]
            midpoints = [example["midpoint"] for example in filtered]
            weights = [example["weight"] for example in filtered]
            midpoint = _weighted_quantile(midpoints, weights, 0.5)

    if len(midpoints) >= 3:
        low = _weighted_quantile(minima, weights, 0.25)
        high = _weighted_quantile(maxima, weights, 0.75)
    else:
        low = min(minima)
        high = max(maxima)

    if low >= high:
        padding = max(midpoint * 0.12, 1500.0)
        range_low = max(0.0, midpoint - padding / 2.0)
        range_high = midpoint + padding / 2.0
    else:
        padding = max((high - low) * 0.08, midpoint * 0.05)
        range_low = max(0.0, low - padding / 2.0)
        range_high = high + padding / 2.0

    description_note = (
        "Scraped LinkedIn-like descriptions were used to enrich retrieval for this prediction."
        if description_examples
        else "This prediction relied mainly on title, location, and salary-role matching."
    )

    internship_note = (
        "Internship-specific salary rows were prioritized."
        if prefer_internship
        else "Both internship and full-time salary rows were considered."
    )

    return {
        "prediction": format_monthly_range(range_low, range_high),
        "confidence": "medium" if len(numeric_examples) >= 3 else "low",
        "summary": f"{description_note} {internship_note}",
    }


def generate_with_cerebras(
    job_title: str,
    job_location: str,
    job_description: str,
    salary_examples: list[dict[str, Any]],
    description_examples: list[dict[str, Any]],
    heuristic_prediction: str,
) -> tuple[dict[str, str] | None, str, str | None]:
    api_key = CEREBRAS_API_KEY
    if not api_key:
        return None, "not_configured", None

    if not salary_examples:
        return None, "not_applicable", None

    salary_context_lines = []
    for idx, example in enumerate(salary_examples, start=1):
        salary_context_lines.append(
            (
                f"Salary example {idx}: title={example['title']}; company={example['company']}; "
                f"location={example['location']}; salary={example['salary']}; "
                f"employment_status={example['employment_status']}; "
                f"role_category={example['role_category']}; similarity={example['similarity']}"
            )
        )

    description_context_lines = []
    for idx, example in enumerate(description_examples, start=1):
        description_context_lines.append(
            (
                f"Description example {idx}: title={example['title']}; "
                f"category={example['role_category']}; query={example['search_query']}; "
                f"similarity={example['similarity']}; snippet={example['description_excerpt']}"
            )
        )

    user_prompt = (
        "Predict a realistic monthly stipend / salary for the pasted LinkedIn role.\n"
        "Use the salary examples as the primary evidence and the scraped description examples "
        "as wording / role-shape support.\n"
        "If the evidence is weak, say so clearly.\n\n"
        f"Job title: {job_title or 'Not provided'}\n"
        f"Job location: {job_location or 'Not provided'}\n"
        f"Job description:\n{job_description}\n\n"
        f"Heuristic estimate: {heuristic_prediction}\n\n"
        "Retrieved salary examples:\n"
        + "\n".join(salary_context_lines)
        + "\n\nRetrieved scraped description examples:\n"
        + ("\n".join(description_context_lines) if description_context_lines else "None")
        + "\n\nReturn valid JSON with exactly these keys: prediction, confidence, summary"
    )


    last_error = "unknown_error"

    for model in _candidate_cerebras_models(CEREBRAS_MODEL):
        if Cerebras is not None:
            try:
                client = _get_cerebras_client(api_key)
                chat_completion = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    max_completion_tokens=300,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a salary prediction assistant. Ground your answer in the retrieved evidence "
                                "and return only the requested JSON object with prediction, confidence, and summary."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = chat_completion.choices[0].message.content
                parsed = _load_json_object(str(content))
            except Exception as exc:
                last_error = str(exc)
                log.warning(
                    "Cerebras SDK prediction attempt failed for model %s: %s",
                    model,
                    exc,
                )
            else:
                return (
                    {
                        "prediction": str(parsed.get("prediction", "")).strip(),
                        "confidence": str(parsed.get("confidence", "")).strip().lower()
                        or "medium",
                        "summary": str(parsed.get("summary", "")).strip(),
                    },
                    "used",
                    None,
                )

        payload = {
            "model": model,
            "temperature": 0.2,
            "max_completion_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "salary_prediction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "prediction": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "summary": {"type": "string"},
                        },
                        "required": ["prediction", "confidence", "summary"],
                        "additionalProperties": False,
                    },
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a salary prediction assistant. Ground your answer in the retrieved evidence "
                        "and return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        }

        req = request.Request(
            "https://api.cerebras.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=25) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            last_error = _read_http_error_body(exc)
            log.warning(
                "Cerebras prediction attempt failed for model %s: %s",
                model,
                last_error,
            )
            continue
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            log.warning(
                "Cerebras prediction attempt failed for model %s: %s",
                model,
                exc,
            )
            continue

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = _load_json_object(content)
        except Exception as exc:
            last_error = str(exc)
            log.warning(
                "Could not parse Cerebras salary response for model %s: %s",
                model,
                exc,
            )
            continue

        return (
            {
                "prediction": str(parsed.get("prediction", "")).strip(),
                "confidence": str(parsed.get("confidence", "")).strip().lower()
                or "medium",
                "summary": str(parsed.get("summary", "")).strip(),
            },
            "used",
            None,
        )

    log.warning("Cerebras prediction fallback activated after all attempts: %s", last_error)
    return None, "failed", last_error


def looks_like_internship(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INTERNSHIP_MARKERS)


def normalize_employment_mode(mode: str) -> str:
    if mode in {"intern", "full_time", "auto"}:
        return mode
    return "intern"


def normalize_employment_statuses(
    statuses: list[str] | None,
    employment_mode: str,
) -> list[str]:
    allowed_lookup = {status.lower(): status for status in KNOWN_EMPLOYMENT_STATUSES}
    normalized = []
    seen = set()

    for status in statuses or []:
        key = str(status).strip().lower()
        if key not in allowed_lookup or key in seen:
            continue
        normalized.append(allowed_lookup[key])
        seen.add(key)

    if normalized:
        return normalized

    if employment_mode == "full_time":
        return ["Full Time"]

    return ["Intern", "Trainee"]


def normalize_scraped_categories(categories: list[str] | None) -> list[str]:
    valid_categories = set(CATEGORY_LABELS.keys())

    normalized = []
    seen = set()

    for category in categories or []:
        key = str(category).strip()
        if key not in valid_categories or key in seen:
            continue
        normalized.append(key)
        seen.add(key)

    return normalized


def resolve_internship_preference(employment_mode: str, query_text: str) -> bool:
    if employment_mode == "intern":
        return True
    if employment_mode == "full_time":
        return False
    return looks_like_internship(query_text)


def parse_salary(raw_salary: str) -> dict[str, Any] | None:
    if not raw_salary:
        return None

    salary = raw_salary.strip().lower()
    if not salary or salary == "nan":
        return None

    if _is_unpaid(salary):
        return {"kind": "unpaid"}

    annual_multiplier = 1.0
    if "lpa" in salary or "lac" in salary or "lakh" in salary:
        annual_multiplier = 100000.0

    numbers = [
        float(num.replace(",", ""))
        for num in re.findall(r"\d[\d,]*\.?\d*", salary)
    ]
    if not numbers:
        return None

    if annual_multiplier > 1:
        numbers = [num * annual_multiplier for num in numbers]

    if _looks_annual(salary):
        monthly_values = [num / 12.0 for num in numbers]
    else:
        monthly_values = numbers

    minimum = min(monthly_values)
    maximum = max(monthly_values)
    return {
        "kind": "monthly_range",
        "minimum": minimum,
        "maximum": maximum,
    }


def format_monthly_range(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "Salary prediction unavailable"

    low_value = int(round(low / 500.0) * 500)
    high_value = int(round(high / 500.0) * 500)

    if abs(high_value - low_value) < 1000:
        return f"Rs {low_value:,} / month"

    return f"Rs {low_value:,} - Rs {high_value:,} / month"


def format_single_monthly(value: float) -> str:
    if pd.isna(value):
        return "Salary unavailable"

    rounded = int(round(value / 500.0) * 500)
    return f"Rs {rounded:,} / month"


def _is_unpaid(value: str) -> bool:
    text = value.lower()
    return any(marker in text for marker in UNPAID_MARKERS)


def _looks_annual(salary: str) -> bool:
    annual_markers = ("lpa", "per annum", "/year", "year", "annum", "pa")
    monthly_markers = ("/month", "month", "monthly")

    if any(marker in salary for marker in monthly_markers):
        return False

    return any(marker in salary for marker in annual_markers)


def _build_scraped_text(row: pd.Series) -> str:
    parts = [row.get(field, "") for field in SCRAPED_TEXT_FIELDS]
    return _join_text_parts(*parts)


def _build_salary_text(row: pd.Series) -> str:
    parts = [row.get(field, "") for field in SALARY_TEXT_FIELDS]
    return _join_text_parts(*parts)


def _join_text_parts(*parts: Any) -> str:
    clean_parts = []
    for value in parts:
        if value is None or pd.isna(value):
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() != "nan":
            clean_parts.append(cleaned)
    return "\n".join(clean_parts)


def _series_from_df(df: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default)

    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _normalize_lookup_text(text: Any) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize_text(text: Any) -> set[str]:
    normalized = _normalize_lookup_text(text)
    if not normalized:
        return set()
    return {token for token in normalized.split() if len(token) > 1}


def _token_overlap_score(query_tokens: set[str], candidate_text: Any) -> float:
    if not query_tokens:
        return 0.0

    candidate_tokens = _tokenize_text(candidate_text)
    if not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    return len(overlap) / len(query_tokens)


def _coerce_positive_number(value: Any, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if pd.isna(numeric):
        return default

    return numeric


def _weighted_quantile(
    values: list[float],
    weights: list[float],
    quantile: float,
) -> float:
    if not values:
        return 0.0

    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total_weight = sum(weight for _value, weight in pairs)
    if total_weight <= 0:
        return float(pairs[len(pairs) // 2][0])

    threshold = total_weight * min(max(quantile, 0.0), 1.0)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += max(weight, 0.0)
        if cumulative >= threshold:
            return float(value)

    return float(pairs[-1][0])


def _load_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@lru_cache(maxsize=1)
def _get_cerebras_client(api_key: str):
    if Cerebras is None:
        raise RuntimeError("cerebras_cloud_sdk is not installed")

    return Cerebras(
        api_key=api_key,
        warm_tcp_connection=False,
    )


def _candidate_cerebras_models(preferred_model: str) -> list[str]:
    candidates = [
        preferred_model,
        "qwen-3-235b-a22b-instruct-2507",
        "gpt-oss-120b",
        "llama3.1-8b",
    ]
    ordered: list[str] = []
    seen = set()

    for model in candidates:
        if not model or model in seen:
            continue
        ordered.append(model)
        seen.add(model)

    return ordered


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""

    if body:
        return f"{exc.code} {exc.reason}: {body}"

    return f"{exc.code} {exc.reason}"
