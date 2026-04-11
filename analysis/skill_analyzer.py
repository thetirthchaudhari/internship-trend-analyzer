"""
analysis/skill_analyzer.py
===========================
NLP Analysis Module for the Internship & Hiring Trend Analyzer.

What this module does:
  1. Skill Frequency Analysis   - counts how often predefined tech skills
                                  appear across all job postings
  2. TF-IDF Keyword Extraction  - discovers statistically important terms
                                  in job descriptions automatically
  3. Per-Skill TF-IDF Scoring   - ranks predefined skills by TF-IDF weight
  4. Co-occurrence Analysis     - finds which skills appear together most
  5. Skill Category Breakdown   - groups skills by domain for trend view
  6. Skill by Job Level         - breaks down skills by intern/junior/senior
  7. Skill-Based Job Search     - mini search engine: find & rank jobs by
                                  user-supplied skills (NEW)
  8. Summary                    - single dict with all results for app.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Skills taxonomy
# ---------------------------------------------------------------------------

SKILLS_TAXONOMY = {
    "Programming Languages": [
        "python", "r", "java", "scala", "c++", "javascript",
        "typescript", "go", "rust", "matlab",
    ],
    "ML / AI Frameworks": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "keras", "scikit-learn", "xgboost", "lightgbm", "catboost",
        "hugging face", "llm", "generative ai", "reinforcement learning",
    ],
    "NLP": [
        "nlp", "natural language processing", "text mining",
        "sentiment analysis", "transformers", "bert", "gpt",
        "word2vec", "named entity recognition",
    ],
    "Data & Analytics": [
        "data analysis", "data science", "statistics", "sql",
        "pandas", "numpy", "data visualization", "tableau",
        "power bi", "excel", "spark", "hadoop", "etl",
        "data engineering", "feature engineering",
    ],
    "Cloud / MLOps": [
        "aws", "azure", "gcp", "docker", "kubernetes",
        "mlops", "git", "linux", "airflow", "mlflow",
        "ci/cd", "rest api", "fastapi", "flask",
    ],
    "Computer Vision": [
        "computer vision", "opencv", "image processing",
        "object detection", "cnn", "yolo",
    ],
}

ALL_SKILLS = [skill for skills in SKILLS_TAXONOMY.values() for skill in skills]


# ---------------------------------------------------------------------------
# 1. Skill Frequency Analysis
# ---------------------------------------------------------------------------

def count_skills(df: pd.DataFrame) -> pd.Series:
    """
    Count how many job postings mention each skill from ALL_SKILLS.

    Searches title + description. Each skill counted at most once per job.

    Returns:
        pd.Series: Skill -> count, sorted descending.
    """
    if df.empty or "title" not in df.columns:
        log.warning("count_skills: empty DataFrame or missing 'title' column")
        return pd.Series(dtype=int)

    combined = df["title"].fillna("") + " " + df["description"].fillna("")
    skill_counts = {}

    for skill in ALL_SKILLS:
        count = combined.str.contains(skill, case=False, na=False, regex=False).sum()
        if count > 0:
            skill_counts[skill] = int(count)

    result = pd.Series(skill_counts).sort_values(ascending=False)
    log.info("Skill frequency analysis: %d skills found across %d postings", len(result), len(df))
    return result


# ---------------------------------------------------------------------------
# 2. TF-IDF Keyword Extraction
# ---------------------------------------------------------------------------

def extract_tfidf_keywords(df: pd.DataFrame, top_n: int = 20) -> list:
    """
    Extract the most statistically distinctive keywords from job descriptions
    using TF-IDF vectorization.

    Returns:
        list[tuple]: [(keyword, tfidf_score), ...] sorted by score descending.
    """
    if "description" not in df.columns:
        log.warning("extract_tfidf_keywords: 'description' column not found")
        return []

    descriptions = df["description"].fillna("").tolist()
    non_empty    = [d for d in descriptions if d.strip()]

    if len(non_empty) < 2:
        log.warning("extract_tfidf_keywords: need at least 2 non-empty descriptions, got %d", len(non_empty))
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=300,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#\-\.]{1,}\b",
    )

    try:
        tfidf_matrix   = vectorizer.fit_transform(non_empty)
        scores         = tfidf_matrix.sum(axis=0).A1
        feature_names  = vectorizer.get_feature_names_out()
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

        log.info("TF-IDF keyword extraction: top %d keywords from %d descriptions", top_n, len(non_empty))
        return keyword_scores[:top_n]

    except Exception as e:
        log.error("TF-IDF keyword extraction failed: %s", e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# 3. Per-Skill TF-IDF Scoring
# ---------------------------------------------------------------------------

def score_skills_by_tfidf(df: pd.DataFrame) -> pd.Series:
    """
    Score predefined skills by their aggregate TF-IDF weight.

    Returns:
        pd.Series: Skill -> TF-IDF score, sorted descending.
    """
    if "description" not in df.columns or df.empty:
        log.warning("score_skills_by_tfidf: empty DataFrame or missing 'description' column")
        return pd.Series(dtype=float)

    descriptions = df["description"].fillna("").tolist()
    non_empty    = [d for d in descriptions if d.strip()]

    if len(non_empty) < 2:
        log.warning("score_skills_by_tfidf: insufficient descriptions (%d)", len(non_empty))
        return pd.Series(dtype=float)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#\-\.]{1,}\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(non_empty)
        vocab        = vectorizer.vocabulary_
        skill_scores = {}

        for skill in ALL_SKILLS:
            if skill in vocab:
                col_idx = vocab[skill]
                score   = float(tfidf_matrix[:, col_idx].sum())
                if score > 0:
                    skill_scores[skill] = round(score, 4)

        result = pd.Series(skill_scores).sort_values(ascending=False)
        log.info("TF-IDF skill scoring complete: %d skills scored", len(result))
        return result

    except Exception as e:
        log.error("TF-IDF skill scoring failed: %s", e, exc_info=True)
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 4. Co-occurrence Analysis
# ---------------------------------------------------------------------------

def compute_skill_cooccurrence(df: pd.DataFrame, top_skills: int = 15) -> pd.DataFrame:
    """
    Build a co-occurrence matrix showing which skills appear together.

    Returns:
        pd.DataFrame: Square matrix of co-occurrence counts.
    """
    if df.empty:
        log.warning("compute_skill_cooccurrence: empty DataFrame")
        return pd.DataFrame()

    combined    = df["title"].fillna("") + " " + df["description"].fillna("")
    freq_counts = {}

    for skill in ALL_SKILLS:
        count = combined.str.contains(skill, case=False, na=False, regex=False).sum()
        if count > 0:
            freq_counts[skill] = count

    if len(freq_counts) < 2:
        log.warning("compute_skill_cooccurrence: fewer than 2 skills found, cannot build matrix")
        return pd.DataFrame()

    top      = sorted(freq_counts, key=freq_counts.get, reverse=True)[:top_skills]
    presence = {
        skill: combined.str.contains(skill, case=False, na=False, regex=False).astype(int)
        for skill in top
    }

    presence_df = pd.DataFrame(presence)
    cooccur     = presence_df.T.dot(presence_df)
    np.fill_diagonal(cooccur.values, 0)

    log.info("Co-occurrence matrix built: %dx%d skills", len(top), len(top))
    return cooccur


# ---------------------------------------------------------------------------
# 5. Skill Category Breakdown
# ---------------------------------------------------------------------------

def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate skill demand by category (ML/AI, Cloud, NLP, etc.)

    Returns:
        pd.DataFrame: Columns: category, total_mentions, skill_count, top_skill
    """
    if df.empty:
        log.warning("get_category_breakdown: empty DataFrame")
        return pd.DataFrame()

    combined = df["title"].fillna("") + " " + df["description"].fillna("")
    rows     = []

    for category, skills in SKILLS_TAXONOMY.items():
        total_mentions = 0
        found_skills   = {}

        for skill in skills:
            count = combined.str.contains(skill, case=False, na=False, regex=False).sum()
            if count > 0:
                found_skills[skill] = int(count)
                total_mentions     += int(count)

        top_skill = max(found_skills, key=found_skills.get) if found_skills else "N/A"
        rows.append({
            "category":       category,
            "total_mentions": total_mentions,
            "skill_count":    len(found_skills),
            "top_skill":      top_skill,
        })

    result = pd.DataFrame(rows).sort_values("total_mentions", ascending=False).reset_index(drop=True)
    log.info("Category breakdown complete: %d categories analyzed", len(result))
    return result


# ---------------------------------------------------------------------------
# 6. Skill by Job Level
# ---------------------------------------------------------------------------

def get_skill_by_job_level(df: pd.DataFrame) -> dict:
    """
    Break down skill demand by job level: intern, junior, senior.

    Returns:
        dict: {level: pd.Series of top skills}
    """
    if df.empty or "title" not in df.columns:
        log.warning("get_skill_by_job_level: empty DataFrame or missing 'title' column")
        return {}

    level_keywords = {
        "intern": ["intern", "trainee", "fresher", "graduate"],
        "junior": ["junior", "associate", "entry", "jr"],
        "senior": ["senior", "lead", "sr", "principal", "staff"],
    }

    results = {}
    for level, keywords in level_keywords.items():
        pattern = "|".join(keywords)
        mask    = df["title"].str.contains(pattern, case=False, na=False)
        subset  = df[mask]

        if len(subset) >= 3:
            results[level] = count_skills(subset).head(5)
            log.debug("Job level '%s': %d postings found", level, len(subset))
        else:
            results[level] = pd.Series(dtype=int)
            log.debug("Job level '%s': insufficient postings (%d), skipping", level, len(subset))

    return results


# ---------------------------------------------------------------------------
# 7. Skill-Based Job Search  (NEW)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Lowercase, replace punctuation with spaces, collapse whitespace.

    Used on both the search corpus and user-supplied skill strings so that
    matching is always on the same normalized surface form.

    Examples:
        "Machine Learning!"  -> "machine learning"
        "C++ / TensorFlow"   -> "c   tensorflow"   (+ collapsed to space)
        "  NLP  "            -> "nlp"
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text.strip()


def _build_search_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Build one normalized text string per job row for skill matching.

    Combines title + description for each job so that skills mentioned only
    in the title (common for short Naukri cards with sparse descriptions) are
    still matched.

    Reuses already-cleaned text where possible:
        data_cleaner.clean_data() lowercases all text columns. We detect
        this by sampling the description column — if >80% of characters are
        already lowercase we skip re-normalization of the description,
        only normalizing the title and concatenating. This avoids redundant
        re.sub work on large already-cleaned datasets.

    Returns:
        pd.Series of normalized strings, index-aligned with df.
    """
    desc  = df["description"].fillna("") if "description" in df.columns else pd.Series("", index=df.index)
    title = df["title"].fillna("")        if "title"       in df.columns else pd.Series("", index=df.index)

    # Detect whether descriptions are already lowercase-cleaned
    sample = desc[desc.str.len() > 20].head(50)
    if not sample.empty:
        lc_ratio      = sample.apply(
            lambda t: sum(1 for c in t if c.islower()) / max(len(t), 1)
        ).mean()
        already_clean = lc_ratio > 0.80
    else:
        already_clean = False

    if already_clean:
        # Descriptions already clean — only normalize title before concat
        corpus = title.apply(_normalize_text) + " " + desc.astype(str)
    else:
        # Raw mixed-case text — normalize everything
        corpus = (title.astype(str) + " " + desc.astype(str)).apply(_normalize_text)

    return corpus


def search_jobs_by_skills(df: pd.DataFrame, skills: list) -> pd.DataFrame:
    """
    Hybrid skill-based job recommender.

    Ranking combines:
      1. Exact skill overlap
      2. TF-IDF cosine similarity between user query and job text

    Returns:
        pd.DataFrame with:
            match_score
            matched_skills
            skill_overlap_score
            tfidf_similarity
            final_score
    """
    if df.empty or not skills:
        log.warning("search_jobs_by_skills: empty DataFrame or empty skills list")
        return pd.DataFrame()

    # Step 1 — normalize user skills
    normalized_skills = [_normalize_text(s) for s in skills if str(s).strip()]
    normalized_skills = [s for s in normalized_skills if s]

    if not normalized_skills:
        log.warning("search_jobs_by_skills: all skills were empty after normalization")
        return pd.DataFrame()

    log.info(
        "search_jobs_by_skills: searching %d jobs for skills: %s",
        len(df), normalized_skills,
    )

    # Step 2 — build search corpus
    corpus = _build_search_corpus(df)

    # Add extra signal columns if available
    role_category = (
        df["role_category"].fillna("").astype(str).apply(_normalize_text)
        if "role_category" in df.columns else pd.Series("", index=df.index)
    )
    search_query_col = (
        df["search_query"].fillna("").astype(str).apply(_normalize_text)
        if "search_query" in df.columns else pd.Series("", index=df.index)
    )

    enriched_corpus = (
        corpus.fillna("") + " " +
        role_category.fillna("") + " " +
        search_query_col.fillna("")
    ).str.strip()

    # Step 3 — exact skill overlap
    skill_hits = {
        skill: enriched_corpus.str.contains(skill, regex=False, na=False)
        for skill in normalized_skills
    }

    hits_df = pd.DataFrame(skill_hits, index=df.index)
    match_score = hits_df.sum(axis=1)

    matched_mask = match_score >= 1
    if not matched_mask.any():
        log.info("search_jobs_by_skills: no matches found for skills: %s", normalized_skills)
        return pd.DataFrame()

    result = df[matched_mask].copy()
    matched_hits = hits_df[matched_mask]

    result["match_score"] = match_score[matched_mask].astype(int)
    result["matched_skills"] = matched_hits.apply(
        lambda row: ", ".join(sk for sk in normalized_skills if row[sk]),
        axis=1,
    )

    # Step 4 — normalize overlap score to 0..1
    max_possible = max(len(normalized_skills), 1)
    result["skill_overlap_score"] = result["match_score"] / max_possible

    # Step 5 — TF-IDF similarity
    query_text = " ".join(normalized_skills)

    try:
        tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#\-\.]{1,}\b",
        )

        matched_corpus = enriched_corpus[matched_mask].fillna("").tolist()
        documents = matched_corpus + [query_text]

        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        job_vectors = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]

        similarity_scores = (job_vectors @ query_vector.T).toarray().ravel()

        result["tfidf_similarity"] = similarity_scores

    except Exception as e:
        log.error("TF-IDF similarity scoring failed: %s", e, exc_info=True)
        result["tfidf_similarity"] = 0.0

    # Step 6 — hybrid score
    result["final_score"] = (
        0.35 * result["skill_overlap_score"] +
        0.65 * result["tfidf_similarity"]
    )

    # Step 7 — rank
    result = result.sort_values(
        by=["final_score", "match_score", "title"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    log.info(
        "search_jobs_by_skills: %d/%d jobs matched | top final_score=%.4f",
        len(result), len(df), result["final_score"].iloc[0],
    )

    return result


# ---------------------------------------------------------------------------
# 8. Master summary
# ---------------------------------------------------------------------------

def get_analysis_summary(df: pd.DataFrame, top_n_keywords: int = 20) -> dict:
    """
    Run the complete NLP analysis pipeline and return all results in one dict.

    Args:
        df: Cleaned job listings DataFrame.
        top_n_keywords: Number of TF-IDF keywords to extract.

    Returns:
        dict with keys: skill_counts, tfidf_skill_scores, keywords,
        cooccurrence, category_breakdown, skill_by_level,
        total_jobs, top_skill, top_location
    """
    log.info("NLP analysis pipeline started: %d job postings", len(df))

    skill_counts       = count_skills(df)
    tfidf_skill_scores = score_skills_by_tfidf(df)
    keywords           = extract_tfidf_keywords(df, top_n=top_n_keywords)
    cooccurrence       = compute_skill_cooccurrence(df, top_skills=12)
    category_breakdown = get_category_breakdown(df)
    skill_by_level     = get_skill_by_job_level(df)

    top_skill    = skill_counts.index[0] if not skill_counts.empty else "N/A"
    top_location = "N/A"

    if "location" in df.columns:
        loc_counts = df["location"].value_counts()
        if not loc_counts.empty:
            top_location = loc_counts.index[0]

    log.info(
        "NLP analysis complete - top skill: %s | top location: %s",
        top_skill, top_location,
    )

    return {
        "skill_counts":       skill_counts,
        "tfidf_skill_scores": tfidf_skill_scores,
        "keywords":           keywords,
        "cooccurrence":       cooccurrence,
        "category_breakdown": category_breakdown,
        "skill_by_level":     skill_by_level,
        "total_jobs":         len(df),
        "top_skill":          top_skill,
        "top_location":       top_location,
    }