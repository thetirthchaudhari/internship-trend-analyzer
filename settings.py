"""
Project-wide settings.

Keep the main configuration for Flask, MongoDB, Cerebras, and dataset paths
in one place under the project root.
"""

from __future__ import annotations

import os
from pathlib import Path


def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"

APP_NAME = "Internship And Hiring Trend Analyzer"
SITE_OWNER_NAME = os.getenv("SITE_OWNER_NAME", "Tirth Chaudhari")
SITE_OWNER_PHONE = os.getenv("SITE_OWNER_PHONE", "+91 9173128308")
SITE_OWNER_EMAIL = os.getenv("SITE_OWNER_EMAIL", "tirth.chaudhari@icloud.com")
SITE_OWNER_LOCATION = os.getenv("SITE_OWNER_LOCATION", "India")

# Flask
LOCAL_FLASK_SECRET_KEY = "super-secret-key-change-me"
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or LOCAL_FLASK_SECRET_KEY
FLASK_DEBUG = _getenv_bool("FLASK_DEBUG", True)

# MongoDB
LOCAL_MONGO_URI = (
    "mongodb+srv://TirthAdmin:Ambe8308@ac-t4psehk.0j0lf4k.mongodb.net/"
)
MONGO_URI = os.getenv("MONGO_URI") or LOCAL_MONGO_URI
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "scrapped_jobs")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "internship_jobs")
MONGO_SERVER_SELECTION_TIMEOUT_MS = int(
    os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "3000")
)
CONTACT_REQUESTS_DATABASE_NAME = os.getenv(
    "CONTACT_REQUESTS_DATABASE_NAME",
    "contact_requests",
)
CONTACT_REQUESTS_COLLECTION_NAME = os.getenv(
    "CONTACT_REQUESTS_COLLECTION_NAME",
    "messages",
)

SOURCE_NAME = "linkedin"

# Cerebras
CEREBRAS_API_KEY = (os.getenv("CEREBRAS_API_KEY") or "").strip() or None
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

# Data sources
KAGGLE_SALARY_DATA_PATH = str(DATASETS_DIR / "Salary_Dataset_with_Extra_Features.csv")
SCRAPED_DATA_PATHS = (
    str(DATA_DIR / "raw_jobs.csv"),
    str(DATA_DIR / "naukri_raw_jobs.csv"),
    str(DATA_DIR / "linkedin_raw_jobs.csv"),
)

# Shared UI labels / options
CATEGORY_LABELS = {
    "ai_ml": "AI / ML",
    "data": "Data & Analytics",
    "backend": "Backend Engineering",
    "frontend": "Frontend Engineering",
    "fullstack": "Full-Stack Engineering",
    "software": "Software Engineering",
    "cloud_devops": "Cloud & DevOps",
    "mlops": "MLOps & Platforms",
    "mobile": "Mobile",
    "security": "Security",
    "qa": "QA & SDET",
    "analytics": "Analytics",
    "research": "Research",
    "general_tech": "General Tech",
}

DEFAULT_SCRAPE_QUERY_LIBRARY = {
    "ai_ml": {
        "intern": ["machine learning intern"],
        "full_time": ["machine learning engineer"],
    },
    "data": {
        "intern": ["data science intern"],
        "full_time": ["data engineer"],
    },
    "backend": {
        "intern": ["backend developer intern"],
        "full_time": ["backend developer"],
    },
    "frontend": {
        "intern": ["frontend developer intern"],
        "full_time": ["frontend developer"],
    },
    "fullstack": {
        "intern": ["full stack developer intern"],
        "full_time": ["full stack developer"],
    },
    "software": {
        "intern": ["software engineer intern"],
        "full_time": ["software engineer"],
    },
    "cloud_devops": {
        "intern": ["devops intern"],
        "full_time": ["devops engineer"],
    },
    "mlops": {
        "intern": ["mlops intern"],
        "full_time": ["mlops engineer"],
    },
    "mobile": {
        "intern": ["mobile developer intern"],
        "full_time": ["android developer"],
    },
    "security": {
        "intern": ["cybersecurity intern"],
        "full_time": ["security engineer"],
    },
    "qa": {
        "intern": ["qa intern"],
        "full_time": ["qa engineer"],
    },
    "analytics": {
        "intern": ["analytics intern"],
        "full_time": ["analytics engineer"],
    },
    "research": {
        "intern": ["research intern ai"],
        "full_time": ["research engineer"],
    },
    "general_tech": {
        "intern": ["computer science intern"],
        "full_time": ["associate software engineer"],
    },
}


SCRAPE_QUERY_LIBRARY = {
    "ai_ml": {
        "intern": [
            "machine learning intern",
            "ai intern",
            "nlp intern",
            "computer vision intern",
        ],
        "full_time": [
            "machine learning engineer",
            "ai engineer",
            "nlp engineer",
            "computer vision engineer",
        ],
    },
    "data": {
        "intern": [
            "data science intern",
            "data analyst intern",
            "data engineer intern",
            "business intelligence intern",
        ],
        "full_time": [
            "data engineer",
            "data analyst",
            "data scientist",
            "analytics engineer",
        ],
    },
    "backend": {
        "intern": [
            "backend developer intern",
            "python developer intern",
            "java developer intern",
            "api developer intern",
        ],
        "full_time": [
            "backend developer",
            "python developer",
            "java developer",
            "api engineer",
        ],
    },
    "frontend": {
        "intern": [
            "frontend developer intern",
            "react developer intern",
            "ui developer intern",
            "web developer intern",
        ],
        "full_time": [
            "frontend developer",
            "react developer",
            "ui engineer",
            "web developer",
        ],
    },
    "fullstack": {
        "intern": [
            "full stack developer intern",
            "mern stack intern",
            "mean stack intern",
            "full stack engineer intern",
        ],
        "full_time": [
            "full stack developer",
            "full stack engineer",
            "mern stack developer",
            "mean stack developer",
        ],
    },
    "software": {
        "intern": [
            "software engineer intern",
            "software developer intern",
            "application developer intern",
            "systems engineer intern",
        ],
        "full_time": [
            "software engineer",
            "software developer",
            "application developer",
            "systems engineer",
        ],
    },
    "cloud_devops": {
        "intern": [
            "devops intern",
            "cloud intern",
            "site reliability intern",
            "platform engineering intern",
        ],
        "full_time": [
            "devops engineer",
            "cloud engineer",
            "site reliability engineer",
            "platform engineer",
        ],
    },
    "mlops": {
        "intern": [
            "mlops intern",
            "machine learning platform intern",
            "data platform intern",
            "ml infrastructure intern",
        ],
        "full_time": [
            "mlops engineer",
            "machine learning platform engineer",
            "data platform engineer",
            "ml infrastructure engineer",
        ],
    },
    "mobile": {
        "intern": [
            "mobile developer intern",
            "android developer intern",
            "ios developer intern",
            "flutter developer intern",
        ],
        "full_time": [
            "android developer",
            "ios developer",
            "flutter developer",
            "react native developer",
        ],
    },
    "security": {
        "intern": [
            "cybersecurity intern",
            "security analyst intern",
            "application security intern",
            "cloud security intern",
        ],
        "full_time": [
            "security engineer",
            "security analyst",
            "application security engineer",
            "cloud security engineer",
        ],
    },
    "qa": {
        "intern": [
            "qa intern",
            "sdet intern",
            "test automation intern",
            "quality engineer intern",
        ],
        "full_time": [
            "qa engineer",
            "sdet",
            "test automation engineer",
            "quality engineer",
        ],
    },
    "analytics": {
        "intern": [
            "analytics intern",
            "business intelligence intern",
            "reporting analyst intern",
            "sql analyst intern",
        ],
        "full_time": [
            "analytics engineer",
            "business intelligence analyst",
            "reporting analyst",
            "sql developer",
        ],
    },
    "research": {
        "intern": [
            "research intern ai",
            "research intern machine learning",
            "applied scientist intern",
            "ai research intern",
        ],
        "full_time": [
            "research engineer",
            "applied scientist",
            "ai researcher",
            "machine learning researcher",
        ],
    },
    "general_tech": {
        "intern": [
            "computer science intern",
            "software intern",
            "technology intern",
            "it intern",
        ],
        "full_time": [
            "associate software engineer",
            "graduate software engineer",
            "junior software engineer",
            "entry level software engineer",
        ],
    },
}


def build_scrape_taxonomy(
    employment_mode: str | None = None,
    query_library: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, list[str]]:
    modes = (
        (employment_mode,)
        if employment_mode in {"intern", "full_time"}
        else ("intern", "full_time")
    )
    taxonomy: dict[str, list[str]] = {}
    library = query_library if query_library is not None else SCRAPE_QUERY_LIBRARY

    for category, mode_map in library.items():
        queries: list[str] = []
        seen: set[str] = set()

        for mode in modes:
            for query in mode_map.get(mode, []):
                normalized = query.strip().lower()
                if not normalized or normalized in seen:
                    continue
                queries.append(query)
                seen.add(normalized)

        if queries:
            taxonomy[category] = queries

    return taxonomy


def infer_employment_status_from_query(query: str | None) -> str:
    normalized = f" {(query or '').strip().lower()} "

    if " trainee" in normalized:
        return "Trainee"
    if " intern" in normalized or " internship" in normalized:
        return "Intern"
    return "Full Time"

LINKEDIN_FIELD_OPTIONS = [
    ("description", "Description"),
    ("job_url", "Job URL"),
    ("search_query", "Search Query"),
    ("role_category", "Role Category"),
    ("employment_status", "Employment Status"),
]

NAUKRI_FIELD_OPTIONS = [
    ("description", "Description"),
    ("job_url", "Job URL"),
    ("search_query", "Search Query"),
    ("role_category", "Role Category"),
    ("employment_status", "Employment Status"),
    ("salary", "Salary"),
    ("duration", "Duration"),
]

PREDICTION_STATUS_OPTIONS = [
    ("Intern", "Intern"),
    ("Trainee", "Trainee"),
    ("Full Time", "Full-Time"),
    ("Contractor", "Contractor"),
]

DEFAULT_LINKEDIN_SEARCH_URL = (
    "https://www.linkedin.com/jobs/search/?keywords=machine%20learning%20intern&f_E=1"
)
STREAMLIT_HEADLESS_DEFAULT = False
