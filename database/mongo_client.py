"""
database/mongo_client.py
========================
Handles all MongoDB operations for the Internship & Hiring Trend Analyzer.

Responsibilities:
  - Connect to MongoDB using pymongo
  - Insert scraped job documents into the collection
  - Prevent duplicate entries using a compound unique index
  - Load stored data back into a pandas DataFrame for analysis

Database schema:
  Database   : scrapped_jobs
  Collection : internship_jobs
"""

import hashlib
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure

from logger import get_logger
from settings import (
    CONTACT_REQUESTS_COLLECTION_NAME,
    CONTACT_REQUESTS_DATABASE_NAME,
    MONGO_COLLECTION_NAME,
    MONGO_DATABASE_NAME,
    MONGO_SERVER_SELECTION_TIMEOUT_MS,
    MONGO_URI,
    SOURCE_NAME,
)

log = get_logger(__name__)
DATABASE_NAME = MONGO_DATABASE_NAME
COLLECTION_NAME = MONGO_COLLECTION_NAME
CONTACT_DATABASE_NAME = CONTACT_REQUESTS_DATABASE_NAME
CONTACT_COLLECTION_NAME = CONTACT_REQUESTS_COLLECTION_NAME
OPTIONAL_JOB_FIELDS = (
    "description",
    "job_url",
    "search_query",
    "role_category",
    "employment_status",
    "salary",
    "duration",
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_client():
    """
    Create and cache the MongoDB client so page navigations can reuse the same
    connection pool instead of reconnecting on every request.
    """
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
    )
    client.admin.command("ping")
    return client


@lru_cache(maxsize=1)
def get_collection():
    """
    Connect to MongoDB and return the internship_jobs collection.

    Ensures a unique index exists on '_id_hash' so duplicate jobs are
    automatically rejected by MongoDB on insert.

    Returns:
        pymongo.collection.Collection

    Raises:
        ConnectionFailure: If MongoDB is not running or unreachable.
    """
    try:
        client = _get_client()
        db         = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        collection.create_index(
            [("_id_hash", ASCENDING)],
            unique=True,
            name="unique_job_hash"
        )
        collection.create_index(
            [("scraped_at", ASCENDING)],
            name="jobs_scraped_at"
        )
        collection.create_index(
            [("source", ASCENDING)],
            name="jobs_source"
        )
        collection.create_index(
            [("role_category", ASCENDING)],
            name="jobs_role_category"
        )
        log.info("Connected to MongoDB: %s.%s", DATABASE_NAME, COLLECTION_NAME)
        return collection

    except ConnectionFailure as e:
        log.critical("MongoDB connection failed: %s", e, exc_info=True)
        log.critical("Ensure MongoDB is running: brew services start mongodb-community")
        raise


@lru_cache(maxsize=1)
def get_contact_requests_collection():
    """
    Return the dedicated collection used for contact form submissions.

    Contact requests are stored in a separate Mongo database so they remain
    isolated from scraped job documents.
    """
    try:
        client = _get_client()
        db = client[CONTACT_DATABASE_NAME]
        collection = db[CONTACT_COLLECTION_NAME]
        collection.create_index(
            [("submitted_at", ASCENDING)],
            name="contact_requests_submitted_at",
        )
        collection.create_index(
            [("email", ASCENDING)],
            name="contact_requests_email",
        )
        log.info(
            "Connected to MongoDB: %s.%s",
            CONTACT_DATABASE_NAME,
            CONTACT_COLLECTION_NAME,
        )
        return collection
    except ConnectionFailure as e:
        log.critical("MongoDB connection failed: %s", e, exc_info=True)
        log.critical("Ensure MongoDB is running: brew services start mongodb-community")
        raise


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _make_hash(title: str, company: str, location: str) -> str:
    """MD5 hash of title + company + location for deduplication."""
    raw = f"{title.strip().lower()}|{company.strip().lower()}|{location.strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_job_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if pd.isna(value):
        return ""

    return value


def _clean_contact_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


# ---------------------------------------------------------------------------
# Insert operations
# ---------------------------------------------------------------------------

def insert_job(collection, job: dict) -> bool:
    """
    Insert a single job document into MongoDB.

    Adds core fields plus any optional scraped fields present in the payload.
    Silently skips if the job already exists.

    Returns:
        bool: True if inserted, False if duplicate.
    """
    document = {
        "title":       _clean_job_value(job.get("title", "")),
        "company":     _clean_job_value(job.get("company", "")),
        "location":    _clean_job_value(job.get("location", "")),
        "source":      _clean_job_value(job.get("source", SOURCE_NAME)) or SOURCE_NAME,
        "scraped_at":  job.get("scraped_at", datetime.now(timezone.utc)),
        "_id_hash":    _make_hash(
                           str(_clean_job_value(job.get("title", ""))),
                           str(_clean_job_value(job.get("company", ""))),
                           str(_clean_job_value(job.get("location", "")))
                       ),
    }

    for field in OPTIONAL_JOB_FIELDS:
        if field in job:
            document[field] = _clean_job_value(job.get(field))

    try:
        collection.insert_one(document)
        return True
    except DuplicateKeyError:
        return False


def insert_jobs_bulk(collection, jobs_df: pd.DataFrame) -> dict:
    """
    Insert all jobs from a DataFrame into MongoDB.

    Returns:
        dict: inserted, duplicates, total counts.
    """
    inserted   = 0
    duplicates = 0

    log.info("Starting bulk insert: %d records to process", len(jobs_df))

    for _, row in jobs_df.iterrows():
        success = insert_job(collection, row.to_dict())
        if success:
            inserted += 1
        else:
            duplicates += 1

    log.info(
        "Bulk insert complete - inserted: %d | duplicates: %d | total: %d",
        inserted, duplicates, len(jobs_df)
    )

    return {"inserted": inserted, "duplicates": duplicates, "total": len(jobs_df)}


def insert_contact_request(collection, contact_request: dict) -> dict:
    """
    Insert a single contact form submission into the dedicated contact database.
    """
    document = {
        "name": _clean_contact_value(contact_request.get("name")),
        "email": _clean_contact_value(contact_request.get("email")).lower(),
        "phone": _clean_contact_value(contact_request.get("phone")),
        "company": _clean_contact_value(contact_request.get("company")),
        "subject": _clean_contact_value(contact_request.get("subject")),
        "message": _clean_contact_value(contact_request.get("message")),
        "ip_address": _clean_contact_value(contact_request.get("ip_address")),
        "user_agent": _clean_contact_value(contact_request.get("user_agent")),
        "submitted_at": contact_request.get("submitted_at", datetime.now(timezone.utc)),
        "status": _clean_contact_value(contact_request.get("status")) or "new",
        "source": _clean_contact_value(contact_request.get("source")) or "about_page",
    }

    result = collection.insert_one(document)
    return {
        "inserted_id": str(result.inserted_id),
        "status": document["status"],
    }


# ---------------------------------------------------------------------------
# Load operations
# ---------------------------------------------------------------------------

def load_jobs_to_dataframe(collection, query: dict = None) -> pd.DataFrame:
    """
    Load job documents from MongoDB into a pandas DataFrame.

    Args:
        collection: pymongo Collection from get_collection().
        query: Optional MongoDB filter. Default None loads all documents.

    Returns:
        pd.DataFrame
    """
    query  = query or {}
    cursor = collection.find(query, {"_id": 0, "_id_hash": 0})
    docs   = list(cursor)

    if not docs:
        log.warning("No documents found in MongoDB matching query: %s", query)
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    log.info("Loaded %d jobs from MongoDB into DataFrame", len(df))
    return df


def get_collection_stats(collection) -> dict:
    """
    Return basic statistics about the stored job collection.

    Returns:
        dict: total_jobs, unique_companies, unique_locations, latest_scraped
    """
    total = collection.count_documents({})

    if total == 0:
        return {
            "total_jobs": 0, "unique_companies": 0,
            "unique_locations": 0, "latest_scraped": None,
        }

    unique_companies = len(collection.distinct("company"))
    unique_locations = len(collection.distinct("location"))
    latest_doc       = collection.find_one({}, {"scraped_at": 1}, sort=[("scraped_at", -1)])
    latest_scraped   = latest_doc.get("scraped_at") if latest_doc else None

    log.debug(
        "Collection stats - total: %d | companies: %d | locations: %d",
        total, unique_companies, unique_locations
    )

    return {
        "total_jobs":       total,
        "unique_companies": unique_companies,
        "unique_locations": unique_locations,
        "latest_scraped":   latest_scraped,
    }
