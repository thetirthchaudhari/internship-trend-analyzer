import os
import re
import sys
from types import SimpleNamespace
from datetime import datetime

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

# ---------------------------------------------------------------------
# Path setup so we can import from project root (database, scraper, etc.)
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from settings import (
    APP_NAME,
    CATEGORY_LABELS,
    FLASK_DEBUG,
    FLASK_SECRET_KEY,
    LINKEDIN_FIELD_OPTIONS,
    NAUKRI_FIELD_OPTIONS,
    PREDICTION_STATUS_OPTIONS,
    SITE_OWNER_EMAIL,
    SITE_OWNER_LOCATION,
    SITE_OWNER_NAME,
    SITE_OWNER_PHONE,
)

# Now these imports should work:
from database.mongo_client import (
    get_collection,
    get_contact_requests_collection,
    insert_contact_request,
    load_jobs_to_dataframe,
)

from scraper.linkedin_scraper import scrape_and_store as scrape_linkedin
from scraper.naukri_scraper import scrape_and_store as scrape_naukri

# Optional: skill search (if the module exists)
try:
    import analysis.skill_analyzer as skill_analyzer
except ImportError:
    skill_analyzer = None

try:
    import rag.salary_predictor as salary_predictor
except ImportError:
    salary_predictor = None

# Optional: import search taxonomies from scrapers if available
try:
    from scraper.linkedin_scraper import SEARCH_TAXONOMY as LINKEDIN_SEARCH_TAXONOMY
except ImportError:
    LINKEDIN_SEARCH_TAXONOMY = {}

try:
    from scraper.naukri_scraper import SEARCH_TAXONOMY as NAUKRI_SEARCH_TAXONOMY
except ImportError:
    NAUKRI_SEARCH_TAXONOMY = {}

# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

CONTACT_FORM_FIELDS = (
    "name",
    "email",
    "phone",
    "company",
    "subject",
    "message",
)


def _phone_href(phone_number: str) -> str:
    digits = "".join(ch for ch in phone_number if ch.isdigit())
    if not digits:
        return ""
    return f"+{digits}"


@app.context_processor
def inject_global_template_vars():
    return {
        "app_name": APP_NAME,
        "current_year": datetime.now().year,
        "flask_debug": FLASK_DEBUG,
        "site_owner_name": SITE_OWNER_NAME,
        "site_owner_email": SITE_OWNER_EMAIL,
        "site_owner_phone": SITE_OWNER_PHONE,
        "site_owner_phone_href": _phone_href(SITE_OWNER_PHONE),
        "site_owner_location": SITE_OWNER_LOCATION,
    }


def get_dashboard_data():
    """
    Load jobs from Mongo and compute stats for the dashboard:
    - total_jobs
    - by_source
    - by_category
    - last_updated
    Also returns the full DataFrame for downstream use.
    """
    col = get_collection()
    df = load_jobs_to_dataframe(col)

    stats = {
        "total_jobs": int(df.shape[0]),
        "by_source": {},
        "by_category": {},
        "last_updated": None,
    }

    if not df.empty:
        if "source" in df.columns:
            stats["by_source"] = (
                df["source"].fillna("unknown").value_counts().to_dict()
            )
        if "role_category" in df.columns:
            stats["by_category"] = (
                df["role_category"].fillna("general_tech").value_counts().to_dict()
            )
        if "scraped_at" in df.columns:
            last = df["scraped_at"].max()
            if pd.notna(last):
                if hasattr(last, "to_pydatetime"):
                    last = last.to_pydatetime()
                stats["last_updated"] = last

    return stats, df


def build_latest_jobs(df: pd.DataFrame, limit: int = 10):
    """
    Convert latest N rows of df into simple objects so Jinja
    can use job.title / job.company nicely.
    """
    if df.empty:
        return []

    if "scraped_at" in df.columns:
        df = df.sort_values("scraped_at", ascending=False)

    df = df.head(limit)

    jobs = []
    for _, row in df.iterrows():
        jobs.append(
            SimpleNamespace(
                title=row.get("title", ""),
                company=row.get("company", ""),
                location=row.get("location", ""),
                source=row.get("source", ""),
                role_category=row.get("role_category", ""),
                employment_status=row.get("employment_status", ""),
                job_url=row.get("job_url", ""),
                scraped_at=row.get("scraped_at"),
            )
        )
    return jobs


def build_dashboard_context() -> dict:
    stats, df = get_dashboard_data()
    by_category = stats.get("by_category", {}) or {}

    return {
        "stats": stats,
        "top_categories": list(by_category.items())[:6],
        "latest_jobs": build_latest_jobs(df, limit=10),
        "CATEGORY_LABELS": CATEGORY_LABELS,
    }


def build_scrape_context() -> dict:
    stats, _ = get_dashboard_data()
    return {
        "stats": stats,
        "CATEGORY_LABELS": CATEGORY_LABELS,
        "linkedin_queries": LINKEDIN_TAX,
        "naukri_queries": NAUKRI_TAX,
        "linkedin_field_options": LINKEDIN_FIELD_OPTIONS,
        "naukri_field_options": NAUKRI_FIELD_OPTIONS,
    }


def _blank_contact_form() -> dict[str, str]:
    return {field: "" for field in CONTACT_FORM_FIELDS}


def _read_contact_form(form) -> dict[str, str]:
    return {
        field: (form.get(field) or "").strip()
        for field in CONTACT_FORM_FIELDS
    }


def _validate_contact_form(contact_form: dict[str, str]) -> list[str]:
    errors: list[str] = []

    if not contact_form["name"]:
        errors.append("Please enter your name.")
    if not contact_form["email"] or "@" not in contact_form["email"]:
        errors.append("Please enter a valid email address.")
    if not contact_form["message"]:
        errors.append("Please add a message before sending.")

    return errors


def _flatten_taxonomy(taxonomy: dict) -> dict:
    """
    Ensure taxonomy is always in the shape:
      { "category_key": ["query1", "query2", ...], ... }
    """
    if not isinstance(taxonomy, dict):
        return {}
    out = {}
    for cat, queries in taxonomy.items():
        if isinstance(queries, (list, tuple)):
            out[cat] = list(queries)
        else:
            out[cat] = [str(queries)]
    return out


def _parse_multiline_queries(raw: str | None) -> list[str] | None:
    if not raw:
        return None

    queries = [line.strip() for line in raw.splitlines() if line.strip()]
    return queries or None


def _get_selected_values(form, *field_names: str) -> list[str] | None:
    values = []
    seen = set()

    for field_name in field_names:
        for value in form.getlist(field_name):
            value = value.strip()
            if not value or value in seen:
                continue
            values.append(value)
            seen.add(value)

    return values or None


def _get_requested_fields(form, source_prefix: str) -> list[str] | None:
    marker = f"{source_prefix}_fields_present"
    field_name = f"{source_prefix}_fields"

    if not form.get(marker):
        return None

    return _get_selected_values(form, field_name) or []


def _default_prediction_statuses(employment_mode: str) -> list[str]:
    if employment_mode == "full_time":
        return ["Full Time"]
    return ["Intern", "Trainee"]


def _default_prediction_industries() -> list[str]:
    return list(CATEGORY_LABELS.keys())


def _parse_page_number(raw_value, default: int = 1) -> int:
    try:
        page = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(page, 1)


def _build_jobs_query(source: str, category: str, search: str) -> dict:
    query: dict = {}

    if source != "all":
        query["source"] = source

    if category != "all":
        query["role_category"] = category

    if search:
        pattern = re.compile(re.escape(search), re.IGNORECASE)
        query["$or"] = [
            {"title": pattern},
            {"company": pattern},
            {"description": pattern},
        ]

    return query


def _load_jobs_page(
    collection,
    source: str,
    category: str,
    search: str,
    page: int,
    page_size: int,
):
    query = _build_jobs_query(source=source, category=category, search=search)
    projection = {
        "_id": 0,
        "title": 1,
        "company": 1,
        "location": 1,
        "source": 1,
        "salary": 1,
        "Salary": 1,
        "role_category": 1,
        "employment_status": 1,
        "scraped_at": 1,
        "description": 1,
        "job_url": 1,
    }

    total = collection.count_documents(query)
    start = (page - 1) * page_size

    cursor = (
        collection.find(query, projection)
        .sort("scraped_at", -1)
        .skip(start)
        .limit(page_size)
    )
    jobs_list = list(cursor)

    return jobs_list, total


LINKEDIN_TAX = _flatten_taxonomy(LINKEDIN_SEARCH_TAXONOMY)
NAUKRI_TAX = _flatten_taxonomy(NAUKRI_SEARCH_TAXONOMY)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("index.html", **build_dashboard_context())


@app.route("/scrape")
def scrape_page():
    return render_template("scrape.html", **build_scrape_context())


@app.route("/about", methods=["GET", "POST"])
def about():
    contact_form = _blank_contact_form()

    if request.method == "POST":
        contact_form = _read_contact_form(request.form)
        validation_errors = _validate_contact_form(contact_form)

        if validation_errors:
            for error in validation_errors:
                flash(error, "warning")
        else:
            try:
                collection = get_contact_requests_collection()
                insert_contact_request(
                    collection,
                    {
                        **contact_form,
                        "ip_address": (
                            request.headers.get("X-Forwarded-For")
                            or request.remote_addr
                            or ""
                        ),
                        "user_agent": request.headers.get("User-Agent", ""),
                        "source": "about_page",
                    },
                )
                flash(
                    "Your message has been saved. Thanks for reaching out.",
                    "success",
                )
                return redirect(url_for("about"))
            except Exception as exc:
                flash(f"Could not save your message right now: {exc}", "error")

    return render_template("about.html", contact_form=contact_form)


@app.route("/jobs")
def jobs():
    col = get_collection()

    source = request.args.get("source", "all")
    category = request.args.get("category", "all")
    search = (request.args.get("q") or "").strip()
    page = _parse_page_number(request.args.get("page", 1))
    page_size = 20

    sources = sorted(
        str(value) for value in col.distinct("source") if value
    )
    categories = sorted(
        str(value) for value in col.distinct("role_category") if value
    )
    jobs_list, total = _load_jobs_page(
        collection=col,
        source=source,
        category=category,
        search=search,
        page=page,
        page_size=page_size,
    )

    return render_template(
        "jobs.html",
        jobs=jobs_list,
        sources=sources,
        categories=categories,
        source=source,
        category=category,
        search=search,
        page=page,
        page_size=page_size,
        total=total,
    )


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Skill-based search page.
    """
    results = []
    skills_input = ""
    message = None

    if request.method == "POST":
        skills_input = (request.form.get("skills") or "").strip()

        if not skills_input:
            message = "Please enter at least one skill."
        elif skill_analyzer is None:
            message = "Skill-based search is not configured yet."
        else:
            skills = [s.strip() for s in skills_input.split(",") if s.strip()]

            col = get_collection()
            df = load_jobs_to_dataframe(col)

            results_df = skill_analyzer.search_jobs_by_skills(df, skills)

            if not results_df.empty:
                results = results_df.to_dict(orient="records")
            else:
                results = []

    return render_template(
        "search.html",
        skills_input=skills_input,
        results=results,
        message=message,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """
    Salary prediction page powered by retrieval-backed examples.
    """
    title_input = ""
    location_input = ""
    description_input = ""
    employment_mode = "intern"
    salary_examples_count = 5
    selected_status_filters = _default_prediction_statuses(employment_mode)
    selected_scraped_categories = _default_prediction_industries()
    result = None
    message = None

    if request.method == "POST":
        title_input = (request.form.get("job_title") or "").strip()
        location_input = (request.form.get("job_location") or "").strip()
        description_input = (request.form.get("job_description") or "").strip()
        employment_mode = (
            "full_time" if request.form.get("full_time_mode") else "intern"
        )
        salary_examples_count = max(
            3,
            min(10, int(request.form.get("salary_examples_count") or 5)),
        )
        selected_status_filters = (
            _get_selected_values(request.form, "salary_statuses")
            or _default_prediction_statuses(employment_mode)
        )
        selected_scraped_categories = (
            _get_selected_values(request.form, "scraped_categories")
            or _default_prediction_industries()
        )

        if not description_input:
            message = "Paste a LinkedIn job description to run the salary prediction."
        elif salary_predictor is None:
            message = "Salary prediction is not configured yet."
        else:
            try:
                result = salary_predictor.predict_salary(
                    job_title=title_input,
                    job_location=location_input,
                    job_description=description_input,
                    employment_mode=employment_mode,
                    allowed_employment_statuses=selected_status_filters,
                    allowed_scraped_categories=selected_scraped_categories,
                    top_k_salary=salary_examples_count,
                )
            except Exception as exc:
                message = f"Prediction failed: {exc}"

    return render_template(
        "prediction.html",
        title_input=title_input,
        location_input=location_input,
        description_input=description_input,
        employment_mode=employment_mode,
        salary_examples_count=salary_examples_count,
        prediction_status_options=PREDICTION_STATUS_OPTIONS,
        selected_status_filters=selected_status_filters,
        prediction_industry_options=list(CATEGORY_LABELS.items()),
        selected_scraped_categories=selected_scraped_categories,
        CATEGORY_LABELS=CATEGORY_LABELS,
        result=result,
        message=message,
    )

@app.route("/scrape/linkedin", methods=["POST"])
def trigger_linkedin_scrape():
    jobs_per_query = int(request.form.get("jobs_per_query") or 3)
    headless = not bool(request.form.get("headed"))
    li_at_cookie = (request.form.get("li_at_cookie") or "").strip() or None

    selected_categories = _get_selected_values(request.form, "linkedin_categories")
    selected_queries = _get_selected_values(
        request.form,
        "linkedin_queries",
        "queries",
    )

    custom_queries = _parse_multiline_queries(
        request.form.get("linkedin_custom_queries")
    )
    selected_fields = _get_requested_fields(request.form, "linkedin")

    try:
        result = scrape_linkedin(
            search_url=None,
            max_jobs=None,
            headless=headless,
            li_at_cookie=li_at_cookie,
            jobs_per_query=jobs_per_query,
            selected_categories=selected_categories,
            selected_queries=selected_queries,
            custom_queries=custom_queries,
            selected_fields=selected_fields,
        )
        inserted = result.get("inserted", 0)
        total = result.get("total", 0)
        flash(
            f"LinkedIn scrape complete: {inserted} new of {total} total rows.",
            "success",
        )
    except Exception as e:
        flash(f"LinkedIn scrape failed: {e}", "error")

    return redirect(url_for("scrape_page"))


@app.route("/scrape/naukri", methods=["POST"])
def trigger_naukri_scrape():
    jobs_per_query = int(request.form.get("jobs_per_query") or 3)
    headless = not bool(request.form.get("headed"))

    selected_categories = _get_selected_values(request.form, "naukri_categories")
    selected_queries = _get_selected_values(
        request.form,
        "naukri_queries",
        "queries",
    )

    custom_queries = _parse_multiline_queries(
        request.form.get("naukri_custom_queries")
    )
    selected_fields = _get_requested_fields(request.form, "naukri")

    try:
        result = scrape_naukri(
            headless=headless,
            jobs_per_query=jobs_per_query,
            selected_categories=selected_categories,
            selected_queries=selected_queries,
            custom_queries=custom_queries,
            selected_fields=selected_fields,
        )
        inserted = result.get("inserted", 0)
        total = result.get("total", 0)
        flash(
            f"Naukri scrape complete: {inserted} new of {total} total rows.",
            "success",
        )
    except Exception as e:
        flash(f"Naukri scrape failed: {e}", "error")

    return redirect(url_for("scrape_page"))

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG)
