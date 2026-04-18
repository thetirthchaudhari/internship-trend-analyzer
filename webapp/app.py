import os
import re
import sys
import hashlib
from collections import Counter
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
    CEREBRAS_API_KEY,
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
from logger import get_logger

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
log = get_logger(__name__)


def _key_fingerprint(value: str | None) -> str:
    if not value:
        return "missing"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{digest[:8]}...{digest[-8:]}"


log.info(
    "Cerebras configured: %s | key fingerprint: %s | key length: %s",
    "yes" if CEREBRAS_API_KEY else "no",
    _key_fingerprint(CEREBRAS_API_KEY),
    len(CEREBRAS_API_KEY) if CEREBRAS_API_KEY else 0,
)

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

ROLE_TITLE_STOPWORDS = {
    "analyst",
    "application",
    "associate",
    "consultant",
    "developer",
    "engineer",
    "executive",
    "fresher",
    "graduate",
    "intern",
    "internship",
    "junior",
    "lead",
    "manager",
    "role",
    "senior",
    "software",
    "specialist",
    "staff",
    "team",
    "technical",
    "trainee",
}


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
    role_chart_items = _build_role_chart_items(stats)

    return {
        "stats": stats,
        "top_categories": list(by_category.items())[:6],
        "role_chart_items": role_chart_items,
        "role_chart_summary": _build_role_chart_summary(stats, role_chart_items),
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


def _format_role_label(role_key: str) -> str:
    role_key = (role_key or "").strip()
    if not role_key:
        return "General Tech"
    return CATEGORY_LABELS.get(role_key, role_key)


def _build_role_chart_items(stats: dict, limit: int = 6) -> list[dict]:
    by_category = stats.get("by_category", {}) or {}
    total_jobs = max(int(stats.get("total_jobs") or 0), 1)
    items: list[dict] = []

    sorted_items = sorted(
        by_category.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

    for role_key, count in sorted_items:
        share = (int(count) / total_jobs) * 100
        items.append(
            {
                "key": role_key,
                "label": _format_role_label(role_key),
                "count": int(count),
                "share": share,
                "share_display": f"{share:.1f}%",
                "fill_width": min(max(share, 10.0), 100.0),
            }
        )

    return items


def _build_role_chart_summary(stats: dict, role_chart_items: list[dict]) -> dict:
    if not role_chart_items:
        return {}

    dominant = role_chart_items[0]
    return {
        "dominant_label": dominant["label"],
        "dominant_share": dominant["share_display"],
        "dominant_count": dominant["count"],
        "tracked_families": len(stats.get("by_category", {}) or {}),
        "active_sources": len(stats.get("by_source", {}) or {}),
    }


def _split_skills_input(skills_input: str) -> list[str]:
    return [value.strip() for value in skills_input.split(",") if value.strip()]


def _ordered_unique_strings(values, limit: int = 3) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for raw_value in values:
        if pd.isna(raw_value):
            continue

        value = str(raw_value).strip()
        if not value or value.lower() == "nan" or value in seen:
            continue

        ordered.append(value)
        seen.add(value)

        if len(ordered) >= limit:
            break

    return ordered


def _top_ranked_matched_skills(skill_values, limit: int = 6) -> list[str]:
    skill_counts: Counter[str] = Counter()

    for raw_value in skill_values:
        if pd.isna(raw_value):
            continue

        for skill in str(raw_value).split(","):
            cleaned = skill.strip()
            if not cleaned:
                continue
            skill_counts[cleaned] += 1

    return [skill for skill, _count in skill_counts.most_common(limit)]


def _resolve_role_group_key(row: pd.Series) -> str:
    role_category = str(row.get("role_category") or "").strip()
    if role_category:
        return role_category

    title = str(row.get("title") or "").strip()
    return title or "General Tech"


def _extract_title_tokens(titles: list[str], limit: int = 5) -> list[str]:
    tokens: list[str] = []

    for title in titles:
        normalized = re.sub(r"[^a-z0-9+#/ ]+", " ", str(title).lower())
        for token in normalized.split():
            if len(token) < 4 or token in ROLE_TITLE_STOPWORDS or token in tokens:
                continue
            tokens.append(token)
            if len(tokens) >= limit:
                return tokens

    return tokens


def _load_role_salary_frame() -> pd.DataFrame:
    if salary_predictor is None:
        return pd.DataFrame()

    try:
        salary_jobs = salary_predictor.load_salary_jobs()
        if salary_jobs.empty:
            return pd.DataFrame()
        return salary_predictor.build_salary_corpus(salary_jobs)
    except Exception as exc:
        log.warning("Role-fit salary corpus unavailable: %s", exc)
        return pd.DataFrame()


def _format_salary_range(low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "Salary signal unavailable"

    if salary_predictor is not None and hasattr(salary_predictor, "format_monthly_range"):
        return salary_predictor.format_monthly_range(float(low), float(high))

    low_value = int(round(float(low) / 500.0) * 500)
    high_value = int(round(float(high) / 500.0) * 500)

    if abs(high_value - low_value) < 1000:
        return f"Rs {low_value:,} / month"

    return f"Rs {low_value:,} - Rs {high_value:,} / month"


def _estimate_role_salary_signal(
    role_key: str,
    sample_titles: list[str],
    salary_df: pd.DataFrame,
    prefer_internship: bool,
) -> dict:
    if salary_df.empty:
        return {
            "display": "Salary signal unavailable",
            "sample_size": 0,
            "note": "Add salary-bearing rows or use the salary predictor for a deeper estimate.",
        }

    candidates = salary_df.copy()

    if prefer_internship and "is_internship_like" in candidates.columns:
        internship_slice = candidates[candidates["is_internship_like"].fillna(False)].copy()
        if len(internship_slice) >= 8:
            candidates = internship_slice

    role_key_normalized = (role_key or "").strip().casefold()
    matched_frames: list[pd.DataFrame] = []

    if role_key_normalized and "role_category" in candidates.columns:
        category_match = candidates[
            candidates["role_category"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(role_key_normalized)
        ].copy()
        if not category_match.empty:
            matched_frames.append(category_match)

    title_tokens = _extract_title_tokens(sample_titles)
    if title_tokens and "title" in candidates.columns:
        title_series = candidates["title"].fillna("").astype(str).str.lower()
        token_mask = pd.Series(False, index=candidates.index)
        for token in title_tokens:
            token_mask = token_mask | title_series.str.contains(
                re.escape(token),
                regex=True,
                na=False,
            )
        title_match = candidates[token_mask].copy()
        if not title_match.empty:
            matched_frames.append(title_match)

    if matched_frames:
        matched = (
            pd.concat(matched_frames, ignore_index=True, sort=False)
            .drop_duplicates(subset=["title", "company", "location", "salary_value"])
            .reset_index(drop=True)
        )
    else:
        matched = pd.DataFrame()

    if matched.empty:
        return {
            "display": "Salary signal unavailable",
            "sample_size": 0,
            "note": "No close salary evidence was found for this role family yet.",
        }

    midpoint = pd.to_numeric(matched.get("salary_monthly_mid"), errors="coerce")
    midpoint = midpoint.dropna()

    if midpoint.empty:
        mins = pd.to_numeric(matched.get("salary_monthly_min"), errors="coerce")
        maxes = pd.to_numeric(matched.get("salary_monthly_max"), errors="coerce")
        midpoint = ((mins + maxes) / 2.0).dropna()

    if midpoint.empty:
        return {
            "display": "Salary signal unavailable",
            "sample_size": int(len(matched)),
            "note": "Salary rows exist, but they are not numeric enough to summarize here.",
        }

    low = float(midpoint.quantile(0.25))
    high = float(midpoint.quantile(0.75))
    if high < low:
        low, high = high, low

    scope = "internship-leaning" if prefer_internship else "mixed-market"
    return {
        "display": _format_salary_range(low, high),
        "sample_size": int(len(matched)),
        "note": f"Built from {len(matched)} {scope} salary records tied to the same role context.",
    }


def build_role_fit_context(skills: list[str], limit: int = 8) -> tuple[list[dict], dict]:
    if skill_analyzer is None:
        return [], {}

    collection = get_collection()
    df = load_jobs_to_dataframe(collection)
    results_df = skill_analyzer.search_jobs_by_skills(df, skills)

    if results_df.empty:
        return [], {
            "searched_skills": skills,
            "matched_jobs": 0,
            "salary_supported_roles": 0,
        }

    results_df = results_df.copy()
    results_df["role_group_key"] = results_df.apply(_resolve_role_group_key, axis=1)
    salary_df = _load_role_salary_frame()

    recommendations: list[dict] = []

    for role_key, group in results_df.groupby("role_group_key", dropna=False):
        group = group.sort_values(
            by=["final_score", "match_score", "title"],
            ascending=[False, False, True],
        )

        top_titles = _ordered_unique_strings(group.get("title", []), limit=3)
        top_locations = _ordered_unique_strings(group.get("location", []), limit=3)
        top_companies = _ordered_unique_strings(group.get("company", []), limit=3)
        matched_skills = _top_ranked_matched_skills(group.get("matched_skills", []), limit=6)

        internship_signal = (
            group.get("employment_status", pd.Series("", index=group.index))
            .fillna("")
            .astype(str)
            .str.contains("intern|trainee", case=False, na=False)
        )
        title_signal = (
            group.get("title", pd.Series("", index=group.index))
            .fillna("")
            .astype(str)
            .str.contains("intern|trainee", case=False, na=False)
        )
        prefer_internship = bool((internship_signal | title_signal).mean() >= 0.5)
        fit_score = float(group["final_score"].head(min(3, len(group))).mean())
        salary_signal = _estimate_role_salary_signal(
            role_key=role_key,
            sample_titles=top_titles,
            salary_df=salary_df,
            prefer_internship=prefer_internship,
        )

        recommendations.append(
            {
                "role_key": role_key,
                "role_label": _format_role_label(str(role_key)),
                "fit_score": fit_score,
                "fit_score_display": f"{round(fit_score * 100)}%",
                "matching_jobs": int(len(group)),
                "top_titles": top_titles,
                "top_locations": top_locations,
                "top_companies": top_companies,
                "matched_skills": matched_skills,
                "salary_display": salary_signal["display"],
                "salary_note": salary_signal["note"],
                "salary_sample_size": salary_signal["sample_size"],
                "employment_hint": "Internship-leaning" if prefer_internship else "Mixed market",
                "top_overlap": int(group["match_score"].max()) if "match_score" in group else 0,
            }
        )

    recommendations.sort(
        key=lambda item: (item["fit_score"], item["matching_jobs"], item["salary_sample_size"]),
        reverse=True,
    )
    recommendations = recommendations[:limit]

    salary_supported_roles = sum(1 for item in recommendations if item["salary_sample_size"] > 0)
    summary = {
        "searched_skills": skills,
        "matched_jobs": int(len(results_df)),
        "recommended_roles": len(recommendations),
        "salary_supported_roles": salary_supported_roles,
    }
    if recommendations:
        summary["top_role_label"] = recommendations[0]["role_label"]

    return recommendations, summary


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


@app.route("/role-fit", methods=["GET", "POST"])
def role_fit():
    skills_input = ""
    searched_skills: list[str] = []
    recommendations: list[dict] = []
    summary: dict = {}
    message = None

    if request.method == "POST":
        skills_input = (request.form.get("skills") or "").strip()

        if not skills_input:
            message = "Please enter at least one skill."
        elif skill_analyzer is None:
            message = "Role-fit search is not configured yet."
        else:
            searched_skills = _split_skills_input(skills_input)
            recommendations, summary = build_role_fit_context(searched_skills)

            if not recommendations:
                message = "No role families matched those skills yet."

    return render_template(
        "role_fit.html",
        skills_input=skills_input,
        searched_skills=searched_skills,
        recommendations=recommendations,
        summary=summary,
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
