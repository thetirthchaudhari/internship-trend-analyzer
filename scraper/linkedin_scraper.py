"""
scraper/linkedin_scraper.py
============================
LinkedIn Job Scraper — URL Navigation
======================================

Strategy:
  - Navigate directly to LinkedIn jobs search URLs for each query
  - Use &start= parameter for pagination
  - Parse job cards from the results list
  - For each card, open the job detail page to get the full description
  - Slow random pacing throughout to avoid detection

Schema stored per job:
  title, company, location, description, job_url,
  source, search_query, role_category, scraped_at
"""

import re
import time
import random
import os
import hashlib
from datetime import datetime, timezone

import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)

from logger import get_logger
from settings import (
    DEFAULT_SCRAPE_QUERY_LIBRARY,
    build_scrape_taxonomy,
    infer_employment_status_from_query,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JOBS_PER_QUERY      = 5      # per query — 5 x 27 = ~135 total
PAGE_SIZE           = 25     # LinkedIn results per page
MAX_PAGES_PER_QUERY = 3      # max result pages per query

JOB_DELAY_MIN       = 4.0    # seconds between individual job fetches
JOB_DELAY_MAX       = 9.0

QUERY_PAUSE_MIN     = 12.0   # seconds between queries (same category)
QUERY_PAUSE_MAX     = 25.0

CATEGORY_PAUSE_MIN  = 30.0   # seconds between category groups
CATEGORY_PAUSE_MAX  = 60.0

LINKEDIN_JOBS_BASE  = "https://www.linkedin.com/jobs/search/"


# ---------------------------------------------------------------------------
# Search taxonomy
# ---------------------------------------------------------------------------

SEARCH_TAXONOMY = build_scrape_taxonomy()
DEFAULT_SEARCH_TAXONOMY = build_scrape_taxonomy(
    query_library=DEFAULT_SCRAPE_QUERY_LIBRARY,
)

QUERY_TO_CATEGORY = {
    query: category
    for category, queries in SEARCH_TAXONOMY.items()
    for query in queries
}

CORE_JOB_FIELDS = (
    "title",
    "company",
    "location",
    "source",
    "scraped_at",
)

DEFAULT_OPTIONAL_FIELDS = (
    "description",
    "job_url",
    "search_query",
    "role_category",
    "employment_status",
)


def build_query_list(
    selected_categories=None,
    selected_queries=None,
    custom_queries=None,
):
    """
    Build list of (query, category) pairs based on:
      - selected_categories: list of keys from SEARCH_TAXONOMY
      - selected_queries: exact preset queries chosen from the UI
      - custom_queries: list of plain strings, category='custom'
    If all are None/empty, falls back to ALL_QUERIES (the default).
    """
    base = []
    seen = set()

    def append_query(query, category):
        query = (query or "").strip()
        if not query:
            return

        dedupe_key = query.lower()
        if dedupe_key in seen:
            return

        base.append((query, category))
        seen.add(dedupe_key)

    if selected_categories:
        for cat, queries in SEARCH_TAXONOMY.items():
            if cat in selected_categories:
                for q in queries:
                    append_query(q, cat)

    if selected_queries:
        for q in selected_queries:
            append_query(q, QUERY_TO_CATEGORY.get(q, "custom"))

    if not base and not custom_queries:
        return list(ALL_QUERIES)

    if custom_queries:
        for q in custom_queries:
            append_query(q, "custom")

    return base


ALL_QUERIES   = [(q, cat) for cat, qs in DEFAULT_SEARCH_TAXONOMY.items() for q in qs]
TOTAL_QUERIES = len(ALL_QUERIES)


def limit_output_fields(
    df: pd.DataFrame,
    selected_fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep core fields plus the requested optional fields.
    If selected_fields is None, preserve the default LinkedIn schema.
    """
    if df.empty:
        return df

    optional_fields = (
        list(DEFAULT_OPTIONAL_FIELDS)
        if selected_fields is None
        else list(selected_fields)
    )

    keep_fields = []
    for field in (*CORE_JOB_FIELDS, *optional_fields):
        if field in df.columns and field not in keep_fields:
            keep_fields.append(field)

    if not keep_fields:
        return df

    return df.loc[:, keep_fields]


# ---------------------------------------------------------------------------
# Rotating user agents
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
]


# ---------------------------------------------------------------------------
# XPath selectors — derived from live LinkedIn Nile UI HTML
# Stable anchors: id=, data-occludable-job-id, data-job-id,
# ---------------------------------------------------------------------------

# Top-level results container on jobs search page
RESULTS_CONTAINER_XPATHS = [
    "//main//section[contains(@class,'two-pane-serp-page__results')]",
    "//section[contains(@class,'two-pane-serp-page__results-list')]",
    "//div[contains(@class,'jobs-search-results-list')]",
]

# Individual job cards within results
CARD_XPATHS = [
    "//li[contains(@class,'jobs-search-results__list-item')]",
    "//div[contains(@data-occludable-job-id,'')]",
    "//li[contains(@data-occludable-job-id,'')]",
]

# Job title link on card
TITLE_XPATHS = [
    ".//a[contains(@class,'job-card-list__title')]",
    ".//a[contains(@class,'job-card-container__link')]",
    ".//a[contains(@data-tracking-control-name,'public_jobs_jserp-result_search-card')]",
]

# Company name text on card
COMPANY_XPATHS = [
    ".//*[contains(@class,'job-card-container__primary-description')]",
    ".//*[contains(@class,'job-card-list__company-name')]",
    ".//*[contains(@class,'job-card-container__company-name')]",
]

# Location text on card
LOCATION_XPATHS = [
    ".//*[contains(@class,'job-card-container__metadata-item')]",
    ".//*[contains(@class,'job-card-container__metadata-wrapper')]//*[contains(@class,'job-card-container__metadata-item')]",
]

# Right side detail panel — header and description
DETAIL_PANEL_XPATHS = [
    "//div[contains(@class,'jobs-search__job-details')]",
    "//section[contains(@class,'two-pane-serp-page__detail-view')]",
]

DESC_XPATHS = [
    "//div[contains(@class,'jobs-description-content__text')]",
    "//div[contains(@class,'jobs-box__html-content')]",
    "//section[contains(@class,'show-more-less-html')]",
]

LOGIN_BLOCK_XPATHS = [
    "//div[contains(@class,'authwall-join-form')]",
    "//div[contains(@class,'join-form__form-body')]",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def human_delay(mn: float = 1.5, mx: float = 4.0):
    time.sleep(random.uniform(mn, mx))


def slow_scroll(driver, pixels: int = 400):
    current = driver.execute_script("return window.pageYOffset")
    target  = current + pixels
    step    = random.randint(60, 120)
    while current < target:
        current = min(current + step, target)
        driver.execute_script(f"window.scrollTo(0, {current});")
        time.sleep(random.uniform(0.04, 0.10))


def simulate_mouse(driver):
    try:
        els = driver.find_elements(By.TAG_NAME, "a")
        if els:
            el = random.choice(els[:15])
            ActionChains(driver).move_to_element(el).perform()
            time.sleep(random.uniform(0.3, 0.9))
    except Exception:
        pass


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_multiline_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def dedup_hash(title: str, company: str, location: str) -> str:
    def n(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return re.sub(r"\s+", " ", s)
    return hashlib.md5(f"{n(title)}|{n(company)}|{n(location)}".encode()).hexdigest()


def build_search_url(query: str, start: int = 0) -> str:
    """
    Build a LinkedIn jobs search URL for a given query and start offset.
    Internship queries keep the internship experience filter.
    Full-time queries are left broad so we can scrape regular software roles too.
    """
    employment_status = infer_employment_status_from_query(query)
    params = {
        "keywords": query,
        "location": "India",
        "position": 1,
        "pageNum": 0,
    }
    if employment_status == "Intern":
        params["f_E"] = 1

    # Encode minimal query string
    from urllib.parse import urlencode
    base = LINKEDIN_JOBS_BASE
    qs   = urlencode(params)
    url  = f"{base}?{qs}"
    if start > 0:
        url += f"&start={start}"
    return url


# ---------------------------------------------------------------------------
# LinkedInScraper class
# ---------------------------------------------------------------------------

class LinkedInScraper:

    def __init__(self, li_at_cookie: str | None = None, headless: bool = False):
        self.li_at_cookie      = li_at_cookie
        self.headless          = headless
        self.user_agent        = random.choice(USER_AGENTS)
        self.driver            = None
        self._session_hashes   = set()
        self._logged_in        = False
        self._setup_driver()

    # --- Driver setup and login ---

    def _setup_driver(self):
        profile_dir = os.path.join(os.path.expanduser("~"), "chrome_linkedin_profile")
        os.makedirs(profile_dir, exist_ok=True)

        opts = uc.ChromeOptions()
        opts.add_argument(f"--user-data-dir={profile_dir}")
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={self.user_agent}")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1440,900")
        opts.add_argument("--lang=en-US,en;q=0.9")
        opts.add_argument("--disable-blink-features=AutomationControlled")

        self.driver = uc.Chrome(options=opts, use_subprocess=True, version_main=145)
        self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins',   {get: () => [1, 2, 3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = { runtime: {} };
        """)
        log.info("LinkedIn browser ready | UA: %s...", self.user_agent[:55])




    def _apply_cookie_if_needed(self):
        if not self.li_at_cookie:
            return
        try:
            self.driver.get("https://www.linkedin.com")
            human_delay(2, 4)
            self.driver.add_cookie({
                "name":  "li_at",
                "value": self.li_at_cookie,
                "domain": ".linkedin.com",
                "path": "/",
                "secure": True,
                "httpOnly": True,
            })
            self.driver.get("https://www.linkedin.com/feed/")
            human_delay(2, 4)
        except Exception as e:
            log.warning("Failed to apply li_at cookie: %s", e)

    def _check_login_state(self):
        try:
            self.driver.get("https://www.linkedin.com/jobs/")
            human_delay(3, 5)
            for xpath in LOGIN_BLOCK_XPATHS:
                try:
                    if self.driver.find_elements(By.XPATH, xpath):
                        log.warning("LinkedIn appears to show an auth wall / login block.")
                        return False
                except Exception:
                    continue
            return True
        except Exception as e:
            log.warning("Login state check failed: %s", e)
            return False

    def _login(self):
        """
        Best-effort login:
          - if li_at cookie provided → apply and hope it's valid
          - otherwise rely on saved Chrome profile
        """
        self._apply_cookie_if_needed()
        logged_in = self._check_login_state()
        if logged_in:
            self._logged_in = True
            log.info("LinkedIn session appears active.")
        else:
            log.warning(
                "LinkedIn session may not be fully authenticated. "
                "Scraper will still attempt public / limited results."
            )

    # --- Detail description fetch ---

    def _extract_description_from_element(self, el) -> str:
        """
        Try multiple JS properties to get the fullest description text.
        """
        chunks = []

        for js in (
            "return arguments[0].innerText || '';",
            "return arguments[0].textContent || '';",
        ):
            try:
                raw = self.driver.execute_script(js, el)
                text = clean_multiline_text(raw)
                if len(text) > len(" ".join(chunks)):
                    chunks = [text]
            except Exception:
                continue

        if not chunks:
            try:
                txt = clean_multiline_text(el.text)
                if txt:
                    chunks = [txt]
            except Exception:
                pass

        return max(chunks, key=len) if chunks else ""

    def _fetch_description(self, job_url: str) -> str:
        """
        Opens job_url in a new tab or via existing panel to grab full description.
        For LinkedIn jobs search, most descriptions appear in the right-side pane,
        so this function can be extended to rely on the pane if needed.
        """
        if not job_url:
            return ""

        desc_texts = []

        try:
            # Try opening in a new tab first
            self.driver.execute_script("window.open(arguments[0], '_blank');", job_url)
            self.driver.switch_to.window(self.driver.window_handles[-1])

            human_delay(2, 4)

            for xpath in DESC_XPATHS:
                try:
                    els = self.driver.find_elements(By.XPATH, xpath)
                    for el in els:
                        txt = self._extract_description_from_element(el)
                        if len(txt) > 50:
                            desc_texts.append(txt)
                except Exception:
                    continue

        except Exception as e:
            log.debug("Detail fetch in new tab failed: %s", e)

        finally:
            # Close tab and return to main
            try:
                if len(self.driver.window_handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
                    human_delay(1.0, 2.0)
            except Exception:
                pass

        if desc_texts:
            best = max(desc_texts, key=len)
            log.debug("Description length: %d chars", len(best))
            return best

        log.debug("No description found for %s", job_url)
        return ""

    # --- Card parsing ---

    def _parse_card(self, card) -> dict | None:
        try:
            title_el = None
            title    = ""
            job_url  = ""

            for xpath in TITLE_XPATHS:
                try:
                    el = card.find_element(By.XPATH, xpath)
                    t  = el.text.strip()
                    href = el.get_attribute("href") or ""
                    if t and href:
                        title_el = el
                        title    = clean_text(t)
                        job_url  = href.split("?")[0]
                        break
                except Exception:
                    continue

            if not title:
                return None

            company = ""
            for xpath in COMPANY_XPATHS:
                try:
                    el = card.find_element(By.XPATH, xpath)
                    company = clean_text(el.text)
                    if company:
                        break
                except Exception:
                    continue

            location = ""
            for xpath in LOCATION_XPATHS:
                try:
                    el = card.find_element(By.XPATH, xpath)
                    location = clean_text(el.text)
                    if location:
                        break
                except Exception:
                    continue

            return {
                "title":    title,
                "company":  company,
                "location": location,
                "job_url":  job_url,
            }

        except StaleElementReferenceException:
            log.debug("Stale card element, skipping")
            return None
        except Exception as e:
            log.debug("Card parse error: %s", e)
            return None

    # --- Scrape one query ---

    def scrape_query(self, query: str, category: str, max_jobs: int) -> list:
        log.info("Query: '%s' | category: %s | target: %d", query, category, max_jobs)

        jobs       = []
        page_num   = 0
        dupes      = 0

        while len(jobs) < max_jobs and page_num < MAX_PAGES_PER_QUERY:
            start = page_num * PAGE_SIZE
            url   = build_search_url(query, start=start)
            log.info("Page %d | %s", page_num + 1, url)

            self.driver.get(url)
            human_delay(4, 7)

            # Scroll to trigger lazy loading
            for _ in range(3):
                slow_scroll(self.driver, pixels=random.randint(400, 700))
                human_delay(1.5, 3.0)

            if random.random() < 0.25:
                simulate_mouse(self.driver)

            # Find cards
            cards = []
            for xpath in CARD_XPATHS:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    cards = self.driver.find_elements(By.XPATH, xpath)
                    if cards:
                        log.debug("%d cards found via %s", len(cards), xpath[:50])
                        break
                except TimeoutException:
                    continue

            if not cards:
                log.warning("No cards on page %d, stopping query", page_num + 1)
                break

            # Parse ALL cards into plain dicts FIRST (titles, URLs, IDs).
            # We extract all metadata upfront so card elements won't go stale.
            parsed_cards = []
            for card in cards:
                parsed = self._parse_card(card)
                if parsed:
                    parsed_cards.append(parsed)
            log.debug("%d cards parsed on page %d", len(parsed_cards), page_num + 1)

            new_this_page = 0
            for parsed in parsed_cards:
                if len(jobs) >= max_jobs:
                    break

                h = dedup_hash(parsed["title"], parsed["company"], parsed["location"])
                if h in self._session_hashes:
                    dupes += 1
                    log.debug("Duplicate: %s @ %s", parsed["title"], parsed["company"])
                    continue

                # Fetch full description from job URL
                human_delay(JOB_DELAY_MIN, JOB_DELAY_MAX)
                description = self._fetch_description(parsed["job_url"])

                self._session_hashes.add(h)
                job = {
                    "title":         parsed["title"],
                    "company":       parsed["company"],
                    "location":      parsed["location"],
                    "description":   description,
                    "job_url":       parsed["job_url"],
                    "source":        "linkedin",
                    "search_query":  query,
                    "role_category": category,
                    "employment_status": infer_employment_status_from_query(query),
                    "scraped_at":    datetime.now(timezone.utc),
                }
                jobs.append(job)
                new_this_page += 1
                log.info(
                    "[%d/%d] %s @ %s | %s",
                    len(jobs), max_jobs, job["title"], job["company"], job["location"]
                )

            log.info("Page %d: +%d new | %d dupes total", page_num + 1, new_this_page, dupes)

            if new_this_page == 0:
                log.info("No new jobs on page %d, stopping", page_num + 1)
                break

            page_num += 1
            human_delay(3, 6)

        log.info("Query complete: %d collected | %d dupes skipped", len(jobs), dupes)
        return jobs

    # --- Run all queries ---

    def scrape_all_queries(
        self,
        jobs_per_query: int = JOBS_PER_QUERY,
        selected_categories: list[str] | None = None,
        selected_queries: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run scraping across multiple queries/categories."""
        self._login()

        query_list = build_query_list(
            selected_categories=selected_categories,
            selected_queries=selected_queries,
            custom_queries=custom_queries,
        )
        total_queries = len(query_list)

        all_jobs = []
        log.info(
            "Scrape starting | queries: %d | per query: %d | estimated total: %d",
            total_queries, jobs_per_query, total_queries * jobs_per_query,
        )

        for idx, (query, category) in enumerate(query_list, start=1):
            log.info(
                "--- [%d/%d] '%s' | %s ---",
                idx, total_queries, query, category.upper(),
            )
            try:
                jobs = self.scrape_query(query, category, max_jobs=jobs_per_query)
                all_jobs.extend(jobs)
                log.info("Running total: %d", len(all_jobs))
            except Exception as e:
                log.error("Query '%s' failed: %s", query, e, exc_info=True)
                continue

            # polite pause between queries
            if idx < total_queries:
                pause = random.uniform(QUERY_PAUSE_MIN, QUERY_PAUSE_MAX)
                log.debug("Pausing %.0fs before next query", pause)
                time.sleep(pause)

        try:
            self.driver.quit()
        except Exception:
            pass

        df = pd.DataFrame(all_jobs)
        log.info("LinkedIn scrape complete: %d total jobs", len(df))

        if not df.empty and "role_category" in df.columns:
            for cat, cnt in df["role_category"].value_counts().items():
                log.info("  %-20s %d", cat, cnt)

        return df

    def quit(self):
        try:
            self.driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def save_to_csv(df: pd.DataFrame, filename: str = "linkedin_jobs.csv", output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    log.info("CSV saved: %s (%d rows)", filepath, len(df))
    return filepath


# ---------------------------------------------------------------------------
# app.py entry point
# ---------------------------------------------------------------------------

def scrape_and_store(
    search_url:        str  = None,
    max_jobs:          int  = None,
    headless:          bool = False,
    li_at_cookie:      str  = None,
    jobs_per_query:    int  = None,
    selected_categories: list[str] | None = None,
    selected_queries:    list[str] | None = None,
    custom_queries:      list[str] | None = None,
    selected_fields:     list[str] | None = None,
) -> dict:
    """
    Top-level entrypoint used by Flask app.
    Now supports:
      - selected_categories: list of category keys from SEARCH_TAXONOMY
      - selected_queries: exact preset queries chosen from the UI
      - custom_queries: extra queries (category='custom')
      - selected_fields: optional output columns to keep
    """
    from database.mongo_client import get_collection, insert_jobs_bulk

    per_query = jobs_per_query or max_jobs or JOBS_PER_QUERY
    scraper   = LinkedInScraper(li_at_cookie=li_at_cookie, headless=headless)
    df        = scraper.scrape_all_queries(
        jobs_per_query=per_query,
        selected_categories=selected_categories,
        selected_queries=selected_queries,
        custom_queries=custom_queries,
    )
    df = limit_output_fields(df, selected_fields=selected_fields)

    if df.empty:
        log.warning("No LinkedIn jobs scraped")
        return {"df": df, "inserted": 0, "duplicates": 0, "total": 0, "by_category": {}}

    save_to_csv(df, filename="linkedin_raw_jobs.csv", output_dir="data")

    try:
        collection = get_collection()
        stats      = insert_jobs_bulk(collection, df)
    except Exception as e:
        log.error("MongoDB insert failed: %s", e, exc_info=True)
        stats = {"inserted": 0, "duplicates": 0, "total": len(df)}

    by_category = (
        df["role_category"].value_counts().to_dict()
        if "role_category" in df.columns else {}
    )

    return {
        "df":          df,
        "inserted":    stats["inserted"],
        "duplicates":  stats["duplicates"],
        "total":       stats["total"],
        "by_category": by_category,
    }
