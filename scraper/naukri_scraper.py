"""
scraper/naukri_scraper.py
==========================
Naukri.com Internship Scraper — Homepage Search Bar Flow
=========================================================

Flow:
  1. Navigate to https://www.naukri.com/mnjuser/homepage
  2. Click the search bar to activate it
  3. Open the job-type dropdown → select "Internship"
  4. Type keyword into the keyword suggestor input → press Enter
  5. Wait for SRP (Search Results Page) to load
  6. Parse all job cards into plain dicts (stale-element safe)
  7. For each card: navigate to job URL, scrape full detail page
     — description, salary, duration, location from detail header
  8. driver.back() to SRP and continue
  9. Paginate via URL /{slug}-jobs-{page}?... pattern

Mongo schema (2 new fields added at the bottom):
  title, company, location, description, job_url,
  source, search_query, role_category, scraped_at,
  salary, duration                                  <- NEW

source is always "naukri"
"""

import re
import time
import random
import os
import hashlib
from datetime import datetime, timezone
import html
from urllib.parse import urlparse, parse_qs, unquote

import pandas as pd
# NOTE: we keep undetected_chromedriver import unused now, scraper runs via Safari
# import undetected_chromedriver as uc
from selenium import webdriver  # <-- Safari WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
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
# Configuration (these can be overridden from Streamlit)
# ---------------------------------------------------------------------------

JOBS_PER_QUERY      = 5
PAGE_SIZE           = 20
MAX_PAGES_PER_QUERY = 3

JOB_DELAY_MIN       = 3.0
JOB_DELAY_MAX       = 7.0

QUERY_PAUSE_MIN     = 10.0
QUERY_PAUSE_MAX     = 20.0

CATEGORY_PAUSE_MIN  = 25.0
CATEGORY_PAUSE_MAX  = 50.0

NAUKRI_BASE         = "https://www.naukri.com"
NAUKRI_HOMEPAGE     = "https://www.naukri.com/mnjuser/homepage"


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
    "salary",
    "duration",
)


def build_query_list(
    selected_categories=None,
    selected_queries=None,
    custom_queries=None,
):
    """
    Build list of (query, category) pairs for Naukri based on:
      - selected_categories: list of keys from SEARCH_TAXONOMY
      - selected_queries: exact preset queries chosen from the UI
      - custom_queries: list of strings, category='custom'
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
    If selected_fields is None, preserve the default Naukri schema.
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
# Rotating user agents (not applied to Safari, but kept for consistency)
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
]


# ---------------------------------------------------------------------------
# XPath selectors
# ---------------------------------------------------------------------------

# Search bar (homepage) -------------------------------------------------------

SEARCHBAR_WRAPPER_XPATHS = [
    "//div[contains(@class,'nI-gNb-sb__main')]",
    "//div[contains(@class,'nI-gNb-sb__full-view')]",
    "//span[contains(@class,'nI-gNb-sb__placeholder')]",
]

JOBTYPE_INPUT_XPATHS = [
    "//input[@id='jobType']",
    "//input[@name='jobType']",
    "//div[contains(@class,'nI-gNb-sb__expDD')]//input",
]

INTERNSHIP_OPTION_XPATHS = [
    "//li[@value='ainternship']",
    "//li[@title='Internship']",
    "//ul[contains(@class,'dropdown')]//li[normalize-space(text())='Internship']",
    "//ul[contains(@class,'dropdown')]//li[contains(.,'Internship')]",
]

KEYWORD_INPUT_XPATHS = [
    "//div[contains(@class,'nI-gNb-sb__keywords')]//input[contains(@class,'suggestor-input')]",
    "//input[@placeholder='Enter keyword / designation / companies']",
    "//div[contains(@class,'nI-gNb-sb__keywords')]//input",
]

SEARCH_BTN_XPATHS = [
    "//button[contains(@class,'nI-gNb-sb__icon-wrapper')]",
    "//button[.//span[contains(@class,'ni-gnb-icn-search')]]",
]

# SRP job cards ---------------------------------------------------------------

CARD_XPATHS = [
    "//div[contains(@class,'srp-jobtuple-wrapper')]",
    "//div[contains(@class,'cust-job-tuple')]",
    "//article[contains(@class,'jobTuple')]",
    "//div[contains(@class,'jobTuple') and @data-job-id]",
]

TITLE_XPATHS = [
    ".//h2//a[contains(@class,'title')]",
    ".//a[contains(@class,'title')]",
    ".//a[contains(@class,'job-title')]",
    ".//h2//a",
    ".//a[contains(@href,'naukri.com/job-listings')]",
]

COMPANY_XPATHS = [
    ".//*[contains(@class,'comp-name')]",
    ".//*[contains(@class,'companyName')]",
    ".//*[contains(@class,'company-name')]",
    ".//*[contains(@class,'org-name')]",
]

LOCATION_CARD_XPATHS = [
    ".//*[contains(@class,'locWdth')]",
    ".//*[@title and contains(@class,'loc')]",
    ".//*[contains(@class,'location')]",
    ".//*[contains(@class,'loc')]//span",
]

SALARY_CARD_XPATHS = [
    ".//*[contains(@class,'sal-wrap')]//span[not(contains(@class,'icon'))]",
    ".//*[contains(@class,'sal')]//span[@title]",
    ".//*[contains(@class,'salary')]",
]

DURATION_CARD_XPATHS = [
    ".//*[contains(@class,'exp-wrap')]//span[not(contains(@class,'icon'))]",
    ".//*[contains(@class,'duration')]//span",
    ".//*[contains(@class,'internship-duration')]",
    ".//*[@title and contains(@class,'exp')]",
]

# Job detail page -------------------------------------------------------------

DETAIL_HEADER_XPATHS = [
    "//section[contains(@class,'job-header-container')]",
    "//section[@id='job_header']",
    "//div[contains(@class,'jd-header')]",
]

LOCATION_DETAIL_XPATHS = [
    "//div[contains(@class,'jhc__loc')]//a",
    "//div[contains(@class,'jhc__loc')]//span",
    "//span[contains(@class,'jhc__location')]//a",
    "//i[contains(@class,'ni-icon-location')]/following-sibling::span//a",
    "//i[contains(@class,'ni-icon-location')]/following-sibling::span",
]

SALARY_DETAIL_XPATHS = [
    "//div[contains(@class,'jhc__salary')]//span[last()]",
    "//i[contains(@class,'ni-icon-salary')]/following-sibling::span",
    "//*[contains(@class,'salary')]//span[not(contains(@class,'icon'))]",
]

DURATION_DETAIL_XPATHS = [
    "//div[contains(@class,'jhc__exp')]//span[last()]",
    "//i[contains(@class,'ni-icon-clock')]/following-sibling::span",
    "//*[contains(@class,'internship-duration')]//span",
    "//*[contains(@class,'jhc__exp')]",
]

DESC_XPATHS = [
    "//div[contains(@class,'dang-inner-html')]",
    "//section[contains(@class,'job-desc')]",
    "//div[contains(@class,'styles_job-desc')]",
    "//div[contains(@class,'styles_JDC')]",
    "//div[@id='jobDescription']",
    "//div[contains(@class,'job-description')]",
]

DETAIL_READY_XPATHS = [
    "//section[contains(@class,'job-header-container')]",
    "//div[contains(@class,'dang-inner-html')]",
    "//div[contains(@class,'job-desc')]",
    "//div[@id='job_description']",
    "//h1[contains(@class,'jd-header-title')]",
]

OVERLAY_CLOSE_XPATHS = [
    "//button[contains(@class,'close-btn')]",
    "//button[@data-ga-track='Login_Nudge_Close']",
    "//span[contains(@class,'close-icon')]",
    "//*[@id='login_layer']//button[contains(@class,'close')]",
    "//div[contains(@class,'modal')]//button[contains(@class,'close')]",
    "//button[contains(@aria-label,'close')]",
    "//button[contains(@aria-label,'Close')]",
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
        time.sleep(random.uniform(0.04, 0.12))


def simulate_mouse(driver):
    try:
        links = driver.find_elements(By.TAG_NAME, "a")
        if links:
            ActionChains(driver).move_to_element(random.choice(links[:10])).perform()
            time.sleep(random.uniform(0.3, 0.8))
    except Exception:
        pass


def clean_multiline_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_query_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if qs.get("k"):
            return qs["k"][0].strip()
        slug = parsed.path.strip("/").split("/")[-1]
        slug = re.sub(r"-jobs-?\d*$", "", slug)
        slug = slug.replace("-", " ").strip()
        return unquote(slug)
    except Exception:
        return ""


def try_text(root, xpaths: list) -> str:
    for xpath in xpaths:
        try:
            el = root.find_element(By.XPATH, xpath)
            t  = el.text.strip()
            if t:
                return t
        except Exception:
            continue
    return ""


def try_attr(root, xpaths: list, attr: str) -> str:
    for xpath in xpaths:
        try:
            el  = root.find_element(By.XPATH, xpath)
            val = el.get_attribute(attr)
            if val:
                return val.strip()
        except Exception:
            continue
    return ""


def dedup_hash(title: str, company: str, location: str) -> str:
    def n(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return re.sub(r"\s+", " ", s)
    return hashlib.md5(f"{n(title)}|{n(company)}|{n(location)}".encode()).hexdigest()


def build_srp_url(query: str, page: int = 1) -> str:
    """
    Build a Naukri SRP URL directly.
    jobType=2 -> Internship filter | experience=0 -> Fresher
    Page 1:  /machine-learning-intern-jobs?k=machine%20learning%20intern&experience=0&jobType=2
    Page 2+: /machine-learning-intern-jobs-2?k=...
    """
    slug    = query.strip().lower().replace(" ", "-")
    encoded = query.strip().replace(" ", "%20")
    base    = f"{NAUKRI_BASE}/{slug}-jobs"
    if page > 1:
        base += f"-{page}"
    return f"{base}?k={encoded}&experience=0&jobType=2"


# ---------------------------------------------------------------------------
# NaukriScraper class
# ---------------------------------------------------------------------------

class NaukriScraper:

    def __init__(self, headless: bool = False):
        self.headless             = headless
        self.user_agent           = random.choice(USER_AGENTS)
        self.driver               = None
        self._session_hashes: set = set()
        self._first_query_done    = False
        self._setup_driver()

    def _setup_driver(self):
        """
        Create a Safari browser window and attach Selenium WebDriver to it.

        NOTE:
        - Safari does NOT support real headless mode; if headless=True is passed,
          we ignore it and run a normal visible Safari window.
        - Make sure:
            * Safari → Develop → “Allow Remote Automation” is enabled
            * `safaridriver --enable` has been run once in terminal
        """
        if self.headless:
            log.warning(
                "Safari WebDriver does not support headless mode. "
                "Running in normal (visible) mode instead."
            )

        try:
            # This uses the system Safari (macOS) WebDriver.
            self.driver = webdriver.Safari()
            # Optional: set a decent window size
            try:
                self.driver.set_window_size(1440, 900)
            except Exception:
                pass

            # The anti-bot JS is harmless on Safari, but we keep it for consistency.
            try:
                self.driver.execute_script("""
                    try {
                        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                        Object.defineProperty(navigator, 'plugins',   {get: () => [1, 2, 3]});
                        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                        window.chrome = window.chrome || { runtime: {} };
                    } catch (e) {}
                """)
            except Exception:
                pass

            log.info("Naukri Safari browser ready.")
        except Exception as e:
            log.error("Failed to start Safari WebDriver: %s", e, exc_info=True)
            raise

    def _check_session(self):
        log.info("Navigating to Naukri homepage...")
        self.driver.get(NAUKRI_HOMEPAGE)
        log.info("Current URL after get(): %s", self.driver.current_url)
        human_delay(3, 5)
        self._dismiss_overlays()
        try:
            user_el = self.driver.find_element(
                By.XPATH,
                "//*[contains(@class,'nI-gNb-drawer__icon') or "
                "contains(@class,'user-name') or contains(@class,'view-profile')]"
            )
            log.info("Naukri session active: %s", user_el.text.strip() or "(logged in)")
        except NoSuchElementException:
            log.warning(
                "Naukri session may not be active. "
                "If needed, log in manually in the opened Safari window."
            )

    def _dismiss_overlays(self):
        for xpath in OVERLAY_CLOSE_XPATHS:
            try:
                btn = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                btn.click()
                log.debug("Dismissed overlay via: %s", xpath[:60])
                time.sleep(0.5)
            except Exception:
                continue

    def _search_via_homepage(self, query: str) -> bool:
        log.info("Homepage search for: '%s'", query)
        try:
            employment_status = infer_employment_status_from_query(query)
            self.driver.get(NAUKRI_HOMEPAGE)
            human_delay(3, 5)
            self._dismiss_overlays()

            # Step 1 — activate search bar
            for xpath in SEARCHBAR_WRAPPER_XPATHS:
                try:
                    el = WebDriverWait(self.driver, 8).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    el.click()
                    human_delay(0.5, 1.2)
                    log.debug("Search bar activated")
                    break
                except Exception:
                    continue

            # Step 2/3 — only force the Internship job-type filter when the query is internship-like.
            if employment_status == "Intern":
                for xpath in JOBTYPE_INPUT_XPATHS:
                    try:
                        jt_input = WebDriverWait(self.driver, 6).until(
                            EC.element_to_be_clickable((By.XPATH, xpath))
                        )
                        self.driver.execute_script("arguments[0].click();", jt_input)
                        human_delay(0.5, 1.0)
                        log.debug("Job-type dropdown opened")
                        break
                    except Exception:
                        continue

                internship_selected = False
                for xpath in INTERNSHIP_OPTION_XPATHS:
                    try:
                        option = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, xpath))
                        )
                        option.click()
                        human_delay(0.4, 0.9)
                        log.debug("'Internship' selected")
                        internship_selected = True
                        break
                    except Exception:
                        continue
                if not internship_selected:
                    log.warning("Could not select 'Internship' from dropdown — proceeding anyway")
            else:
                log.debug(
                    "Leaving Naukri job-type unfiltered for '%s' query",
                    employment_status,
                )

            # Step 4 — type keyword
            kw_input = None
            for xpath in KEYWORD_INPUT_XPATHS:
                try:
                    kw_input = WebDriverWait(self.driver, 6).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    break
                except Exception:
                    continue

            if kw_input is None:
                log.error("Keyword input not found on homepage")
                return False

            kw_input.click()
            human_delay(0.3, 0.6)
            kw_input.send_keys(Keys.CONTROL + "a")
            human_delay(0.1, 0.3)
            kw_input.send_keys(Keys.DELETE)
            human_delay(0.2, 0.5)

            for char in query:
                kw_input.send_keys(char)
                time.sleep(random.uniform(0.04, 0.14))

            human_delay(0.8, 1.5)

            # Dismiss autocomplete then submit
            kw_input.send_keys(Keys.ESCAPE)
            time.sleep(0.3)
            kw_input.send_keys(Keys.RETURN)
            human_delay(3, 6)

            # Step 5 — verify SRP loaded
            current_url = self.driver.current_url
            if "naukri.com" in current_url and (
                "-jobs" in current_url or "?k=" in current_url
            ):
                log.info("SRP loaded: %s", current_url[:80])
                self._first_query_done = True
                return True

            # Fallback: click search button
            for xpath in SEARCH_BTN_XPATHS:
                try:
                    btn = self.driver.find_element(By.XPATH, xpath)
                    btn.click()
                    human_delay(3, 5)
                    break
                except Exception:
                    continue

            current_url = self.driver.current_url
            if "naukri.com" in current_url and "-jobs" in current_url:
                log.info("SRP loaded after button click: %s", current_url[:80])
                self._first_query_done = True
                return True

            log.warning("Unclear if SRP loaded: %s", current_url[:80])
            return False

        except Exception as e:
            log.error("Homepage search failed: %s", e, exc_info=True)
            return False

    def _open_in_new_tab(self, url: str) -> bool:
        try:
            self.driver.execute_script("window.open(arguments[0], '_blank');", url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            return True
        except Exception as e:
            log.debug("New-tab open failed for %s: %s", url, e)
            return False

    def _close_detail_tab(self, opened_new_tab: bool):
        try:
            if opened_new_tab and len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
                human_delay(0.8, 1.5)
            else:
                self.driver.back()
                human_delay(2, 3.5)
        except Exception:
            pass

    def _extract_description_from_element(self, el) -> str:
        """
        Try multiple ways of reading text from a description element,
        then clean and return the longest version.
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

    def _fetch_detail(self, job_url: str) -> dict:
        """
        Navigate to full job detail page.
        Returns dict: description, salary, duration, location_detail.
        Uses a new tab when possible so SRP state stays stable.
        """
        result = {
            "description": "",
            "salary": "",
            "duration": "",
            "location_detail": ""
        }

        if not job_url:
            return result

        opened_new_tab = False

        try:
            opened_new_tab = self._open_in_new_tab(job_url)
            if not opened_new_tab:
                self.driver.get(job_url)

            human_delay(2, 4)
            self._dismiss_overlays()

            ready = False
            for xpath in DETAIL_READY_XPATHS:
                try:
                    WebDriverWait(self.driver, 4).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    ready = True
                    break
                except Exception:
                    continue

            if not ready:
                log.debug("Detail page not clearly 'ready': %s", job_url)

            slow_scroll(self.driver, pixels=1600)
            human_delay(0.5, 1.0)

            # Location
            loc = try_text(self.driver, LOCATION_DETAIL_XPATHS)
            if loc:
                result["location_detail"] = clean_multiline_text(loc.split("\n")[0])

            # Salary
            salary = ""
            for xpath in SALARY_DETAIL_XPATHS:
                try:
                    el = self.driver.find_element(By.XPATH, xpath)
                    salary = el.get_attribute("title") or el.text.strip()
                    if salary:
                        break
                except Exception:
                    continue
            result["salary"] = clean_multiline_text(salary.split("\n")[0]) if salary else ""

            # Duration
            duration = ""
            for xpath in DURATION_DETAIL_XPATHS:
                try:
                    el = self.driver.find_element(By.XPATH, xpath)
                    duration = el.get_attribute("title") or el.text.strip()
                    if duration:
                        break
                except Exception:
                    continue
            result["duration"] = clean_multiline_text(duration.split("\n")[0]) if duration else ""

            # Description
            descriptions = []
            for xpath in DESC_XPATHS:
                try:
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    for el in elements:
                        text = self._extract_description_from_element(el)
                        if len(text) > 50:
                            descriptions.append(text)
                except Exception:
                    continue

            if descriptions:
                result["description"] = clean_multiline_text(max(descriptions, key=len))
                log.debug("Description captured: %d chars", len(result["description"]))
            else:
                log.debug("No description found: %s", job_url)

        except Exception as e:
            log.warning("Detail fetch failed for %s: %s", job_url, e)

        finally:
            self._close_detail_tab(opened_new_tab)

        return result

    def _parse_card(self, card) -> dict | None:
        try:
            title    = try_attr(card, TITLE_XPATHS, "title") or try_text(card, TITLE_XPATHS)
            raw_url  = try_attr(card, TITLE_XPATHS, "href")
            company  = try_text(card, COMPANY_XPATHS)
            location = try_text(card, LOCATION_CARD_XPATHS)
            salary   = (try_attr(card, SALARY_CARD_XPATHS, "title")
                        or try_text(card, SALARY_CARD_XPATHS))
            duration = (try_attr(card, DURATION_CARD_XPATHS, "title")
                        or try_text(card, DURATION_CARD_XPATHS))

            title    = title.split("\n")[0].strip()    if title    else ""
            company  = company.split("\n")[0].strip()  if company  else ""
            location = location.split("\n")[0].strip() if location else ""
            salary   = salary.split("\n")[0].strip()   if salary   else ""
            duration = duration.split("\n")[0].strip() if duration else ""
            job_url  = raw_url.strip()                 if raw_url  else ""

            if job_url and job_url.startswith("/"):
                job_url = NAUKRI_BASE + job_url

            if not title:
                return None

            return {
                "title":    title,
                "company":  company,
                "location": location,
                "salary":   salary,
                "duration": duration,
                "job_url":  job_url,
            }

        except StaleElementReferenceException:
            log.debug("Stale card element, skipping")
            return None
        except Exception as e:
            log.debug("Card parse error: %s", e)
            return None

    def _wait_for_cards(self) -> list:
        for xpath in CARD_XPATHS:
            try:
                WebDriverWait(self.driver, 12).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                cards = self.driver.find_elements(By.XPATH, xpath)
                if cards:
                    log.debug("%d cards via %s", len(cards), xpath[:55])
                    return cards
            except TimeoutException:
                continue
        return []

    def scrape_query(
        self,
        query: str,
        category: str,
        max_jobs: int,
        job_type: str = "2",
        location: str = "",
    ) -> list:
        """
        Scrape a single (query, category) pair from Naukri.

        IMPORTANT:
        We now *always* drive the browser via the explicit SRP URL
        (see build_srp_url) instead of trying to type into the homepage
        search bar. This is much less brittle and also makes it obvious
        in the visible browser window what is happening.
        """
        log.info("Query: '%s' | category: %s | target: %d", query, category, max_jobs)

        jobs: list[dict] = []
        dupes = 0
        page_num = 1

        while len(jobs) < max_jobs and page_num <= MAX_PAGES_PER_QUERY:
            # NOTE: build_srp_url signature in this version only takes query + page.
            url = build_srp_url(query=query, page=page_num)
            log.info("Opening SRP page %d | %s", page_num, url)
            self.driver.get(url)
            human_delay(4, 7)

            self._dismiss_overlays()

            # Fake some scrolling / mouse movement so the page fully loads
            for _ in range(3):
                slow_scroll(self.driver, pixels=random.randint(300, 700))
                human_delay(1.0, 2.0)

            if random.random() < 0.2:
                simulate_mouse(self.driver)

            cards = self._wait_for_cards()
            if not cards:
                log.warning(
                    "No cards on page %d — stopping query (URL=%s)",
                    page_num,
                    self.driver.current_url,
                )
                break

            parsed_cards: list[dict] = []
            for card in cards:
                parsed = self._parse_card(card)
                if parsed:
                    parsed_cards.append(parsed)
            log.debug("%d valid cards on page %d", len(parsed_cards), page_num)

            new_this_page = 0
            for parsed in parsed_cards:
                if len(jobs) >= max_jobs:
                    break

                h = dedup_hash(parsed["title"], parsed["company"], parsed["location"])
                if h in self._session_hashes:
                    dupes += 1
                    log.debug(
                        "Duplicate: %s @ %s",
                        parsed["title"],
                        parsed["company"],
                    )
                    continue

                human_delay(JOB_DELAY_MIN, JOB_DELAY_MAX)
                detail = self._fetch_detail(parsed["job_url"])

                location_final = detail["location_detail"] or parsed["location"]
                salary_final   = detail["salary"]           or parsed["salary"]
                duration_final = detail["duration"]         or parsed["duration"]

                self._session_hashes.add(h)
                job = {
                    "title":         parsed["title"],
                    "company":       parsed["company"],
                    "location":      location_final,
                    "description":   detail["description"],
                    "job_url":       parsed["job_url"],
                    "source":        "naukri",
                    "search_query":  query
                        or extract_query_from_url(self.driver.current_url),
                    "role_category": category,
                    "employment_status": infer_employment_status_from_query(query),
                    "scraped_at":    datetime.now(timezone.utc),
                    "salary":        salary_final,
                    "duration":      duration_final,
                }
                # ✅ Only add "Salary" field when we actually have a salary value
                if salary_final:
                    job["Salary"] = salary_final

                jobs.append(job)
                new_this_page += 1
                log.info(
                    "[%d/%d] %s @ %s | loc=%s | sal=%s | dur=%s",
                    len(jobs),
                    max_jobs,
                    job["title"],
                    job["company"],
                    job["location"],
                    job["salary"] or "—",
                    job["duration"] or "—",
                )

            log.info(
                "Page %d done: +%d new | %d dupes | %d collected",
                page_num,
                new_this_page,
                dupes,
                len(jobs),
            )

            if new_this_page == 0:
                log.info("No new jobs on page %d — stopping early")
                break

            page_num += 1
            human_delay(2, 5)

        log.info("Query '%s' done: %d collected | %d dupes skipped", query, len(jobs), dupes)
        return jobs

    def scrape_all_queries(
        self,
        jobs_per_query: int = JOBS_PER_QUERY,
        selected_categories: list[str] | None = None,
        selected_queries: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> pd.DataFrame:
        self._check_session()

        query_list = build_query_list(
            selected_categories=selected_categories,
            selected_queries=selected_queries,
            custom_queries=custom_queries,
        )
        total_queries = len(query_list)

        all_jobs = []
        log.info(
            "Naukri scrape starting | queries: %d | per query: %d | est. total: %d",
            total_queries, jobs_per_query, total_queries * jobs_per_query,
        )

        for idx, (query, category) in enumerate(query_list, start=1):
            log.info("--- [%d/%d] '%s' | %s ---", idx, total_queries, query, category.upper())
            try:
                jobs = self.scrape_query(query, category, max_jobs=jobs_per_query)
                all_jobs.extend(jobs)
                log.info("Running total: %d", len(all_jobs))
            except Exception as e:
                log.error("Query '%s' failed: %s", query, e, exc_info=True)
                continue

            if idx < total_queries:
                pause = random.uniform(QUERY_PAUSE_MIN, QUERY_PAUSE_MAX)
                log.debug("Pausing %.0fs", pause)
                time.sleep(pause)

        try:
            self.driver.quit()
        except Exception:
            pass

        df = pd.DataFrame(all_jobs)
        log.info("Naukri scrape complete: %d total jobs", len(df))
        if not df.empty and "role_category" in df.columns:
            for cat, cnt in df["role_category"].value_counts().items():
                log.info("  %-20s %d", cat, cnt)
        return df

    def quit(self):
        try:
            self.driver.quit()
        except Exception:
            pass


def save_to_csv(df: pd.DataFrame, filename: str = "naukri_jobs.csv", output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    log.info("CSV saved: %s (%d rows)", filepath, len(df))
    return filepath


def scrape_and_store(
    search_url:        str  = None,
    max_jobs:          int  = None,
    headless:          bool = False,
    li_at_cookie:      str  = None,   # unused, kept for signature compatibility
    jobs_per_query:    int  = None,
    selected_categories: list[str] | None = None,
    selected_queries:    list[str] | None = None,
    custom_queries:      list[str] | None = None,
    selected_fields:     list[str] | None = None,
) -> dict:
    """
    Drop-in equivalent of linkedin_scraper.scrape_and_store().
    Called by app.py Naukri scrape button. Returns same dict shape.
    Now supports selected_categories, selected_queries, custom_queries,
    and optional output field selection.
    """
    from database.mongo_client import get_collection, insert_jobs_bulk

    per_query = jobs_per_query or max_jobs or JOBS_PER_QUERY
    scraper   = NaukriScraper(headless=headless)
    df        = scraper.scrape_all_queries(
        jobs_per_query=per_query,
        selected_categories=selected_categories,
        selected_queries=selected_queries,
        custom_queries=custom_queries,
    )
    df = limit_output_fields(df, selected_fields=selected_fields)

    if df.empty:
        log.warning("No Naukri jobs scraped")
        return {"df": df, "inserted": 0, "duplicates": 0, "total": 0, "by_category": {}}

    save_to_csv(df, filename="naukri_raw_jobs.csv", output_dir="data")

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
