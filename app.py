"""
Streamlit Dashboard — Internship & Hiring Trend Analyzer
=========================================================
Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import scraper.linkedin_scraper as linkedin_scraper
from settings import (
    APP_NAME,
    DEFAULT_LINKEDIN_SEARCH_URL,
    STREAMLIT_HEADLESS_DEFAULT,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
)

st.title(f"📊 {APP_NAME}")
st.markdown("Scrape → Clean → Analyze → Visualize job/internship trends from LinkedIn & Naukri.")

# ── MongoDB helpers (lazy, cached) ────────────────────────────────────────────

@st.cache_resource
def _get_collection():
    from database.mongo_client import get_collection
    return get_collection()


@st.cache_data(ttl=300)
def _load_df_from_mongo() -> pd.DataFrame:
    from database.mongo_client import load_jobs_to_dataframe
    col = _get_collection()
    return load_jobs_to_dataframe(col)


@st.cache_data(ttl=300)
def _get_stats() -> dict:
    from database.mongo_client import get_collection_stats
    col = _get_collection()
    return get_collection_stats(col)


def _invalidate_cache():
    _load_df_from_mongo.clear()
    _get_stats.clear()


# ── Session state ─────────────────────────────────────────────────────────────
for key in ("df_raw", "df_clean", "skill_counts", "keywords"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Analysis helper ───────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame):
    from processing.data_cleaner import clean_data
    from analysis.skill_analyzer import count_skills, extract_tfidf_keywords

    with st.spinner("Cleaning data..."):
        df_clean = clean_data(df)
    with st.spinner("Counting skills..."):
        skill_counts = count_skills(df_clean)
    with st.spinner("Extracting TF-IDF keywords..."):
        keywords = extract_tfidf_keywords(df_clean, top_n=15)

    st.session_state.df_clean = df_clean
    st.session_state.skill_counts = skill_counts
    st.session_state.keywords = keywords
    return df_clean, skill_counts, keywords


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# LinkedIn URL input (kept for direct-URL scraping compatibility)
search_url = st.sidebar.text_input(
    "LinkedIn search URL (optional)",
    value=DEFAULT_LINKEDIN_SEARCH_URL,
)
max_jobs = st.sidebar.slider("Max jobs to scrape", 5, 50, 20, 5)
headless = st.sidebar.checkbox(
    "Headless browser",
    value=STREAMLIT_HEADLESS_DEFAULT,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Or load existing data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_up = pd.read_csv(uploaded_file)
    st.session_state.df_raw = df_up
    st.sidebar.success(f"Loaded {len(df_up)} rows from CSV.")
    run_analysis(df_up)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(
    tab_data, tab_ml, tab_skills, tab_tfidf,
    tab_charts, tab_loc, tab_search, tab_export
) = st.tabs([
    "🏠 Data",
    "🧹 Clean & ML",
    "📈 Skill Frequency",
    "🔑 TF-IDF Keywords",
    "📊 Charts",
    "🌍 Locations & Titles",
    "🔍 Skill Job Search",
    "💾 Scrape & Export",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Raw Data
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("🏠 Raw Data")

    # DB stats row
    try:
        stats = _get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total in DB", stats.get("total", 0))
        c2.metric("LinkedIn", stats.get("linkedin", 0))
        c3.metric("Naukri", stats.get("naukri", 0))
        c4.metric("Unique companies", stats.get("unique_companies", "—"))
    except Exception as e:
        st.info(f"DB stats unavailable: {e}")

    st.divider()

    # Decide which dataframe to show: session scrape > mongo > empty
    display_df = None
    if st.session_state.df_raw is not None:
        display_df = st.session_state.df_raw
        st.caption("Showing data from current scrape session.")
    else:
        try:
            display_df = _load_df_from_mongo()
            if display_df.empty:
                st.info("No data in MongoDB yet. Scrape some jobs in the **Scrape & Export** tab.")
            else:
                st.caption(f"Showing {len(display_df)} jobs loaded from MongoDB.")
        except Exception as e:
            st.warning(f"Could not load from MongoDB: {e}")

    if display_df is not None and not display_df.empty:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        sources = ["All"] + sorted(display_df["source"].dropna().unique().tolist()) if "source" in display_df.columns else ["All"]
        src_sel = fc1.selectbox("Source", sources, key="d_src")
        kw_sel = fc2.text_input("Search title / company", key="d_kw")
        if "role_category" in display_df.columns:
            cats = ["All"] + sorted(display_df["role_category"].dropna().unique().tolist())
            cat_sel = fc3.selectbox("Category", cats, key="d_cat")
        else:
            cat_sel = "All"

        fdf = display_df.copy()
        if src_sel != "All":
            fdf = fdf[fdf["source"] == src_sel]
        if cat_sel != "All":
            fdf = fdf[fdf["role_category"] == cat_sel]
        if kw_sel:
            mask = (
                fdf.get("title", pd.Series(dtype=str)).str.contains(kw_sel, case=False, na=False)
                | fdf.get("company", pd.Series(dtype=str)).str.contains(kw_sel, case=False, na=False)
            )
            fdf = fdf[mask]

        st.dataframe(fdf, use_container_width=True)
        st.caption(f"Showing {len(fdf)} of {len(display_df)} records")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Clean & ML
# ═════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.header("🧹 Cleaned Data")

    source_df = st.session_state.df_raw
    if source_df is None:
        try:
            source_df = _load_df_from_mongo()
        except Exception:
            source_df = None

    if source_df is None or source_df.empty:
        st.info("No data available. Scrape or upload a CSV first.")
    else:
        if st.button("▶️ Run Cleaning & Analysis", key="run_analysis_btn"):
            run_analysis(source_df)

        if st.session_state.df_clean is not None:
            st.subheader(f"Cleaned: {len(st.session_state.df_clean)} records")
            st.dataframe(st.session_state.df_clean, use_container_width=True)

            csv_clean = st.session_state.df_clean.to_csv(index=False).encode()
            st.download_button("⬇️ Download cleaned CSV", csv_clean, "cleaned_jobs.csv", "text/csv")
        else:
            st.info("Click **Run Cleaning & Analysis** above.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Skill Frequency
# ═════════════════════════════════════════════════════════════════════════════
with tab_skills:
    st.header("📈 Skill Frequency")

    if st.session_state.skill_counts is None:
        st.info("Run **Cleaning & Analysis** in the 🧹 tab first.")
    else:
        skill_counts = st.session_state.skill_counts
        top_n = st.slider("Top N skills", 3, len(skill_counts), min(10, len(skill_counts)), key="sk_n")
        top = skill_counts.head(top_n)

        from visualization.chart_generator import plot_skill_demand
        fig = plot_skill_demand(top, title="Top Skills in Job Postings", top_n=top_n)
        st.pyplot(fig)

        st.subheader("Frequency Table")
        st.dataframe(
            top.reset_index().rename(columns={"index": "Skill", 0: "Count"}),
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TF-IDF Keywords
# ═════════════════════════════════════════════════════════════════════════════
with tab_tfidf:
    st.header("🔑 TF-IDF Keywords")

    if st.session_state.keywords is None:
        st.info("Run **Cleaning & Analysis** in the 🧹 tab first.")
    else:
        keywords = st.session_state.keywords
        if not keywords:
            st.warning("No keywords extracted — descriptions may be empty.")
        else:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
            kw_df["TF-IDF Score"] = kw_df["TF-IDF Score"].round(4)

            st.dataframe(kw_df, use_container_width=True)
            st.bar_chart(kw_df.set_index("Keyword")["TF-IDF Score"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Charts
# ═════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.header("📊 Charts")

    df_clean = st.session_state.df_clean
    if df_clean is None:
        st.info("Run **Cleaning & Analysis** in the 🧹 tab first.")
    else:
        from visualization.chart_generator import (
            plot_skill_demand,
            plot_job_title_distribution,
            plot_location_distribution,
        )

        top_n_chart = st.slider("Top N items per chart", 5, 20, 10, key="chart_n")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Skills")
            if st.session_state.skill_counts is not None:
                fig1 = plot_skill_demand(
                    st.session_state.skill_counts,
                    title="Top Skills in Demand",
                    top_n=top_n_chart,
                )
                st.pyplot(fig1)
            else:
                st.info("No skill data.")

        with col2:
            st.subheader("Top Job Titles")
            if "title" in df_clean.columns:
                fig2 = plot_job_title_distribution(df_clean, title="Most Common Titles", top_n=top_n_chart)
                st.pyplot(fig2)
            else:
                st.info("No title column.")

        st.subheader("Top Locations")
        if "location" in df_clean.columns:
            fig3 = plot_location_distribution(df_clean, title="Top Locations", top_n=top_n_chart)
            st.pyplot(fig3)
        else:
            st.info("No location column.")

        # Source breakdown (if multi-source data)
        if "source" in df_clean.columns and df_clean["source"].nunique() > 1:
            st.subheader("Jobs by Source")
            st.bar_chart(df_clean["source"].value_counts())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Locations & Titles
# ═════════════════════════════════════════════════════════════════════════════
with tab_loc:
    st.header("🌍 Locations & Titles")

    df_view = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df_view is None:
        try:
            df_view = _load_df_from_mongo()
        except Exception:
            df_view = None

    if df_view is None or df_view.empty:
        st.info("No data available.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Top Locations")
            if "location" in df_view.columns:
                loc_counts = df_view["location"].dropna().value_counts().head(20)
                st.bar_chart(loc_counts)
                st.dataframe(
                    loc_counts.reset_index().rename(columns={"index": "Location", "location": "Count"}),
                    use_container_width=True,
                )

        with c2:
            st.subheader("Top Job Titles")
            if "title" in df_view.columns:
                title_counts = df_view["title"].dropna().value_counts().head(20)
                st.bar_chart(title_counts)
                st.dataframe(
                    title_counts.reset_index().rename(columns={"index": "Title", "title": "Count"}),
                    use_container_width=True,
                )

        # Naukri-only: salary & duration
        if "salary" in df_view.columns and df_view["salary"].notna().any():
            st.subheader("💰 Salary Distribution (Naukri)")
            sal = df_view["salary"].dropna().astype(str)
            sal = sal[sal.str.strip() != ""]
            if not sal.empty:
                st.bar_chart(sal.value_counts().head(15))

        if "duration" in df_view.columns and df_view["duration"].notna().any():
            st.subheader("⏱️ Internship Duration (Naukri)")
            dur = df_view["duration"].dropna().astype(str)
            dur = dur[dur.str.strip() != ""]
            if not dur.empty:
                st.bar_chart(dur.value_counts().head(10))


# =============================================================================
# TAB 7 — Skill-Based Job Search
# =============================================================================
with tab_search:
    st.header("🔍 Skill-Based Job Search")
    st.markdown(
        "Enter one or more skills to find matching internships from your scraped dataset. "
        "Results are ranked by how many of your skills appear in the job description and title."
    )

    # Load data — reuse the same cached MongoDB call used by every other tab
    try:
        search_base_df = _load_df_from_mongo()
    except Exception as e:
        search_base_df = pd.DataFrame()
        st.warning(f"Could not load data from MongoDB: {e}")

    if search_base_df.empty:
        st.info("No data in MongoDB yet. Scrape some jobs in the **💾 Scrape & Export** tab first.")
    else:
        # ── Input row ──────────────────────────────────────────────────────
        skill_input = st.text_input(
            "🔎 Skills (comma-separated)",
            placeholder="e.g.  python, machine learning, tensorflow",
            key="skill_search_input",
        )

        # ── Optional pre-filters ───────────────────────────────────────────
        fc1, fc2 = st.columns(2)
        with fc1:
            if "source" in search_base_df.columns:
                src_opts = ["All"] + sorted(search_base_df["source"].dropna().unique().tolist())
                src_filter = st.selectbox("Filter by source", src_opts, key="ss_src")
            else:
                src_filter = "All"
        with fc2:
            if "role_category" in search_base_df.columns:
                cat_opts = ["All"] + sorted(search_base_df["role_category"].dropna().unique().tolist())
                cat_filter = st.selectbox("Filter by category", cat_opts, key="ss_cat")
            else:
                cat_filter = "All"

        # ── Search button ──────────────────────────────────────────────────
        search_clicked = st.button("🔎 Search Jobs", key="skill_search_btn")

        if search_clicked:
            raw_skills = [s.strip() for s in skill_input.split(",") if s.strip()]

            if not raw_skills:
                st.warning("Please enter at least one skill before searching.")
            else:
                from analysis.skill_analyzer import search_jobs_by_skills

                # Apply source / category pre-filters
                pool = search_base_df.copy()
                if src_filter != "All":
                    pool = pool[pool["source"] == src_filter]
                if cat_filter != "All":
                    pool = pool[pool["role_category"] == cat_filter]

                with st.spinner(f"Searching {len(pool):,} jobs for: **{', '.join(raw_skills)}** …"):
                    results = search_jobs_by_skills(pool, raw_skills)

                # ── No results ─────────────────────────────────────────────
                if results.empty:
                    st.info(
                        f"No jobs found matching: **{', '.join(raw_skills)}**.\n\n"
                        "Suggestions: try shorter terms, remove filters, or check spelling."
                    )

                # ── Results found ──────────────────────────────────────────
                else:
                    n_total = len(pool)
                    n_found = len(results)
                    pct = n_found / n_total * 100

                    st.success(
                        f"✅ Found **{n_found}** matching job{'s' if n_found != 1 else ''} "
                        f"({pct:.1f}% of {n_total:,} searched)"
                    )

                    # Score breakdown: how many jobs matched 1 skill, 2, 3 …
                    st.subheader("Match Score Breakdown")
                    score_counts = (
                        results["match_score"]
                        .value_counts()
                        .sort_index(ascending=False)
                    )
                    breakdown_df = pd.DataFrame({
                        "Skills matched": [
                            f"{i} / {len(raw_skills)}" for i in score_counts.index
                        ],
                        "Jobs found": score_counts.values,
                    })
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

                    # Quick bar so the distribution is immediately visible
                    st.bar_chart(
                        score_counts.rename("Jobs").rename_axis("Skills matched"),
                    )

                    st.divider()

                    # ── Results table ──────────────────────────────────────
                    st.subheader("Results — sorted by relevance")

                    preferred_cols = [
                        "match_score",
                        "matched_skills",
                        "title",
                        "company",
                        "location",
                        "source",
                        "search_query",
                        "role_category",
                        "job_url",
                    ]
                    display_cols = [c for c in preferred_cols if c in results.columns]

                    st.dataframe(
                        results[display_cols],
                        use_container_width=True,
                        column_config={
                            "match_score": st.column_config.NumberColumn(
                                "Score",
                                help="How many of your skills appear in this job",
                                format="%d",
                            ),
                            "matched_skills": st.column_config.TextColumn(
                                "Matched skills",
                                help="Which of your skills were found",
                            ),
                            "job_url": st.column_config.LinkColumn("Apply"),
                        },
                    )

                    # Download
                    csv_out = results[display_cols].to_csv(index=False).encode()
                    st.download_button(
                        label="⬇️ Download results as CSV",
                        data=csv_out,
                        file_name="skill_search_results.csv",
                        mime="text/csv",
                        key="skill_search_dl",
                    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — Scrape & Export
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("💾 Scrape & Export")

    # Import scraper modules so we can tweak their config from the UI
    import scraper.linkedin_scraper as linkedin_scraper
    import scraper.naukri_scraper as naukri_scraper

    # ── LinkedIn ──────────────────────────────────────────────────────────────
    st.subheader("🔵 LinkedIn Scraper")

    li_jobs_slider = st.slider(
        "Jobs per query (LinkedIn)",
        1, 25,
        int(getattr(linkedin_scraper, "JOBS_PER_QUERY", 5)),
        key="li_slider",
    )

    with st.expander("⚙️ Advanced LinkedIn settings"):
        lc1, lc2 = st.columns(2)

        li_max_pages = lc1.slider(
            "Max pages per query",
            1, 8,
            int(getattr(linkedin_scraper, "MAX_PAGES_PER_QUERY", 3)),
            key="li_max_pages",
        )

        li_job_delay_min = lc1.number_input(
            "Job delay min (sec)",
            min_value=0.5,
            max_value=30.0,
            value=float(getattr(linkedin_scraper, "JOB_DELAY_MIN", 4.0)),
            step=0.5,
            key="li_job_delay_min",
        )

        li_job_delay_max = lc1.number_input(
            "Job delay max (sec)",
            min_value=0.5,
            max_value=60.0,
            value=float(getattr(linkedin_scraper, "JOB_DELAY_MAX", 9.0)),
            step=0.5,
            key="li_job_delay_max",
        )

        li_query_pause_min = lc2.number_input(
            "Query pause min (sec)",
            min_value=1.0,
            max_value=180.0,
            value=float(getattr(linkedin_scraper, "QUERY_PAUSE_MIN", 12.0)),
            step=1.0,
            key="li_query_pause_min",
        )

        li_query_pause_max = lc2.number_input(
            "Query pause max (sec)",
            min_value=1.0,
            max_value=240.0,
            value=float(getattr(linkedin_scraper, "QUERY_PAUSE_MAX", 25.0)),
            step=1.0,
            key="li_query_pause_max",
        )

        st.caption("Tune these if you hit captchas / blocking, or keep them small for quick demos.")

    if st.button("🚀 Scrape LinkedIn Jobs", key="li_btn"):
        # Apply UI settings to scraper config before running it
        linkedin_scraper.JOBS_PER_QUERY      = int(li_jobs_slider)
        linkedin_scraper.MAX_PAGES_PER_QUERY = int(li_max_pages)
        linkedin_scraper.JOB_DELAY_MIN       = float(li_job_delay_min)
        linkedin_scraper.JOB_DELAY_MAX       = float(li_job_delay_max)
        linkedin_scraper.QUERY_PAUSE_MIN     = float(li_query_pause_min)
        linkedin_scraper.QUERY_PAUSE_MAX     = float(li_query_pause_max)

        with st.spinner("Scraping LinkedIn… this may take several minutes."):
            try:
                result = linkedin_scraper.scrape_and_store(
                    search_url=search_url,   # from sidebar
                    max_jobs=max_jobs,       # from sidebar
                    headless=headless,       # from sidebar
                    jobs_per_query=li_jobs_slider,
                )
                _invalidate_cache()
                st.success(
                    f"✅ LinkedIn done! "
                    f"Inserted: **{result['inserted']}** | "
                    f"Duplicates: **{result['duplicates']}** | "
                    f"Total: **{result['total']}**"
                )
                if result.get("by_category"):
                    st.json(result["by_category"])
                if result.get("df") is not None and not result["df"].empty:
                    st.session_state.df_raw = result["df"]
                    st.dataframe(result["df"].head(20), use_container_width=True)
            except Exception as e:
                st.error(f"LinkedIn scrape failed: {e}")

    st.divider()

    # ── Naukri ────────────────────────────────────────────────────────────────
    st.subheader("🔶 Naukri Scraper")

    nc1, nc2 = st.columns(2)

    with nc1:
        naukri_mode = st.selectbox(
            "Job Type",
            ["Internship", "Job"],
            key="naukri_job_type",
        )

    with nc2:
        naukri_location = st.text_input(
            "Location (optional)",
            value="",
            key="naukri_location",
        )

    naukri_keywords = st.text_area(
        "Enter keywords / categories (one per line)",
        value="machine learning intern\ndata science intern\npython developer intern",
        height=160,
        key="naukri_keywords",
    )

    naukri_jobs_slider = st.slider(
        "Jobs per keyword (Naukri)",
        1, 20,
        int(getattr(naukri_scraper, "JOBS_PER_QUERY", 5)),
        key="naukri_slider",
    )

    with st.expander("⚙️ Advanced Naukri settings"):
        ac1, ac2 = st.columns(2)

        naukri_max_pages = ac1.slider(
            "Max pages per keyword",
            1, 5,
            int(getattr(naukri_scraper, "MAX_PAGES_PER_QUERY", 3)),
            key="naukri_max_pages",
        )

        naukri_job_delay_min = ac1.number_input(
            "Job delay min (sec)",
            min_value=0.5,
            max_value=30.0,
            value=float(getattr(naukri_scraper, "JOB_DELAY_MIN", 2.0)),
            step=0.5,
            key="naukri_job_delay_min",
        )

        naukri_job_delay_max = ac1.number_input(
            "Job delay max (sec)",
            min_value=0.5,
            max_value=60.0,
            value=float(getattr(naukri_scraper, "JOB_DELAY_MAX", 5.0)),
            step=0.5,
            key="naukri_job_delay_max",
        )

        naukri_query_pause_min = ac2.number_input(
            "Query pause min (sec)",
            min_value=1.0,
            max_value=180.0,
            value=float(getattr(naukri_scraper, "QUERY_PAUSE_MIN", 8.0)),
            step=1.0,
            key="naukri_query_pause_min",
        )

        naukri_query_pause_max = ac2.number_input(
            "Query pause max (sec)",
            min_value=1.0,
            max_value=240.0,
            value=float(getattr(naukri_scraper, "QUERY_PAUSE_MAX", 16.0)),
            step=1.0,
            key="naukri_query_pause_max",
        )

        st.caption("Same idea for Naukri – tune speed vs safety.")

    if st.button("🚀 Scrape Naukri Jobs", key="naukri_btn"):
        from scraper.naukri_scraper import scrape_and_store as naukri_scrape

        queries = [q.strip() for q in naukri_keywords.splitlines() if q.strip()]

        if not queries:
            st.warning("Please enter at least one keyword.")
        else:
            # Apply UI settings to scraper module-level config
            naukri_scraper.JOBS_PER_QUERY      = int(naukri_jobs_slider)
            naukri_scraper.MAX_PAGES_PER_QUERY = int(naukri_max_pages)
            naukri_scraper.JOB_DELAY_MIN       = float(naukri_job_delay_min)
            naukri_scraper.JOB_DELAY_MAX       = float(naukri_job_delay_max)
            naukri_scraper.QUERY_PAUSE_MIN     = float(naukri_query_pause_min)
            naukri_scraper.QUERY_PAUSE_MAX     = float(naukri_query_pause_max)

            with st.spinner("Scraping Naukri… this may take several minutes."):
                try:
                    result = naukri_scrape(
                        queries=queries,
                        job_type=naukri_mode,
                        location=naukri_location,
                        jobs_per_query=naukri_jobs_slider,
                        headless=headless,
                    )
                    _invalidate_cache()
                    st.success(
                        f"✅ Naukri done! "
                        f"Inserted: **{result['inserted']}** | "
                        f"Duplicates: **{result['duplicates']}** | "
                        f"Total: **{result['total']}**"
                    )
                    if result.get("by_category"):
                        st.json(result["by_category"])
                    if result.get("df") is not None and not result["df"].empty:
                        st.session_state.df_raw = result["df"]
                        st.dataframe(result["df"].head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Naukri scrape failed: {e}")

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("📤 Export All Data")
    fmt = st.radio("Format", ["CSV", "JSON"], horizontal=True, key="exp_fmt")

    if st.button("⬇️ Export from MongoDB", key="exp_btn"):
        try:
            df_exp = _load_df_from_mongo()
            if df_exp.empty:
                st.warning("Nothing to export.")
            else:
                os.makedirs("data", exist_ok=True)
                if fmt == "CSV":
                    data = df_exp.to_csv(index=False).encode()
                    fname = "jobs_export.csv"
                    mime = "text/csv"
                else:
                    data = df_exp.to_json(orient="records", indent=2).encode()
                    fname = "jobs_export.json"
                    mime = "application/json"
                st.download_button(f"📥 Download {fmt}", data, fname, mime)
                st.success(f"Ready to download — {len(df_exp)} records.")
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.divider()

    # ── DB stats ──────────────────────────────────────────────────────────────
    st.subheader("🗄️ Database Stats")
    if st.button("🔄 Refresh", key="refresh_btn"):
        _invalidate_cache()
        st.rerun()

    try:
        stats = _get_stats()
        st.json(stats)
    except Exception as e:
        st.warning(f"Could not load stats: {e}")
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with Selenium · pandas · scikit-learn · matplotlib · MongoDB · Streamlit")
