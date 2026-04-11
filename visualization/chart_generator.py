"""
visualization/chart_generator.py
==================================
Creates all matplotlib visualizations for the Streamlit dashboard.

Charts:
  1. plot_skill_demand()           - horizontal bar: skill frequency counts
  2. plot_tfidf_keywords()         - horizontal bar: TF-IDF keyword scores
  3. plot_tfidf_vs_frequency()     - scatter: TF-IDF score vs frequency
  4. plot_skill_cooccurrence()     - heatmap: skills appearing together
  5. plot_category_breakdown()     - bar: demand by skill category
  6. plot_skill_by_level()         - bar: skills per job level
  7. plot_job_title_distribution() - horizontal bar: most common titles
  8. plot_location_distribution()  - horizontal bar: top hiring locations
  9. save_chart()                  - utility: save figure to disk
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

FONT_TITLE = 13
FONT_LABEL = 10
FONT_TICK  = 9
FIGURE_DPI = 130

COLORS = {
    "blue":   "#4C9BE8",
    "green":  "#57C4A0",
    "orange": "#F28C6A",
    "purple": "#9B8FE0",
    "pink":   "#E87B9B",
    "teal":   "#4CC8C8",
    "yellow": "#F2C76A",
}

CATEGORY_COLORS = [
    "#4C9BE8", "#57C4A0", "#F28C6A", "#9B8FE0",
    "#E87B9B", "#4CC8C8", "#F2C76A",
]


def _style_ax(ax, title: str):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK)


def _annotate_bars_h(ax, bars, values, fmt="{}", offset=0.15):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val), va="center", ha="left", fontsize=FONT_TICK
        )


# ---------------------------------------------------------------------------
# 1. Skill Frequency
# ---------------------------------------------------------------------------

def plot_skill_demand(skill_counts: pd.Series, title: str = "Top Skills in Demand (Frequency)", top_n: int = 15) -> plt.Figure:
    data = skill_counts.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    bars = ax.barh(data.index, data.values, color=COLORS["blue"], edgecolor="white", height=0.7)
    _annotate_bars_h(ax, bars, data.values)
    ax.set_xlabel("Number of Job Postings", fontsize=FONT_LABEL)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered skill demand chart: top %d skills", top_n)
    return fig


# ---------------------------------------------------------------------------
# 2. TF-IDF Keywords
# ---------------------------------------------------------------------------

def plot_tfidf_keywords(keywords: list, title: str = "Top TF-IDF Keywords (Auto-Discovered)", top_n: int = 20) -> plt.Figure:
    if not keywords:
        log.warning("plot_tfidf_keywords: no keywords provided")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No keywords available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")
        return fig

    items  = keywords[:top_n]
    pairs  = sorted(zip([k for k, _ in items], [round(s, 3) for _, s in items]), key=lambda x: x[1])
    terms  = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    bars = ax.barh(terms, scores, color=COLORS["purple"], edgecolor="white", height=0.7)
    _annotate_bars_h(ax, bars, scores, fmt="{:.3f}", offset=0.003)
    ax.set_xlabel("Aggregate TF-IDF Score", fontsize=FONT_LABEL)
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered TF-IDF keywords chart: top %d keywords", top_n)
    return fig


# ---------------------------------------------------------------------------
# 3. TF-IDF vs Frequency Scatter
# ---------------------------------------------------------------------------

def plot_tfidf_vs_frequency(skill_counts: pd.Series, tfidf_scores: pd.Series, title: str = "Skill Importance: TF-IDF Score vs Frequency") -> plt.Figure:
    common = skill_counts.index.intersection(tfidf_scores.index)
    if len(common) < 3:
        log.warning("plot_tfidf_vs_frequency: only %d common skills, need at least 3", len(common))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "Not enough data for scatter plot\n(need 3 or more skills in both series)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.axis("off")
        return fig

    x = skill_counts[common].values.astype(float)
    y = tfidf_scores[common].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(x, y, c=y, cmap="viridis", s=100, alpha=0.8, edgecolors="white", linewidths=0.5)

    for xi, yi, label in zip(x, y, list(common)):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=7, alpha=0.85)

    ax.axvline(np.median(x), color="gray", linestyle="--", alpha=0.4, lw=1)
    ax.axhline(np.median(y), color="gray", linestyle="--", alpha=0.4, lw=1)
    ax.text(0.98, 0.98, "High demand and distinctive", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#2E7D32", alpha=0.7)
    ax.text(0.02, 0.98, "Niche but important", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#1565C0", alpha=0.7)
    ax.text(0.98, 0.02, "Common but generic", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#E65100", alpha=0.7)

    plt.colorbar(scatter, ax=ax, label="TF-IDF Score", shrink=0.7)
    ax.set_xlabel("Frequency (number of job postings)", fontsize=FONT_LABEL)
    ax.set_ylabel("Aggregate TF-IDF Score", fontsize=FONT_LABEL)
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered TF-IDF vs frequency scatter: %d skills plotted", len(common))
    return fig


# ---------------------------------------------------------------------------
# 4. Co-occurrence Heatmap
# ---------------------------------------------------------------------------

def plot_skill_cooccurrence(cooccurrence_df: pd.DataFrame, title: str = "Skill Co-occurrence Heatmap") -> plt.Figure:
    if cooccurrence_df.empty:
        log.warning("plot_skill_cooccurrence: empty co-occurrence DataFrame")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "Not enough data for co-occurrence heatmap",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.axis("off")
        return fig

    n      = len(cooccurrence_df)
    data   = cooccurrence_df.values.astype(float)
    max_v  = data.max() if data.max() > 0 else 1
    normed = data / max_v

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
    im = ax.imshow(normed, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cooccurrence_df.columns, rotation=45, ha="right", fontsize=FONT_TICK)
    ax.set_yticklabels(cooccurrence_df.index, fontsize=FONT_TICK)

    for i in range(n):
        for j in range(n):
            val = int(data[i, j])
            if val > 0:
                color = "white" if normed[i, j] > 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=max(6, FONT_TICK - 1), color=color)

    plt.colorbar(im, ax=ax, label="Relative Co-occurrence", shrink=0.8)
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered co-occurrence heatmap: %dx%d matrix", n, n)
    return fig


# ---------------------------------------------------------------------------
# 5. Category Breakdown
# ---------------------------------------------------------------------------

def plot_category_breakdown(category_df: pd.DataFrame, title: str = "Skill Demand by Category") -> plt.Figure:
    if category_df.empty:
        log.warning("plot_category_breakdown: empty category DataFrame")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    df     = category_df.sort_values("total_mentions")
    colors = CATEGORY_COLORS[:len(df)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.7)))
    bars = ax.barh(df["category"], df["total_mentions"], color=colors, edgecolor="white", height=0.6)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(row['total_mentions'])}  (top: {row['top_skill']})",
                va="center", ha="left", fontsize=FONT_TICK)

    ax.set_xlabel("Total Skill Mentions", fontsize=FONT_LABEL)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered category breakdown chart: %d categories", len(df))
    return fig


# ---------------------------------------------------------------------------
# 6. Skills by Job Level
# ---------------------------------------------------------------------------

def plot_skill_by_level(skill_by_level: dict, title: str = "Top Skills by Job Level") -> plt.Figure:
    levels = {k: v for k, v in skill_by_level.items() if not v.empty}

    if not levels:
        log.warning("plot_skill_by_level: no level data available")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5,
                "Not enough data to split by job level\n"
                "(need postings with intern, junior, or senior in title)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")
        return fig

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 5), sharey=False)
    if n_levels == 1:
        axes = [axes]

    level_colors = {"intern": COLORS["green"], "junior": COLORS["blue"], "senior": COLORS["orange"]}

    for ax, (level, skills) in zip(axes, levels.items()):
        if skills.empty:
            ax.axis("off")
            continue
        data  = skills.sort_values()
        color = level_colors.get(level, COLORS["purple"])
        bars  = ax.barh(data.index, data.values, color=color, edgecolor="white", height=0.6)
        _annotate_bars_h(ax, bars, data.values)
        ax.set_xlabel("Job Postings", fontsize=FONT_LABEL)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        _style_ax(ax, f"{level.title()} Roles")

    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    log.debug("Rendered skills-by-level chart: %d levels", n_levels)
    return fig


# ---------------------------------------------------------------------------
# 7. Job Title Distribution
# ---------------------------------------------------------------------------

def plot_job_title_distribution(df: pd.DataFrame, title: str = "Most Common Job Titles", top_n: int = 12) -> plt.Figure:
    counts = df["title"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    bars = ax.barh(counts.index, counts.values, color=COLORS["green"], edgecolor="white", height=0.7)
    _annotate_bars_h(ax, bars, counts.values)
    ax.set_xlabel("Number of Postings", fontsize=FONT_LABEL)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered job title distribution chart: top %d titles", top_n)
    return fig


# ---------------------------------------------------------------------------
# 8. Location Distribution
# ---------------------------------------------------------------------------

def plot_location_distribution(df: pd.DataFrame, title: str = "Top Hiring Locations", top_n: int = 12) -> plt.Figure:
    counts = df["location"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    bars = ax.barh(counts.index, counts.values, color=COLORS["orange"], edgecolor="white", height=0.7)
    _annotate_bars_h(ax, bars, counts.values)
    ax.set_xlabel("Number of Postings", fontsize=FONT_LABEL)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _style_ax(ax, title)
    plt.tight_layout()
    log.debug("Rendered location distribution chart: top %d locations", top_n)
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_chart(fig: plt.Figure, filename: str, output_dir: str = "data") -> str:
    """Save a matplotlib figure to disk. Returns the full filepath."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info("Chart saved: %s", filepath)
    return filepath