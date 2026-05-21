"""
Gradio GUI for the Book Recommender System.

Run from the project root:
    python src/app.py
"""
from __future__ import annotations

import os

import gradio as gr

from recommender import (
    build_tfidf_matrix,
    get_categories,
    load_data,
    recommend_by_query,
    recommend_by_title,
)
from utils import truncate

# ---------------------------------------------------------------------------
# Load data and build model once at startup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_books.csv")

df = load_data(DATA_PATH)
vectorizer, tfidf_matrix = build_tfidf_matrix(df)

CATEGORIES = get_categories(df)
VALID_YEARS = df.loc[df["published_year"] > 0, "published_year"]
MIN_YEAR = int(VALID_YEARS.min()) if not VALID_YEARS.empty else 1900
MAX_YEAR = int(VALID_YEARS.max()) if not VALID_YEARS.empty else 2025


# ---------------------------------------------------------------------------
# HTML card renderer
# ---------------------------------------------------------------------------

def _results_to_html(results) -> str:
    if results is None or results.empty:
        return (
            "<p style='color:#c0392b; font-size:1.05em; padding:12px;'>"
            "No books found. Try a different title / query or relax your filters."
            "</p>"
        )

    cards = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        year_str = str(row["Year"]) if row["Year"] > 0 else "Unknown"
        snippet = truncate(row["Description"], 300)
        cards.append(f"""
<div style="border:1px solid #dde3ef; border-radius:10px; padding:16px 20px;
            margin-bottom:14px; background:#ffffff;
            box-shadow:0 2px 6px rgba(0,0,0,.07);">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <h3 style="margin:0 0 4px; color:#1a1a2e; font-size:1.1em;">
      {rank}. {row['Title']}
    </h3>
    <span style="background:#f0f4ff; color:#3a5bd9; border-radius:6px;
                 padding:2px 10px; font-size:.82em; white-space:nowrap; margin-left:12px;">
      Score: {row['Score']:.3f}
    </span>
  </div>
  <p style="margin:0 0 8px; color:#555; font-size:.9em;">
    <b>Author(s):</b> {row['Authors']} &nbsp;·&nbsp;
    <b>Genre:</b> {row['Category']} &nbsp;·&nbsp;
    <b>Year:</b> {year_str} &nbsp;·&nbsp;
    <b>Rating:</b> ⭐ {row['Rating']:.2f}
  </p>
  <p style="margin:0; color:#444; font-size:.93em; line-height:1.45;">{snippet}</p>
</div>""")

    return "<div>" + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Callback functions (called by Gradio on button click)
# ---------------------------------------------------------------------------

def search_by_title(title, n, min_year, max_year, min_rating, category, text_weight):
    if not str(title).strip():
        return "<p style='padding:8px;'>Please enter a book title.</p>"
    results = recommend_by_title(
        query_title=str(title),
        df=df,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        n=int(n),
        min_year=int(min_year),
        max_year=int(max_year),
        min_rating=float(min_rating),
        category=str(category),
        text_weight=float(text_weight),
    )
    return _results_to_html(results)


def search_by_query(query, n, min_year, max_year, min_rating, category, text_weight):
    if not str(query).strip():
        return "<p style='padding:8px;'>Please describe what kind of book you are looking for.</p>"
    results = recommend_by_query(
        query_text=str(query),
        df=df,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        n=int(n),
        min_year=int(min_year),
        max_year=int(max_year),
        min_rating=float(min_rating),
        category=str(category),
        text_weight=float(text_weight),
    )
    return _results_to_html(results)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Book Recommender System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 📚 Book Recommender System
Find books similar to one you love, or describe exactly what you want to read.
""")

    with gr.Row():

        # ── Left column: shared filters ──────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ⚙️ Filters")

            n_results = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Number of results",
            )
            min_year_input = gr.Number(
                value=0, label="Published from year (0 = no filter)", precision=0,
            )
            max_year_input = gr.Number(
                value=0, label="Published up to year (0 = no filter)", precision=0,
            )
            min_rating_input = gr.Slider(
                minimum=0.0, maximum=5.0, value=0.0, step=0.1,
                label="Minimum rating (0 = no filter)",
            )
            category_input = gr.Dropdown(
                choices=CATEGORIES, value="All", label="Category",
            )
            text_weight_input = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                label="Text weight  [0 = popularity-driven · 1 = pure text match]",
            )

        # ── Right column: search tabs ─────────────────────────────────────────
        with gr.Column(scale=3):

            with gr.Tab("🔍 Find similar books"):
                gr.Markdown(
                    "Enter the title of a book you enjoyed and get recommendations "
                    "based on similar content, genre, and author style."
                )
                title_input = gr.Textbox(
                    label="Book title",
                    placeholder="e.g. The Great Gatsby",
                )
                title_btn = gr.Button("Find similar books", variant="primary")
                title_output = gr.HTML()

                shared_inputs_title = [
                    title_input, n_results,
                    min_year_input, max_year_input,
                    min_rating_input, category_input,
                    text_weight_input,
                ]
                title_btn.click(search_by_title, inputs=shared_inputs_title, outputs=title_output)
                title_input.submit(search_by_title, inputs=shared_inputs_title, outputs=title_output)

            with gr.Tab("💬 Search by description"):
                gr.Markdown(
                    "Describe the kind of book you want in plain language — "
                    "themes, setting, mood, or anything else."
                )
                query_input = gr.Textbox(
                    label="Describe what you want to read",
                    placeholder=(
                        "e.g. a gripping thriller set in Victorian London "
                        "with a brilliant detective and dark secrets"
                    ),
                    lines=3,
                )
                query_btn = gr.Button("Search", variant="primary")
                query_output = gr.HTML()

                shared_inputs_query = [
                    query_input, n_results,
                    min_year_input, max_year_input,
                    min_rating_input, category_input,
                    text_weight_input,
                ]
                query_btn.click(search_by_query, inputs=shared_inputs_query, outputs=query_output)
                query_input.submit(search_by_query, inputs=shared_inputs_query, outputs=query_output)

    gr.Markdown(
        "<p style='text-align:center; color:#999; font-size:.85em; margin-top:16px;'>"
        "Book Recommender System · EADA Business School · Deep Learning & LLM Project"
        "</p>"
    )

demo.launch()
