"""
BookWise — Apple Books-Inspired Book Recommendation System
==========================================================
Tabs:
  📖 Home        — Hero · Top 10 · Recommended For You · New & Trending
  🔍 Find Similar — Content-based title-to-title recommendations
  📂 Browse       — Filter by category, year, rating
  💬 Describe     — Free-text query → matching books
  ⬆️  Upload       — Swap in your own CSV dataset

Run from the project root:
    python src/app.py

Install dependencies:
    pip install gradio pandas numpy scikit-learn
"""
from __future__ import annotations

import os
import urllib.parse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gradio as gr

from recommender import (
    browse_by_category,
    build_tfidf_matrix,
    get_categories,
    load_data,
    recommend_by_query,
    recommend_by_title,
)
from utils import truncate

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA  (loaded once at startup; can be replaced via the Upload tab)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_books.csv")

df               = load_data(DATA_PATH)
vectorizer, tfidf_matrix = build_tfidf_matrix(df)
CATEGORIES       = get_categories(df)
ALL_TITLES       = sorted(df["title"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 2. VISUAL PALETTE
# ─────────────────────────────────────────────────────────────────────────────

_GRADS = [
    "linear-gradient(160deg,#1a1a2e,#16213e)",
    "linear-gradient(160deg,#2d5016,#52734d)",
    "linear-gradient(160deg,#7b2d8b,#c06c84)",
    "linear-gradient(160deg,#c0392b,#e74c3c)",
    "linear-gradient(160deg,#b7410e,#f39c12)",
    "linear-gradient(160deg,#1565C0,#42A5F5)",
    "linear-gradient(160deg,#7B3F00,#D4A017)",
    "linear-gradient(160deg,#134e5e,#71b280)",
    "linear-gradient(160deg,#0F2027,#203a43)",
    "linear-gradient(160deg,#360033,#0b8793)",
    "linear-gradient(160deg,#4e54c8,#8f94fb)",
    "linear-gradient(160deg,#cc2b5e,#753a88)",
    "linear-gradient(160deg,#ee0979,#ff6a00)",
    "linear-gradient(160deg,#373b44,#4286f4)",
    "linear-gradient(160deg,#f093fb,#f5576c)",
]
_EMOJIS  = ["📚","📖","✨","🌟","💡","🎯","🔥","💎","🌙","⭐","🏆","🎨","🌿","💫","🦋"]
_T10_BADGE = {
    1: "linear-gradient(135deg,#FFD700,#FFA500)",
    2: "linear-gradient(135deg,#C0C0C0,#9E9E9E)",
    3: "linear-gradient(135deg,#CD7F32,#A0522D)",
}

def _grad(t: str)  -> str: return _GRADS [abs(hash(t)) % len(_GRADS)]
def _emoji(t: str) -> str: return _EMOJIS[abs(hash(t)) % len(_EMOJIS)]
def _cover(t: str) -> str:
    return f"https://covers.openlibrary.org/b/title/{urllib.parse.quote(str(t))}-M.jpg"
def _stars(r: float) -> str:
    n = int(round(float(r))); return "★" * n + "☆" * (5 - n)
def _fmt_n(n) -> str:
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SHARED BOOK-COVER COMPONENT
#    Renders a 3D book with spine + gradient fallback when no image loads.
# ─────────────────────────────────────────────────────────────────────────────

def _cover_html(title: str, author: str,
                w: int = 110, h: int = 158, spine: bool = True) -> str:
    """
    Returns an HTML snippet: a 3-D book cover (spine + front face).
    Falls back to a gradient + emoji placeholder when the remote image fails.
    The onerror / onload pattern uses this.nextElementSibling so no unique IDs
    are needed.
    """
    g   = _grad(title)
    em  = _emoji(title)
    u   = _cover(title)
    sw  = max(8, w // 11) if spine else 0
    rr  = "0 6px 6px 0" if spine else "6px"
    fst = max(9,  w // 12)
    fsa = max(8,  w // 14)
    ems = max(1.5, w / 56)
    st  = title[:22]  + "…" if len(title)  > 22 else title
    sa  = author[:18] + "…" if len(author) > 18 else author

    spine_div = (
        f'<div style="width:{sw}px;height:{h}px;background:{g};'
        f'filter:brightness(.56);border-radius:3px 0 0 3px;flex-shrink:0;"></div>'
    ) if spine else ""

    return f'''\
<div style="display:flex;flex-shrink:0;
            filter:drop-shadow(6px 10px 22px rgba(0,0,0,.30));">
  {spine_div}
  <div style="position:relative;width:{w}px;height:{h}px;background:{g};
              border-radius:{rr};overflow:hidden;flex-shrink:0;">
    <img src="{u}" alt=""
         style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;"
         onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"
         onload="if(this.naturalWidth<10){{this.style.display='none';this.nextElementSibling.style.display='flex';}}"/>
    <div style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;
                flex-direction:column;justify-content:space-between;padding:12px 10px;">
      <div style="font-size:{ems:.1f}em;filter:drop-shadow(0 2px 6px rgba(0,0,0,.3));">{em}</div>
      <div>
        <div style="color:rgba(255,255,255,.95);font-size:{fst}px;font-weight:700;
                    line-height:1.3;">{st}</div>
        <div style="color:rgba(255,255,255,.68);font-size:{fsa}px;margin-top:2px;">{sa}</div>
      </div>
    </div>
  </div>
</div>'''

# ─────────────────────────────────────────────────────────────────────────────
# 4. HERO SECTION  — "Reading Now" featured book
# ─────────────────────────────────────────────────────────────────────────────

def render_hero(data: pd.DataFrame) -> str:
    """Large featured-book hero card (top book by popularity_score)."""
    if data.empty:
        return ""
    row  = data.nlargest(1, "popularity_score").iloc[0]
    t    = str(row["title"])
    a    = str(row["authors"])
    d    = str(row.get("description", "")) or "No description available."
    r    = float(row.get("average_rating", 0))
    yr   = int(row.get("published_year", 0))
    cat  = str(row.get("categories", "")).split("·")[0].strip()
    g    = _grad(t)
    cover = _cover_html(t, a, w=135, h=195, spine=True)
    yr_str = str(yr) if yr > 0 else ""

    return f'''\
<div style="background:{g};border-radius:22px;padding:24px;margin-bottom:18px;
            display:flex;gap:20px;align-items:flex-start;position:relative;
            overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.20);">
  <!-- decorative orbs -->
  <div style="position:absolute;top:-40px;right:-40px;width:210px;height:210px;
              background:rgba(255,255,255,.06);border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;bottom:-70px;left:42%;width:260px;height:260px;
              background:rgba(255,255,255,.04);border-radius:50%;pointer-events:none;"></div>
  <!-- book cover -->
  <div style="z-index:1;flex-shrink:0;">{cover}</div>
  <!-- info -->
  <div style="flex:1;min-width:0;z-index:1;padding-top:4px;">
    <div style="display:inline-block;background:rgba(255,255,255,.18);color:#fff;
                font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;
                letter-spacing:1px;text-transform:uppercase;margin-bottom:14px;
                backdrop-filter:blur(10px);">📖 Reading Now</div>
    <div style="font-size:22px;font-weight:800;color:#fff;line-height:1.2;
                margin-bottom:6px;letter-spacing:-.3px;">{t}</div>
    <div style="font-size:13px;color:rgba(255,255,255,.75);margin-bottom:10px;">by {a}</div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
      <span style="color:#FFD700;font-size:14px;">{_stars(r)}</span>
      <span style="color:rgba(255,255,255,.9);font-size:13px;font-weight:600;">{r:.1f}</span>
      {('<span style="color:rgba(255,255,255,.55);font-size:12px;">·&nbsp;' + yr_str + '</span>') if yr_str else ''}
      {('<span style="background:rgba(255,255,255,.15);color:rgba(255,255,255,.85);font-size:11px;'
        'padding:2px 10px;border-radius:20px;">' + cat[:24] + '</span>') if cat else ''}
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,.78);line-height:1.6;
                margin-bottom:18px;">{truncate(d, 240)}</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div style="background:rgba(255,255,255,.16);color:#fff;
                  border:1px solid rgba(255,255,255,.28);font-size:12px;font-weight:600;
                  padding:8px 18px;border-radius:25px;cursor:pointer;
                  backdrop-filter:blur(10px);">+ Add to Library</div>
      <div style="background:#fff;color:#1D1D1F;font-size:12px;font-weight:700;
                  padding:8px 20px;border-radius:25px;cursor:pointer;
                  box-shadow:0 3px 12px rgba(0,0,0,.18);">Get Book</div>
    </div>
  </div>
</div>'''

# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP 10 SECTION  — 3-D book grid + expandable detail panel
# ─────────────────────────────────────────────────────────────────────────────

def _t10_card(i: int, row: pd.Series) -> str:
    t   = str(row["title"])
    a   = str(row["authors"])
    g   = _grad(t)
    em  = _emoji(t)
    u   = _cover(t)
    r   = float(row.get("average_rating", 0))
    cat = str(row.get("categories", "")).split("·")[0].strip()[:18]
    badge = _T10_BADGE.get(i + 1, "rgba(0,0,0,.55)")
    return f'''\
<div class="t10-card" data-card="{i}" onclick="selectBook({i})">
  <div class="t10-rank" style="background:{badge};">{i+1}</div>
  <div class="t10-cover" style="background:{g};">
    <img class="t10-cover-img" src="{u}" alt=""
         onload="if(this.naturalWidth<10){{this.style.display='none';this.nextElementSibling.style.display='block';}}"
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="t10-emoji" style="display:none;">{em}</span>
    <div class="t10-foot">
      <p class="t10-ctitle">{t}</p>
      <p class="t10-cauthor">{a[:30]}</p>
    </div>
  </div>
  <div class="t10-card-meta">
    <span class="t10-cstars">{_stars(r)}</span>
    <span class="t10-cgenre">{cat}</span>
  </div>
</div>'''


def _t10_detail(i: int, row: pd.Series) -> str:
    t   = str(row["title"])
    a   = str(row["authors"])
    g   = _grad(t)
    em  = _emoji(t)
    u   = _cover(t)
    r   = float(row.get("average_rating", 0))
    yr  = int(row.get("published_year", 0))
    cat = str(row.get("categories", "")).strip()
    d   = str(row.get("description", "")) or "No description available."
    cnt = int(row.get("ratings_count", 0))
    display = "flex" if i == 0 else "none"
    yr_str  = str(yr) if yr > 0 else "—"
    cnt_str = _fmt_n(cnt) if cnt > 0 else ""

    return f'''\
<div class="t10-detail" data-dp="{i}" style="display:{display};">
  <div class="t10-det-cover" style="background:{g};">
    <span class="t10-det-rank">#{i+1}</span>
    <img class="t10-det-img" src="{u}" alt=""
         onload="if(this.naturalWidth<10){{this.style.display='none';this.nextElementSibling.style.display='block';}}"
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="t10-det-emoji" style="display:none;">{em}</span>
  </div>
  <div class="t10-det-info">
    <h2 class="t10-det-title">{t}</h2>
    <p class="t10-det-author">by {a}</p>
    <div class="t10-chips">
      {('<span class="t10-chip">' + cat + '</span>') if cat else ''}
      <span class="t10-chip">📅 {yr_str}</span>
      {('<span class="t10-chip">👥 ' + cnt_str + ' ratings</span>') if cnt_str else ''}
    </div>
    <p class="t10-det-stars">{_stars(r)}<span class="t10-det-rnum">&nbsp;{r:.1f} / 5.0</span></p>
    <p class="t10-det-desc">{truncate(d, 450)}</p>
  </div>
</div>'''


def render_top10(data: pd.DataFrame) -> str:
    """Render the full Top 10 section: grid + interactive detail panel."""
    if data.empty:
        return ""
    top = (data[data["average_rating"] > 0]
           .nlargest(10, "popularity_score")
           .reset_index(drop=True))

    grid    = "".join(_t10_card(i, row)   for i, (_, row) in enumerate(top.iterrows()))
    details = "".join(_t10_detail(i, row) for i, (_, row) in enumerate(top.iterrows()))

    return f'''\
<div class="bw-section">
  <div class="bw-section-header">
    <div>
      <p class="bw-section-title">⭐ Top 10 Books</p>
      <p class="bw-section-sub">Ranked by popularity &amp; rating</p>
    </div>
    <span class="bw-see-all">See All →</span>
  </div>

  <div class="t10-grid">{grid}</div>

  <div class="t10-det-wrap">
    <p class="t10-det-heading">Book Details</p>
    {details}
  </div>
</div>

<script>
(function() {{
  function selectBook(idx) {{
    document.querySelectorAll('[data-dp]').forEach(function(el)   {{ el.style.display = 'none';          }});
    document.querySelectorAll('[data-card]').forEach(function(el) {{ el.classList.remove('t10-selected'); }});
    var dp   = document.querySelector('[data-dp="'   + idx + '"]');
    var card = document.querySelector('[data-card="' + idx + '"]');
    if (dp)   dp.style.display   = 'flex';
    if (card) card.classList.add('t10-selected');
    if (dp)   dp.scrollIntoView({{ behavior:'smooth', block:'nearest' }});
  }}
  window.selectBook = selectBook;
  document.addEventListener('DOMContentLoaded', function() {{ selectBook(0); }});
}})();
</script>'''

# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMMENDED FOR YOU  — horizontal scroll of auto-suggestions
# ─────────────────────────────────────────────────────────────────────────────

def render_recommended_for_you(data: pd.DataFrame, vec, mat) -> str:
    """
    Generates recommendations based on the most-popular book so the home page
    always has personalised-looking content without user input.
    """
    if data.empty:
        return ""
    top_row   = data.nlargest(1, "popularity_score").iloc[0]
    top_title = str(top_row["title"])
    # recommender.py searches the clean_title column (lower-cased, no punctuation);
    # pass clean_title directly so special characters don't break the match.
    query = str(top_row.get("clean_title", top_title.lower())).strip() or top_title.lower()
    recs = recommend_by_title(
        query_title=query, df=data, tfidf_matrix=mat, vectorizer=vec,
        n=9, text_weight=0.65,
    )
    if recs is None or recs.empty:
        return ""
    recs = recs[recs["Title"] != top_title].head(7)

    cards = ""
    for _, row in recs.iterrows():
        t = str(row["Title"])
        a = str(row["Authors"])
        r = float(row.get("Rating", 0))
        cover = _cover_html(t, a, w=92, h=132, spine=True)
        cards += f'''\
<div style="flex-shrink:0;width:108px;cursor:pointer;text-align:center;" title="{t}">
  {cover}
  <div style="margin-top:8px;">
    <div style="font-size:12px;font-weight:600;color:#1D1D1F;line-height:1.3;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      {t[:17]}{"…" if len(t)>17 else ""}</div>
    <div style="font-size:11px;color:#8E8E93;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;">
      {a[:15]}{"…" if len(a)>15 else ""}</div>
    <div style="color:#FF9F0A;font-size:11px;margin-top:2px;">★ {r:.1f}</div>
  </div>
</div>'''

    return f'''\
<div class="bw-section">
  <div class="bw-section-header">
    <div>
      <p class="bw-section-title">✨ Recommended For You</p>
      <p class="bw-section-sub">Because you might enjoy
        <em>{truncate(top_title, 40)}</em></p>
    </div>
    <span class="bw-see-all">See All →</span>
  </div>
  <div class="bw-hscroll">{cards}</div>
</div>'''

# ─────────────────────────────────────────────────────────────────────────────
# 7. NEW & TRENDING  — recently published, high-rated titles
# ─────────────────────────────────────────────────────────────────────────────

def render_new_trending(data: pd.DataFrame) -> str:
    """Shows books from the last decade with the highest popularity score."""
    if data.empty:
        return ""
    max_yr  = int(data["published_year"].max())
    cutoff  = max(max_yr - 10, 2000)
    recent  = (data[(data["published_year"] >= cutoff) & (data["average_rating"] >= 3.5)]
               .nlargest(8, "popularity_score")
               .reset_index(drop=True))
    if recent.empty:
        recent = data.nlargest(8, "popularity_score").reset_index(drop=True)

    cards = ""
    for _, row in recent.iterrows():
        t  = str(row["title"])
        a  = str(row["authors"])
        r  = float(row.get("average_rating", 0))
        yr = int(row.get("published_year", 0))
        cover = _cover_html(t, a, w=92, h=132, spine=True)
        yr_lbl = str(yr) if yr > 0 else ""
        cards += f'''\
<div style="flex-shrink:0;width:108px;cursor:pointer;text-align:center;" title="{t}">
  {cover}
  <div style="margin-top:8px;">
    <div style="font-size:12px;font-weight:600;color:#1D1D1F;line-height:1.3;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      {t[:17]}{"…" if len(t)>17 else ""}</div>
    <div style="font-size:11px;color:#8E8E93;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;">
      {a[:15]}{"…" if len(a)>15 else ""}</div>
    <div style="font-size:10px;color:#8E8E93;margin-top:2px;">
      {"★ " + f"{r:.1f}" if r > 0 else ""}{" · " + yr_lbl if yr_lbl else ""}</div>
  </div>
</div>'''

    return f'''\
<div class="bw-section">
  <div class="bw-section-header">
    <div>
      <p class="bw-section-title">🔥 New &amp; Trending</p>
      <p class="bw-section-sub">Recently popular reads</p>
    </div>
    <span class="bw-see-all">See All →</span>
  </div>
  <div class="bw-hscroll">{cards}</div>
</div>'''

# ─────────────────────────────────────────────────────────────────────────────
# 8. RECOMMENDATION RESULT CARDS  (shared by Find Similar / Describe / Browse)
# ─────────────────────────────────────────────────────────────────────────────

def _results_to_html(results) -> str:
    """
    Converts a recommendations DataFrame into an Apple Books-style grid.
    DataFrame must have columns: Title, Authors, Category, Rating, Year,
    Score, Description.
    """
    if results is None or results.empty:
        return '''\
<div style="text-align:center;padding:60px 20px;
            font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
  <div style="font-size:4em;margin-bottom:12px;">📚</div>
  <p style="font-size:1.1em;color:#86868B;margin:0;">
    No books found. Try a different title or relax your filters.
  </p>
</div>'''

    cards = []
    for _, row in results.iterrows():
        t       = str(row["Title"])
        a       = str(row["Authors"])
        cat     = str(row["Category"])
        r       = float(row["Rating"])
        yr      = int(row["Year"])
        score   = float(row["Score"])
        desc    = str(row["Description"])
        g       = _grad(t)
        u       = _cover(t)
        stars   = _stars(r)
        yr_str  = str(yr) if yr > 0 else ""
        meta    = cat + (f" · {yr_str}" if yr_str else "")
        snippet = truncate(desc, 220)

        cards.append(f'''\
<div class="rec-card">
  <div class="rec-scene">
    <div class="rec-wrap">
      <img class="rec-img" src="{u}" alt=""
           onload="if(this.naturalWidth<10){{this.style.display='none';this.nextElementSibling.style.display='flex';}}"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" />
      <div class="rec-ph" style="background:{g};">
        <span class="rec-initial">{t[0].upper() if t else "?"}</span>
      </div>
    </div>
  </div>
  <div class="rec-info">
    <span class="rec-score">{score:.2f}</span>
    <p class="rec-title">{t}</p>
    <p class="rec-author">{a}</p>
    <p class="rec-stars">{stars} <span class="rec-rnum">{r:.1f}</span></p>
    <details class="rec-details">
      <summary class="rec-summary">More info ›</summary>
      <p class="rec-desc">{snippet}</p>
      <p class="rec-meta">{meta}</p>
    </details>
  </div>
</div>''')

    return '<div class="rec-grid">' + "".join(cards) + "</div>"

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVENT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_english(text: str) -> str:
    """Optionally translate to English using deep_translator (if installed)."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result or text
    except Exception:
        return text


def search_by_title(title, n, min_year, max_year, min_rating, category, text_weight):
    if not str(title).strip():
        return "<p class='rec-hint'>Enter a book title above.</p>"
    return _results_to_html(recommend_by_title(
        query_title=_to_english(str(title)), df=df,
        tfidf_matrix=tfidf_matrix, vectorizer=vectorizer,
        n=int(n), min_year=int(min_year), max_year=int(max_year),
        min_rating=float(min_rating), category=str(category),
        text_weight=float(text_weight),
    ))


def search_by_query(query, n, min_year, max_year, min_rating, category, text_weight):
    if not str(query).strip():
        return "<p class='rec-hint'>Describe what you want to read above.</p>"
    return _results_to_html(recommend_by_query(
        query_text=_to_english(str(query)), df=df,
        tfidf_matrix=tfidf_matrix, vectorizer=vectorizer,
        n=int(n), min_year=int(min_year), max_year=int(max_year),
        min_rating=float(min_rating), category=str(category),
        text_weight=float(text_weight),
    ))


def browse_category(category, n, sort_by, min_rating, min_year, max_year):
    results = browse_by_category(
        category=str(category), df=df, n=int(n), sort_by=str(sort_by),
        min_rating=float(min_rating), min_year=int(min_year), max_year=int(max_year),
    )
    return _results_to_html(results)


def handle_upload(file):
    """
    Replace the global dataset with a user-uploaded CSV, rebuild the TF-IDF
    model, and re-render all home-page sections.

    Required CSV columns (flexible naming):
        title / book_title
        authors / author / book_author
        categories / genre / category
        description / summary / synopsis
        average_rating / rating
        ratings_count / ratings / num_ratings
        published_year   (optional)

    Missing columns are filled with sensible defaults.
    """
    global df, vectorizer, tfidf_matrix, CATEGORIES, ALL_TITLES

    if file is None:
        return (
            "*No file selected — still using the default dataset.*",
            render_hero(df),
            render_top10(df),
            render_recommended_for_you(df, vectorizer, tfidf_matrix),
            render_new_trending(df),
        )

    try:
        new_df = pd.read_csv(file.name)

        col_map = {
            "book_title": "title", "book_author": "authors", "author": "authors",
            "genre": "categories", "category": "categories",
            "rating": "average_rating", "ratings": "ratings_count",
            "num_ratings": "ratings_count", "summary": "description",
            "synopsis": "description", "overview": "description",
        }
        new_df = new_df.rename(columns={k: v for k, v in col_map.items()
                                         if k in new_df.columns})

        for col, default in [
            ("title",          "Unknown"),
            ("authors",        "Unknown"),
            ("categories",     "General"),
            ("description",    ""),
            ("average_rating", 3.5),
            ("ratings_count",  100),
            ("published_year", 2000),
        ]:
            if col not in new_df.columns:
                new_df[col] = default

        new_df["average_rating"] = (
            pd.to_numeric(new_df["average_rating"], errors="coerce").fillna(3.0).clip(0, 5)
        )
        new_df["ratings_count"]  = (
            pd.to_numeric(new_df["ratings_count"],  errors="coerce").fillna(0)
        )
        new_df["published_year"] = (
            pd.to_numeric(new_df["published_year"], errors="coerce").fillna(2000).astype(int)
        )

        # Recompute popularity_score
        log_rc  = np.log1p(new_df["ratings_count"])
        max_log = log_rc.max() or 1
        new_df["popularity_score"] = (
            0.6 * (log_rc / max_log) + 0.4 * (new_df["average_rating"] / 5.0)
        ) * 10

        # Build combined_text for TF-IDF
        new_df["combined_text"] = (
            new_df["title"].astype(str) + " " +
            new_df["authors"].astype(str) + " " +
            new_df["categories"].astype(str) + " " +
            new_df["description"].astype(str)
        )
        # Columns that recommender.py expects
        for col in ("clean_title", "clean_authors", "clean_categories", "clean_description"):
            if col not in new_df.columns:
                src = col.replace("clean_", "")
                new_df[col] = new_df.get(src, pd.Series([""] * len(new_df))).str.lower().fillna("")

        new_df = new_df.reset_index(drop=True)

        from recommender import build_tfidf_matrix as _btm, get_categories as _gc
        new_vec, new_mat = _btm(new_df)
        new_cats   = _gc(new_df)
        new_titles = sorted(new_df["title"].dropna().unique().tolist())

        df            = new_df
        vectorizer    = new_vec
        tfidf_matrix  = new_mat
        CATEGORIES    = new_cats
        ALL_TITLES    = new_titles

        msg = f"✅ Loaded **{len(df):,}** books from your CSV. Home page updated."
    except Exception as exc:
        msg = f"❌ Error loading file: `{exc}`. Previous dataset unchanged."

    return (
        msg,
        render_hero(df),
        render_top10(df),
        render_recommended_for_you(df, vectorizer, tfidf_matrix),
        render_new_trending(df),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. CSS  — Apple Books / iBooks visual style
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ═══════════════════════════════════════════════
   BookWise  —  Apple Books-Inspired Stylesheet
═══════════════════════════════════════════════ */

/* ── Google Font import (optional, falls back to system font) ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #F2F2F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter",
                 "Helvetica Neue", Arial, sans-serif !important;
    color: #1D1D1F;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Suppress default Gradio chrome ── */
footer { display: none !important; }
.gr-form, .gr-panel  { border: none !important; background: transparent !important;
                        box-shadow: none !important; padding: 0 !important; }
.block, .gap         { background: transparent !important; border: none !important;
                        box-shadow: none !important; }

/* ── Inputs & Selects ── */
input[type="text"], input[type="number"], textarea, select {
    border-radius: 10px !important;
    border: 1.5px solid #E5E5EA !important;
    background: #FAFAFA !important;
    font-size: 0.94em !important;
    padding: 10px 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
    font-family: inherit !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0,122,255,.12) !important;
    outline: none !important;
}
label {
    font-size: .85em !important;
    font-weight: 600 !important;
    color: #3C3C43 !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg,#007AFF,#5856D6) !important;
    border: none !important;
    border-radius: 980px !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    padding: 10px 28px !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(0,122,255,.36) !important;
    transition: all .22s !important;
    font-family: inherit !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(0,122,255,.46) !important;
}
button.secondary {
    background: #F2F2F7 !important;
    border: 1.5px solid #E5E5EA !important;
    border-radius: 980px !important;
    color: #007AFF !important;
    font-weight: 600 !important;
    font-family: inherit !important;
}

/* ── Tabs ── */
.tab-nav {
    background: #fff !important;
    border-bottom: 1px solid #E5E5EA !important;
    border-radius: 0 !important;
    padding: 0 8px !important;
}
.tab-nav button {
    font-size: .90em !important;
    font-weight: 500 !important;
    color: #6E6E73 !important;
    border-radius: 0 !important;
    border-bottom: 2.5px solid transparent !important;
    padding: 12px 18px !important;
    transition: color .18s, border-color .18s !important;
    font-family: inherit !important;
}
.tab-nav button.selected {
    color: #007AFF !important;
    font-weight: 600 !important;
    border-bottom-color: #007AFF !important;
}

/* ── BookWise section cards ── */
.bw-section {
    background: #fff;
    border-radius: 20px;
    padding: 22px 22px 18px;
    margin-bottom: 18px;
    box-shadow: 0 2px 20px rgba(0,0,0,.065);
}
.bw-section-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 18px;
}
.bw-section-title { margin: 0; font-size: 1.15em; font-weight: 700; color: #1D1D1F; }
.bw-section-sub   { margin: 3px 0 0; font-size: .82em; color: #8E8E93; }
.bw-see-all       { font-size: .84em; color: #007AFF; font-weight: 600;
                     cursor: pointer; white-space: nowrap; margin-top: 2px; }

/* ── Horizontal scroll strips ── */
.bw-hscroll { display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px;
               scrollbar-width: none; -ms-overflow-style: none; }
.bw-hscroll::-webkit-scrollbar { display: none; }

/* ── Filter label ── */
.filter-label {
    font-size: 1em !important; font-weight: 700 !important;
    color: #1D1D1F !important; margin: 0 0 12px !important;
}

/* ══════════════════════════════════════════════════
   TOP 10  (t10- prefix)
══════════════════════════════════════════════════ */

.t10-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px 16px;
    margin-bottom: 24px;
}

/* 3-D book card */
.t10-card {
    cursor: pointer;
    border-radius: 4px 12px 12px 4px;
    overflow: hidden;
    box-shadow: 6px 8px 22px rgba(0,0,0,.20), -2px 0 5px rgba(0,0,0,.10);
    transition: transform .28s ease, box-shadow .28s ease;
    position: relative;
    transform: perspective(600px) rotateY(-4deg);
    background: #fff;
    outline: 3px solid transparent;
    outline-offset: 3px;
}
.t10-card:hover, .t10-selected {
    transform: perspective(600px) rotateY(-14deg) translateX(-5px) scale(1.04) !important;
    box-shadow: 14px 16px 40px rgba(0,0,0,.28), -4px 0 10px rgba(0,0,0,.15) !important;
}
.t10-selected { outline: 3px solid #007AFF !important; }

/* rank badge */
.t10-rank {
    position: absolute; top: 8px; left: 8px; z-index: 2;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .73em; font-weight: 700; color: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,.30);
}

/* cover area */
.t10-cover      { position: relative; padding-top: 145%; overflow: hidden; }
.t10-cover-img  { position: absolute; top: 0; left: 0;
                   width: 100%; height: 100%; object-fit: cover; }
.t10-emoji {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -60%);
    font-size: 3.2em;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,.25));
    pointer-events: none;
}
.t10-foot {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, rgba(0,0,0,.72) 0%, transparent 100%);
    padding: 28px 10px 10px;
}
.t10-ctitle {
    margin: 0; font-size: .77em; font-weight: 700; color: #fff; line-height: 1.25;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.t10-cauthor { margin: 2px 0 0; font-size: .67em; color: rgba(255,255,255,.75);
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* card footer strip */
.t10-card-meta { display: flex; align-items: center; justify-content: space-between;
                 padding: 6px 10px 7px; background: #fff; }
.t10-cstars    { color: #FF9F0A; font-size: .70em; letter-spacing: .4px; }
.t10-cgenre    { font-size: .64em; color: #8E8E93; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; max-width: 58%; }

/* detail panel */
.t10-det-wrap {
    background: #F8F8FA; border-radius: 16px;
    padding: 20px 22px 22px; margin-top: 4px;
}
.t10-det-heading { margin: 0 0 14px; font-size: .78em; font-weight: 600;
                   color: #8E8E93; text-transform: uppercase; letter-spacing: .9px; }
.t10-detail      { display: none; align-items: flex-start; gap: 26px; }

.t10-det-cover {
    flex-shrink: 0; width: 160px; height: 236px;
    border-radius: 4px 14px 14px 4px;
    display: flex; align-items: center; justify-content: center;
    position: relative; overflow: hidden;
    box-shadow: 8px 12px 32px rgba(0,0,0,.28), -3px 0 8px rgba(0,0,0,.15);
    transform: perspective(600px) rotateY(-8deg);
}
.t10-det-rank {
    position: absolute; top: 10px; left: 12px; z-index: 2;
    font-size: 1em; font-weight: 800;
    color: rgba(255,255,255,.9); text-shadow: 0 2px 6px rgba(0,0,0,.4);
}
.t10-det-img   { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                  object-fit: cover; border-radius: 4px 14px 14px 4px; }
.t10-det-emoji { font-size: 5em; filter: drop-shadow(0 6px 12px rgba(0,0,0,.3)); }

.t10-det-info   { flex: 1; padding-top: 4px; }
.t10-det-title  { margin: 0 0 4px; font-size: 1.5em; font-weight: 700;
                   color: #1D1D1F; line-height: 1.2; letter-spacing: -.4px; }
.t10-det-author { margin: 0 0 12px; font-size: .94em; color: #6E6E73; font-weight: 500; }
.t10-chips      { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
.t10-chip       { background: #F2F2F7; color: #3A3A3C; border-radius: 980px;
                  padding: 4px 13px; font-size: .78em; font-weight: 500; }
.t10-det-stars  { margin: 0 0 14px; font-size: 1em; color: #FF9F0A; }
.t10-det-rnum   { color: #3A3A3C; font-size: .87em; font-weight: 600; }
.t10-det-desc   { margin: 0; font-size: .93em; color: #3A3A3C; line-height: 1.62; }

/* ══════════════════════════════════════════════════
   RECOMMENDATION GRID  (rec- prefix)
══════════════════════════════════════════════════ */

.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(165px, 1fr));
    gap: 28px 20px;
    padding: 12px 4px 24px;
}
.rec-card  { display: flex; flex-direction: column; align-items: center; }

.rec-scene { width: 100%; perspective: 900px; margin-bottom: 12px; }
.rec-wrap  {
    position: relative; padding-top: 148%;
    transform-style: preserve-3d;
    transition: transform .35s ease;
}
.rec-card:hover .rec-wrap {
    transform: rotateY(-14deg) translateX(-4px) scale(1.03);
}
.rec-img, .rec-ph {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    border-radius: 3px 10px 10px 3px;
    box-shadow: 8px 10px 30px rgba(0,0,0,.28), -3px 0 6px rgba(0,0,0,.15),
                inset -3px 0 8px rgba(0,0,0,.10);
}
.rec-img     { object-fit: cover; }
.rec-ph      { display: none; align-items: center; justify-content: center; }
.rec-initial { font-size: 3.8em; font-weight: 700; color: rgba(255,255,255,.92);
               text-shadow: 0 2px 8px rgba(0,0,0,.25); }

.rec-info    { width: 100%; text-align: center; padding: 0 2px; }
.rec-score   { display: inline-block; background: rgba(0,122,255,.10); color: #007AFF;
               border-radius: 6px; padding: 1px 8px; font-size: .70em; font-weight: 700;
               margin-bottom: 5px; }
.rec-title   { margin: 0 0 3px; font-size: .86em; font-weight: 600; color: #1D1D1F;
               line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
               -webkit-box-orient: vertical; overflow: hidden; }
.rec-author  { margin: 0 0 4px; font-size: .76em; color: #6E6E73;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rec-stars   { margin: 0 0 5px; font-size: .8em; color: #FF9F0A; }
.rec-rnum    { color: #8E8E93; font-size: .88em; }
.rec-details { text-align: left; width: 100%; }
.rec-summary { font-size: .75em; color: #007AFF; cursor: pointer;
               list-style: none; -webkit-appearance: none; text-align: center; }
.rec-desc    { margin: 6px 0 0; font-size: .75em; color: #3A3A3C; line-height: 1.46; }
.rec-meta    { margin: 4px 0 0; font-size: .71em; color: #8E8E93; }
.rec-hint    { padding: 12px 0; color: #8E8E93; font-size: .92em; }

/* ── Scrollbar ── */
::-webkit-scrollbar        { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: transparent; }
::-webkit-scrollbar-thumb  { background: #C7C7CC; border-radius: 3px; }

/* ── Upload section ── */
.upload-box {
    background: #fff; border-radius: 20px; padding: 28px;
    box-shadow: 0 2px 20px rgba(0,0,0,.065); text-align: center;
}

/* ── Responsive ── */
@media (max-width: 960px) {
    .t10-grid { grid-template-columns: repeat(4, 1fr) !important; }
}
@media (max-width: 720px) {
    .t10-grid { grid-template-columns: repeat(3, 1fr) !important; }
}
@media (max-width: 540px) {
    .t10-grid   { grid-template-columns: repeat(2, 1fr) !important; }
    .t10-detail { flex-direction: column !important; }
    .t10-det-cover { width: 130px !important; height: 192px !important; transform: none !important; }
    .rec-grid   { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)) !important; }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# 11. STATIC HTML FRAGMENTS
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = '''\
<div style="background:#fff;border-bottom:1px solid #E5E5EA;
            padding:14px 24px;display:flex;justify-content:space-between;
            align-items:center;margin-bottom:16px;
            position:sticky;top:0;z-index:200;
            box-shadow:0 1px 0 rgba(0,0,0,.06);">
  <!-- Logo + wordmark -->
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:42px;height:42px;
                background:linear-gradient(135deg,#007AFF,#5856D6);
                border-radius:12px;display:flex;align-items:center;
                justify-content:center;font-size:24px;
                box-shadow:0 4px 14px rgba(0,122,255,.35);">📚</div>
    <div>
      <div style="font-size:1.38em;font-weight:800;color:#1D1D1F;
                  letter-spacing:-.5px;line-height:1;">BookWise</div>
      <div style="font-size:.76em;color:#8E8E93;margin-top:1px;">
        Discover your next favourite read</div>
    </div>
  </div>
  <!-- Avatar -->
  <div style="width:36px;height:36px;
              background:linear-gradient(135deg,#30D158,#34C759);
              border-radius:50%;display:flex;align-items:center;
              justify-content:center;font-size:18px;
              box-shadow:0 3px 10px rgba(52,199,89,.35);">👤</div>
</div>'''

_BOTTOM_NAV = '''\
<div style="background:rgba(255,255,255,.92);
            backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
            border-top:1px solid rgba(0,0,0,.08);
            padding:10px 0 6px;
            display:flex;justify-content:space-around;
            margin-top:20px;border-radius:20px 20px 0 0;
            box-shadow:0 -4px 20px rgba(0,0,0,.06);">
  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;">
    <span style="font-size:22px;">📖</span>
    <span style="font-size:10px;color:#007AFF;font-weight:600;">Reading Now</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;">
    <span style="font-size:22px;">📚</span>
    <span style="font-size:10px;color:#8E8E93;">Library</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;">
    <span style="font-size:22px;">🏪</span>
    <span style="font-size:10px;color:#8E8E93;">Book Store</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;">
    <span style="font-size:22px;">🎧</span>
    <span style="font-size:10px;color:#8E8E93;">Audiobooks</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;">
    <span style="font-size:22px;">🔍</span>
    <span style="font-size:10px;color:#8E8E93;">Search</span>
  </div>
</div>'''

_FOOTER = '''\
<p style="text-align:center;color:#8E8E93;font-size:.76em;margin:8px 0 0;
          font-family:-apple-system,sans-serif;">
  BookWise &nbsp;·&nbsp; EADA Business School
  &nbsp;·&nbsp; Deep Learning &amp; LLM Project
</p>'''

def _filter_label(text: str) -> str:
    return (f'<p class="filter-label">{text}</p>')

# ─────────────────────────────────────────────────────────────────────────────
# 12. GRADIO BLOCKS LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="BookWise", css=CSS) as demo:

    # ── App-wide header ────────────────────────────────────────────────────
    gr.HTML(_HEADER)

    with gr.Tabs(elem_id="main-tabs"):

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — HOME
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📖  Home"):
            hero_html     = gr.HTML(render_hero(df))
            top10_html    = gr.HTML(render_top10(df))
            rfy_html      = gr.HTML(
                render_recommended_for_you(df, vectorizer, tfidf_matrix)
            )
            trending_html = gr.HTML(render_new_trending(df))

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — FIND SIMILAR BOOKS  (title-based content recommendation)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍  Find Similar"):
            with gr.Row(equal_height=False):
                # ── Sidebar filters ──
                with gr.Column(scale=1, min_width=220):
                    gr.HTML(_filter_label("Filters"))
                    fs_n       = gr.Slider(1, 20, value=8, step=1, label="Results")
                    fs_year_a  = gr.Number(value=0, precision=0,
                                           label="From year  (0 = any)")
                    fs_year_b  = gr.Number(value=0, precision=0,
                                           label="To year    (0 = any)")
                    fs_rating  = gr.Slider(0.0, 5.0, value=0.0, step=0.1,
                                           label="Min rating")
                    fs_cat     = gr.Dropdown(choices=CATEGORIES, value="All",
                                             label="Category")
                    fs_weight  = gr.Slider(0.0, 1.0, value=0.70, step=0.05,
                                           label="Text weight  [0 popular · 1 content]")
                # ── Main panel ──
                with gr.Column(scale=3):
                    fs_title = gr.Textbox(
                        label="Book title",
                        placeholder="e.g.  The Great Gatsby, Dune, Educated…",
                    )
                    fs_btn   = gr.Button("✨ Find Similar Books", variant="primary")
                    fs_out   = gr.HTML()

            _fs_inputs = [fs_title, fs_n, fs_year_a, fs_year_b,
                          fs_rating, fs_cat, fs_weight]
            fs_btn.click(search_by_title, inputs=_fs_inputs, outputs=fs_out)
            fs_title.submit(search_by_title, inputs=_fs_inputs, outputs=fs_out)

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — BROWSE BY CATEGORY
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📂  Browse"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220):
                    gr.HTML(_filter_label("Filters"))
                    bc_cat    = gr.Dropdown(choices=CATEGORIES, value="All",
                                            label="Category")
                    bc_sort   = gr.Radio(choices=["popularity", "rating"],
                                         value="popularity", label="Sort by")
                    bc_n      = gr.Slider(1, 50, value=20, step=1, label="Results")
                    bc_rating = gr.Slider(0.0, 5.0, value=0.0, step=0.1,
                                          label="Min rating")
                    bc_year_a = gr.Number(value=0, precision=0,
                                          label="From year  (0 = any)")
                    bc_year_b = gr.Number(value=0, precision=0,
                                          label="To year    (0 = any)")
                    bc_btn    = gr.Button("Browse", variant="primary")
                with gr.Column(scale=3):
                    bc_out = gr.HTML(
                        "<p class='rec-hint'>Select a category and click Browse.</p>"
                    )

            _bc_inputs = [bc_cat, bc_n, bc_sort, bc_rating, bc_year_a, bc_year_b]
            bc_btn.click(browse_category, inputs=_bc_inputs, outputs=bc_out)
            bc_cat.change(browse_category, inputs=_bc_inputs, outputs=bc_out)

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — SEARCH BY DESCRIPTION  (free-text semantic query)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("💬  Describe"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220):
                    gr.HTML(_filter_label("Filters"))
                    desc_n      = gr.Slider(1, 20, value=8, step=1, label="Results")
                    desc_year_a = gr.Number(value=0, precision=0,
                                            label="From year  (0 = any)")
                    desc_year_b = gr.Number(value=0, precision=0,
                                            label="To year    (0 = any)")
                    desc_rating = gr.Slider(0.0, 5.0, value=0.0, step=0.1,
                                            label="Min rating")
                    desc_cat    = gr.Dropdown(choices=CATEGORIES, value="All",
                                              label="Category")
                    desc_weight = gr.Slider(0.0, 1.0, value=0.70, step=0.05,
                                            label="Text weight  [0 popular · 1 content]")
                with gr.Column(scale=3):
                    desc_query = gr.Textbox(
                        label="Describe what you want to read",
                        placeholder=(
                            "e.g.  a gripping thriller set in Victorian London "
                            "with dark secrets and a strong female lead"
                        ),
                        lines=3,
                    )
                    desc_btn = gr.Button("🔍 Search", variant="primary")
                    desc_out = gr.HTML()

            _desc_inputs = [desc_query, desc_n, desc_year_a, desc_year_b,
                             desc_rating, desc_cat, desc_weight]
            desc_btn.click(search_by_query, inputs=_desc_inputs, outputs=desc_out)
            desc_query.submit(search_by_query, inputs=_desc_inputs, outputs=desc_out)

        # ══════════════════════════════════════════════════════════════════
        # TAB 5 — UPLOAD YOUR OWN CSV DATASET
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⬆️  Upload"):
            with gr.Column(elem_classes=["upload-box"]):
                gr.HTML('''\
<div style="margin-bottom:22px;">
  <div style="font-size:2.8em;margin-bottom:10px;">📂</div>
  <h3 style="margin:0 0 8px;font-size:1.12em;font-weight:700;color:#1D1D1F;">
    Upload Your Book Dataset</h3>
  <p style="margin:0;font-size:.88em;color:#8E8E93;line-height:1.6;">
    Upload any CSV with book data. Supported columns (flexible naming):<br>
    <code>title</code> · <code>authors</code> · <code>categories</code> ·
    <code>description</code> · <code>average_rating</code> ·
    <code>ratings_count</code> · <code>published_year</code><br>
    Missing columns are filled with sensible defaults automatically.
  </p>
</div>''')
                up_file   = gr.File(label="Choose a CSV file", file_types=[".csv"])
                up_btn    = gr.Button("🚀 Load Dataset", variant="primary")
                up_status = gr.Markdown(
                    "*The app already uses the cleaned dataset. "
                    "Upload a CSV to swap it out.*"
                )

            # Upload updates the Home tab sections live
            _up_outputs = [up_status, hero_html, top10_html, rfy_html, trending_html]
            up_btn.click(handle_upload, inputs=[up_file], outputs=_up_outputs)

    # ── Bottom navigation bar (decorative, iOS-style) ──────────────────────
    gr.HTML(_BOTTOM_NAV)
    gr.HTML(_FOOTER)

# ─────────────────────────────────────────────────────────────────────────────
# 13. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # localhost — funciona directo en el navegador
        server_port=None,         # Gradio elige el primer puerto libre desde 7860
        share=False,              # True requiere internet; ponlo en True si lo necesitas
        show_error=True,
    )
