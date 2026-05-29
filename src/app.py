"""
Book Recommender — Combined App
  Tab 1: Top 10  (real data, iBooks style)
  Tab 2: Find similar books
  Tab 3: Search by description

Run from the project root:
    python src/app.py
"""
from __future__ import annotations

import os
import urllib.parse

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
# Load data once
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_books.csv")

df = load_data(DATA_PATH)
vectorizer, tfidf_matrix = build_tfidf_matrix(df)
CATEGORIES = get_categories(df)

# ---------------------------------------------------------------------------
# ── TOP 10 ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_T10_GRADS = [
    ("#1a1a2e", "#16213e"), ("#2d5016", "#52734d"), ("#7b2d8b", "#c06c84"),
    ("#c0392b", "#e74c3c"), ("#b7410e", "#f39c12"), ("#1565C0", "#42A5F5"),
    ("#7B3F00", "#D4A017"), ("#134e5e", "#71b280"), ("#0F2027", "#4286f4"),
    ("#360033", "#0b8793"),
]
_T10_EMOJIS = ["🏆", "⭐", "🌟", "📖", "💡", "🎯", "🔥", "✨", "💫", "🌙"]
_T10_BADGE  = {
    1: "linear-gradient(135deg,#FFD700,#FFA500)",
    2: "linear-gradient(135deg,#C0C0C0,#9E9E9E)",
    3: "linear-gradient(135deg,#CD7F32,#A0522D)",
}


def _build_top10(data):
    top = (data[data["average_rating"] > 0]
           .nlargest(10, "popularity_score")
           .reset_index(drop=True))
    books = []
    for i, (_, row) in enumerate(top.iterrows()):
        title = str(row["title"])
        books.append({
            "rank":   i + 1,
            "title":  title,
            "author": str(row["authors"]),
            "genre":  str(row["categories"]) if row["categories"] else "General",
            "year":   int(row["published_year"]) if row["published_year"] > 0 else 0,
            "rating": float(row["average_rating"]),
            "desc":   str(row["description"]) if row["description"] else "No description available.",
            "from":   _T10_GRADS[i][0],
            "to":     _T10_GRADS[i][1],
            "emoji":  _T10_EMOJIS[i],
            "cover":  f"https://covers.openlibrary.org/b/title/{urllib.parse.quote(title)}-M.jpg",
        })
    return books


def _t10_stars(r: float) -> str:
    f = int(round(r))
    return "★" * f + "☆" * (5 - f)


def _t10_card(i: int, b: dict) -> str:
    badge = _T10_BADGE.get(b["rank"], "rgba(0,0,0,0.45)")
    genre_short = b["genre"].split("·")[0].strip()
    return f"""
<div class="t10-card" data-card="{i}" onclick="selectBook({i})">
  <div class="t10-rank" style="background:{badge};">{b['rank']}</div>
  <div class="t10-cover" style="background:linear-gradient(160deg,{b['from']},{b['to']});">
    <img class="t10-cover-img" src="{b['cover']}" alt=""
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="t10-emoji" style="display:none;">{b['emoji']}</span>
    <div class="t10-foot">
      <p class="t10-ctitle">{b['title']}</p>
      <p class="t10-cauthor">{b['author']}</p>
    </div>
  </div>
  <div class="t10-card-meta">
    <span class="t10-cstars">{_t10_stars(b['rating'])}</span>
    <span class="t10-cgenre">{genre_short}</span>
  </div>
</div>"""


def _t10_detail(i: int, b: dict) -> str:
    display = "flex" if i == 0 else "none"
    year_str = str(b["year"]) if b["year"] > 0 else "—"
    return f"""
<div class="t10-detail" data-dp="{i}" style="display:{display};">
  <div class="t10-det-cover" style="background:linear-gradient(160deg,{b['from']},{b['to']});">
    <span class="t10-det-rank">#{b['rank']}</span>
    <img class="t10-det-img" src="{b['cover']}" alt=""
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="t10-det-emoji" style="display:none;">{b['emoji']}</span>
  </div>
  <div class="t10-det-info">
    <h2 class="t10-det-title">{b['title']}</h2>
    <p class="t10-det-author">by {b['author']}</p>
    <div class="t10-chips">
      <span class="t10-chip">{b['genre']}</span>
      <span class="t10-chip">📅 {year_str}</span>
    </div>
    <p class="t10-det-stars">{_t10_stars(b['rating'])}
      <span class="t10-det-rnum">&nbsp;{b['rating']:.1f} / 5.0</span>
    </p>
    <p class="t10-det-desc">{truncate(b['desc'], 400)}</p>
  </div>
</div>"""


def render_top10_tab(data) -> str:
    books   = _build_top10(data)
    grid    = "".join(_t10_card(i, b)   for i, b in enumerate(books))
    details = "".join(_t10_detail(i, b) for i, b in enumerate(books))
    return f"""
<div class="t10-wrap">

  <div class="t10-shelf-row">
    <span class="t10-shelf-lbl">⭐&nbsp; Top Picks from the library</span>
    <span class="t10-shelf-hr"></span>
  </div>

  <div class="t10-grid">{grid}</div>

  <div class="t10-det-wrap">
    <p class="t10-det-heading">Book Details</p>
    {details}
  </div>

</div>

<script>
function selectBook(idx) {{
  document.querySelectorAll('[data-dp]').forEach(function(el) {{ el.style.display = 'none'; }});
  document.querySelectorAll('[data-card]').forEach(function(el) {{ el.classList.remove('t10-selected'); }});
  var dp   = document.querySelector('[data-dp="'   + idx + '"]');
  var card = document.querySelector('[data-card="' + idx + '"]');
  if (dp)   dp.style.display = 'flex';
  if (card) card.classList.add('t10-selected');
  if (dp)   dp.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}
document.addEventListener('DOMContentLoaded', function() {{ selectBook(0); }});
</script>
"""

# ---------------------------------------------------------------------------
# ── RECOMMENDER ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_REC_GRADS = [
    "#667eea,#764ba2", "#f093fb,#f5576c", "#4facfe,#00f2fe",
    "#43e97b,#38f9d7", "#fa709a,#fee140", "#a18cd1,#fbc2eb",
    "#fda085,#f6d365", "#84fab0,#8fd3f4", "#cd9cf2,#f6f3ff",
    "#fd7043,#ff8a65",
]


def _rec_gradient(title: str) -> str:
    g = _REC_GRADS[abs(hash(title)) % len(_REC_GRADS)]
    return f"linear-gradient(160deg,{g})"


def _rec_cover_url(title: str) -> str:
    return f"https://covers.openlibrary.org/b/title/{urllib.parse.quote(title)}-M.jpg"


def _rec_stars(r: float) -> str:
    f = int(round(r))
    return "★" * f + "☆" * (5 - f)


def _results_to_html(results) -> str:
    if results is None or results.empty:
        return """
<div style="text-align:center;padding:60px 20px;
            font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
  <div style="font-size:4em;margin-bottom:12px;">📚</div>
  <p style="font-size:1.1em;color:#86868B;margin:0;">
    No books found. Try a different title or relax your filters.
  </p>
</div>"""

    cards = []
    for _, row in results.iterrows():
        year_str = str(int(row["Year"])) if row["Year"] > 0 else ""
        snippet  = truncate(row["Description"], 240)
        cover    = _rec_cover_url(row["Title"])
        grad     = _rec_gradient(row["Title"])
        initial  = row["Title"][0].upper() if row["Title"] else "?"
        stars    = _rec_stars(row["Rating"])
        meta     = row["Category"] + (f" · {year_str}" if year_str else "")

        cards.append(f"""
<div class="rec-card">
  <div class="rec-scene">
    <div class="rec-wrap">
      <img class="rec-img" src="{cover}" alt=""
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" />
      <div class="rec-ph" style="background:{grad};">
        <span class="rec-initial">{initial}</span>
      </div>
    </div>
  </div>
  <div class="rec-info">
    <span class="rec-score">{row['Score']:.2f}</span>
    <p class="rec-title">{row['Title']}</p>
    <p class="rec-author">{row['Authors']}</p>
    <p class="rec-stars">{stars} <span class="rec-rnum">{row['Rating']:.1f}</span></p>
    <details class="rec-details">
      <summary class="rec-summary">More info ›</summary>
      <p class="rec-desc">{snippet}</p>
      <p class="rec-meta">{meta}</p>
    </details>
  </div>
</div>""")

    return '<div class="rec-grid">' + "".join(cards) + "</div>"


def search_by_title(title, n, min_year, max_year, min_rating, category, text_weight):
    if not str(title).strip():
        return "<p class='rec-hint'>Enter a book title above.</p>"
    return _results_to_html(recommend_by_title(
        query_title=str(title), df=df, tfidf_matrix=tfidf_matrix, vectorizer=vectorizer,
        n=int(n), min_year=int(min_year), max_year=int(max_year),
        min_rating=float(min_rating), category=str(category), text_weight=float(text_weight),
    ))


def search_by_query(query, n, min_year, max_year, min_rating, category, text_weight):
    if not str(query).strip():
        return "<p class='rec-hint'>Describe what you want to read above.</p>"
    return _results_to_html(recommend_by_query(
        query_text=str(query), df=df, tfidf_matrix=tfidf_matrix, vectorizer=vectorizer,
        n=int(n), min_year=int(min_year), max_year=int(max_year),
        min_rating=float(min_rating), category=str(category), text_weight=float(text_weight),
    ))

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── global ── */
body, .gradio-container {
    background: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif !important;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

/* ── blocks / inputs ── */
.block { background:#fff !important; border-radius:18px !important;
         border:none !important; box-shadow:0 1px 6px rgba(0,0,0,.07) !important; }
input, textarea, select {
    border-radius:10px !important; border:1.5px solid #E5E5EA !important;
    font-size:0.94em !important; background:#FAFAFA !important;
}

/* ── primary button ── */
button.primary {
    background:#007AFF !important; border:none !important;
    border-radius:980px !important; font-weight:600 !important;
    font-size:0.95em !important; padding:10px 28px !important;
    transition:background 0.2s !important;
}
button.primary:hover { background:#0063CC !important; }

/* ── tabs ── */
.tab-nav button { font-size:0.92em !important; }
.tab-nav button.selected { color:#007AFF !important; border-bottom-color:#007AFF !important; }

/* ════════════════════════════════════════════════
   TOP 10  (t10- prefix)
════════════════════════════════════════════════ */
.t10-wrap { padding:8px 4px 24px;
            font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif; }

.t10-shelf-row  { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
.t10-shelf-lbl  { font-size:0.88em; font-weight:600; color:#1D1D1F; white-space:nowrap; }
.t10-shelf-hr   { flex:1; height:1px; background:#D1D1D6; }

/* grid */
.t10-grid {
    display:grid;
    grid-template-columns:repeat(5,1fr);
    gap:20px 16px;
    margin-bottom:28px;
}

/* card */
.t10-card {
    cursor:pointer;
    border-radius:4px 12px 12px 4px;
    overflow:hidden;
    box-shadow:6px 8px 22px rgba(0,0,0,.20),-2px 0 5px rgba(0,0,0,.10);
    transition:transform .28s ease,box-shadow .28s ease;
    position:relative;
    transform:perspective(600px) rotateY(-4deg);
    background:#fff;
    outline:3px solid transparent;
    outline-offset:3px;
}
.t10-card:hover {
    transform:perspective(600px) rotateY(-13deg) translateX(-5px) scale(1.04);
    box-shadow:14px 16px 40px rgba(0,0,0,.28),-4px 0 10px rgba(0,0,0,.15);
}
.t10-selected {
    outline:3px solid #007AFF !important;
    transform:perspective(600px) rotateY(-13deg) translateX(-5px) scale(1.04) !important;
    box-shadow:14px 16px 40px rgba(0,0,0,.28),-4px 0 10px rgba(0,0,0,.15) !important;
}

/* rank badge */
.t10-rank {
    position:absolute; top:8px; left:8px; z-index:2;
    width:26px; height:26px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.73em; font-weight:700; color:#fff;
    box-shadow:0 2px 6px rgba(0,0,0,.30);
}

/* cover */
.t10-cover { position:relative; padding-top:145%; overflow:hidden; }
.t10-cover-img {
    position:absolute; top:0; left:0;
    width:100%; height:100%; object-fit:cover; pointer-events:none;
}
.t10-emoji {
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-60%);
    font-size:3.2em;
    filter:drop-shadow(0 4px 8px rgba(0,0,0,.25));
    pointer-events:none;
}
.t10-foot {
    position:absolute; bottom:0; left:0; right:0;
    background:linear-gradient(to top,rgba(0,0,0,.72) 0%,transparent 100%);
    padding:28px 10px 10px;
}
.t10-ctitle {
    margin:0; font-size:.77em; font-weight:700; color:#fff; line-height:1.25;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}
.t10-cauthor { margin:2px 0 0; font-size:.67em; color:rgba(255,255,255,.75);
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* card footer */
.t10-card-meta { display:flex; align-items:center; justify-content:space-between;
                 padding:6px 10px 7px; background:#fff; }
.t10-cstars { color:#FF9F0A; font-size:.70em; letter-spacing:.4px; }
.t10-cgenre { font-size:.64em; color:#8E8E93; white-space:nowrap;
              overflow:hidden; text-overflow:ellipsis; max-width:58%; }

/* detail wrap */
.t10-det-wrap { background:#fff; border-radius:20px; padding:22px 26px 26px;
                box-shadow:0 2px 16px rgba(0,0,0,.08); }
.t10-det-heading { margin:0 0 16px; font-size:.78em; font-weight:600;
                   color:#8E8E93; text-transform:uppercase; letter-spacing:.9px; }

/* detail panel */
.t10-detail { display:none; align-items:flex-start; gap:28px; }

.t10-det-cover {
    flex-shrink:0; width:175px; height:258px;
    border-radius:4px 14px 14px 4px;
    display:flex; align-items:center; justify-content:center;
    position:relative;
    box-shadow:8px 12px 32px rgba(0,0,0,.28),-3px 0 8px rgba(0,0,0,.15);
    transform:perspective(600px) rotateY(-8deg);
}
.t10-det-rank {
    position:absolute; top:10px; left:12px; z-index:2;
    font-size:1em; font-weight:800;
    color:rgba(255,255,255,.9); text-shadow:0 2px 6px rgba(0,0,0,.4);
}
.t10-det-img {
    position:absolute; top:0; left:0;
    width:100%; height:100%; object-fit:cover;
    border-radius:4px 14px 14px 4px;
}
.t10-det-emoji { font-size:5em; filter:drop-shadow(0 6px 12px rgba(0,0,0,.3)); }

.t10-det-info { flex:1; padding-top:4px; }
.t10-det-title { margin:0 0 4px; font-size:1.55em; font-weight:700;
                 color:#1D1D1F; line-height:1.2; letter-spacing:-.4px; }
.t10-det-author { margin:0 0 12px; font-size:.94em; color:#6E6E73; font-weight:500; }
.t10-chips      { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px; }
.t10-chip       { background:#F2F2F7; color:#3A3A3C; border-radius:980px;
                  padding:4px 13px; font-size:.78em; font-weight:500; }
.t10-det-stars  { margin:0 0 14px; font-size:1em; color:#FF9F0A; }
.t10-det-rnum   { color:#3A3A3C; font-size:.87em; font-weight:600; }
.t10-det-desc   { margin:0; font-size:.93em; color:#3A3A3C; line-height:1.62; }

/* ════════════════════════════════════════════════
   RECOMMENDER  (rec- prefix)
════════════════════════════════════════════════ */
.rec-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(175px,1fr));
    gap:28px 22px;
    padding:12px 4px 24px;
}
.rec-card   { display:flex; flex-direction:column; align-items:center; }

.rec-scene  { width:100%; perspective:900px; margin-bottom:12px; }
.rec-wrap   {
    position:relative; padding-top:148%;
    transform-style:preserve-3d;
    transition:transform .35s ease;
}
.rec-card:hover .rec-wrap {
    transform:rotateY(-14deg) translateX(-4px) scale(1.03);
}
.rec-img, .rec-ph {
    position:absolute; top:0; left:0; width:100%; height:100%;
    border-radius:3px 10px 10px 3px;
    box-shadow:8px 10px 30px rgba(0,0,0,.28),-3px 0 6px rgba(0,0,0,.15),
               inset -3px 0 8px rgba(0,0,0,.10);
}
.rec-img    { object-fit:cover; }
.rec-ph     { display:none; align-items:center; justify-content:center; }
.rec-initial { font-size:3.8em; font-weight:700; color:rgba(255,255,255,.92);
               text-shadow:0 2px 8px rgba(0,0,0,.25); }

.rec-info   { width:100%; text-align:center; padding:0 2px; }
.rec-score  { display:inline-block; background:rgba(0,122,255,.10); color:#007AFF;
              border-radius:6px; padding:1px 8px; font-size:.7em; font-weight:700; margin-bottom:5px; }
.rec-title  { margin:0 0 3px; font-size:.86em; font-weight:600; color:#1D1D1F;
              line-height:1.3; display:-webkit-box; -webkit-line-clamp:2;
              -webkit-box-orient:vertical; overflow:hidden; }
.rec-author { margin:0 0 4px; font-size:.76em; color:#6E6E73;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rec-stars  { margin:0 0 5px; font-size:.8em; color:#FF9F0A; }
.rec-rnum   { color:#8E8E93; font-size:.88em; }

.rec-details { text-align:left; }
.rec-summary { font-size:.75em; color:#007AFF; cursor:pointer;
               list-style:none; -webkit-appearance:none; text-align:center; }
.rec-desc   { margin:6px 0 0; font-size:.75em; color:#3A3A3C; line-height:1.46; }
.rec-meta   { margin:4px 0 0; font-size:.71em; color:#8E8E93; }
.rec-hint   { padding:12px; color:#8E8E93; font-family:-apple-system,sans-serif; }

/* ── responsive ── */
@media (max-width:920px) {
    .t10-grid { grid-template-columns:repeat(3,1fr) !important; }
}
@media (max-width:580px) {
    .t10-grid { grid-template-columns:repeat(2,1fr) !important; }
    .t10-detail { flex-direction:column !important; }
    .t10-det-cover { width:140px !important; height:206px !important; }
}
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Book Recommender", css=CSS) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 16px;
                font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif;">
      <svg width="52" height="52" viewBox="0 0 52 52" fill="none"
           style="margin-bottom:8px;filter:drop-shadow(0 4px 12px rgba(0,122,255,0.30));">
        <rect width="52" height="52" rx="14" fill="#007AFF"/>
        <text x="50%" y="56%" dominant-baseline="middle" text-anchor="middle"
              font-size="28" fill="white">📚</text>
      </svg>
      <h1 style="margin:4px 0 0;font-size:2em;font-weight:700;color:#1D1D1F;letter-spacing:-.6px;">
        Book Recommender
      </h1>
      <p style="margin:6px 0 0;color:#86868B;font-size:.98em;">
        Discover your next favourite read
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Top 10 ──────────────────────────────────────────────────
        with gr.Tab("📚  Top 10"):
            gr.HTML(render_top10_tab(df))

        # ── Tab 2 & 3: Recommender ─────────────────────────────────────────
        with gr.Tab("🔍  Find Similar Books"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220):
                    gr.HTML("<p style='font-size:1em;font-weight:600;color:#1D1D1F;"
                            "margin:0 0 8px;'>Filters</p>")
                    n1  = gr.Slider(1, 20, value=8, step=1, label="Results")
                    y1a = gr.Number(value=0, precision=0, label="From year  (0 = any)")
                    y1b = gr.Number(value=0, precision=0, label="To year  (0 = any)")
                    r1  = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Min rating")
                    c1  = gr.Dropdown(choices=CATEGORIES, value="All", label="Category")
                    w1  = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                    label="Text weight  [0 popular · 1 content]")
                with gr.Column(scale=3):
                    ti  = gr.Textbox(label="Book title", placeholder="e.g.  The Great Gatsby")
                    tb  = gr.Button("Find similar books", variant="primary")
                    to_ = gr.HTML()
                    _t  = [ti, n1, y1a, y1b, r1, c1, w1]
                    tb.click(search_by_title, inputs=_t, outputs=to_)
                    ti.submit(search_by_title, inputs=_t, outputs=to_)

        with gr.Tab("💬  Search by Description"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=220):
                    gr.HTML("<p style='font-size:1em;font-weight:600;color:#1D1D1F;"
                            "margin:0 0 8px;'>Filters</p>")
                    n2  = gr.Slider(1, 20, value=8, step=1, label="Results")
                    y2a = gr.Number(value=0, precision=0, label="From year  (0 = any)")
                    y2b = gr.Number(value=0, precision=0, label="To year  (0 = any)")
                    r2  = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Min rating")
                    c2  = gr.Dropdown(choices=CATEGORIES, value="All", label="Category")
                    w2  = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                    label="Text weight  [0 popular · 1 content]")
                with gr.Column(scale=3):
                    qi  = gr.Textbox(
                        label="Describe what you want to read",
                        placeholder="e.g.  a gripping thriller set in Victorian London with dark secrets",
                        lines=3,
                    )
                    qb  = gr.Button("Search", variant="primary")
                    qo  = gr.HTML()
                    _q  = [qi, n2, y2a, y2b, r2, c2, w2]
                    qb.click(search_by_query, inputs=_q, outputs=qo)
                    qi.submit(search_by_query, inputs=_q, outputs=qo)

    gr.HTML("""
    <p style="text-align:center;color:#8E8E93;font-size:.78em;margin-top:8px;
              font-family:-apple-system,sans-serif;">
      Book Recommender &nbsp;·&nbsp; EADA Business School
      &nbsp;·&nbsp; Deep Learning &amp; LLM Project
    </p>
    """)

demo.launch()
