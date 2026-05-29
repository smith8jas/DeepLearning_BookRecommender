"""
Top 10 Books Library — Apple Books style.

Run from the project root:
    python src/top10.py
"""
from __future__ import annotations
import gradio as gr

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _cover(isbn: str) -> str:
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"


BOOKS = [
    {
        "rank": 1, "title": "To Kill a Mockingbird", "author": "Harper Lee",
        "genre": "Fiction · Drama", "year": 1960, "rating": 4.8, "pages": 281,
        "description": "A gripping tale of racial injustice and childhood innocence in the Deep South. "
                       "Through the eyes of young Scout Finch, we witness her father Atticus defend a "
                       "Black man falsely accused of a crime — a timeless story of courage, compassion, "
                       "and moral conviction.",
        "from": "#2d5016", "to": "#52734d", "emoji": "🌳",
        "cover": _cover("9780061935466"),
    },
    {
        "rank": 2, "title": "1984", "author": "George Orwell",
        "genre": "Dystopian · Fiction", "year": 1949, "rating": 4.7, "pages": 328,
        "description": "In a totalitarian future society where the Party controls every aspect of life, "
                       "Winston Smith secretly rebels against the oppressive regime. Orwell's masterpiece "
                       "remains the definitive warning against authoritarian control and mass surveillance.",
        "from": "#1a1a2e", "to": "#16213e", "emoji": "👁",
        "cover": _cover("9780451524935"),
    },
    {
        "rank": 3, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald",
        "genre": "Fiction · Classic", "year": 1925, "rating": 4.4, "pages": 180,
        "description": "Set in the Roaring Twenties, this iconic American novel follows the mysterious "
                       "millionaire Jay Gatsby and his obsession with the beautiful Daisy Buchanan. "
                       "A timeless critique of wealth, class, and the elusive American Dream.",
        "from": "#7b2d8b", "to": "#c06c84", "emoji": "🥂",
        "cover": _cover("9780743273565"),
    },
    {
        "rank": 4, "title": "Sapiens", "author": "Yuval Noah Harari",
        "genre": "Non-fiction · History", "year": 2011, "rating": 4.6, "pages": 443,
        "description": "A sweeping history of humankind from the first Homo sapiens to the present day. "
                       "Harari examines how biology and history shaped us and explores whether we are "
                       "happier today than our forager ancestors — a thought-provoking essential read.",
        "from": "#c0392b", "to": "#e74c3c", "emoji": "🦴",
        "cover": _cover("9780062316097"),
    },
    {
        "rank": 5, "title": "Atomic Habits", "author": "James Clear",
        "genre": "Self-help · Psychology", "year": 2018, "rating": 4.8, "pages": 320,
        "description": "The definitive guide to building good habits and breaking bad ones. Clear reveals "
                       "how tiny incremental changes compound into remarkable results — backed by "
                       "cutting-edge science and real-world examples from elite athletes and business leaders.",
        "from": "#b7410e", "to": "#f39c12", "emoji": "⚛️",
        "cover": _cover("9780735211292"),
    },
    {
        "rank": 6, "title": "The Alchemist", "author": "Paulo Coelho",
        "genre": "Fiction · Philosophy", "year": 1988, "rating": 4.5, "pages": 208,
        "description": "Santiago, an Andalusian shepherd boy, dreams of discovering a worldly treasure. "
                       "His quest leads him to encounter the mystical and the mundane alike. "
                       "A beautiful allegory about following your dreams and listening to your heart.",
        "from": "#1565C0", "to": "#42A5F5", "emoji": "🌙",
        "cover": _cover("9780062315007"),
    },
    {
        "rank": 7, "title": "Dune", "author": "Frank Herbert",
        "genre": "Science Fiction · Epic", "year": 1965, "rating": 4.7, "pages": 412,
        "description": "On the desert planet Arrakis, young Paul Atreides discovers his destiny amidst "
                       "political intrigue and ecological warfare. The greatest science-fiction novel "
                       "ever written — a rich tapestry of ecology, religion, and power.",
        "from": "#7B3F00", "to": "#D4A017", "emoji": "🏜️",
        "cover": _cover("9780441013593"),
    },
    {
        "rank": 8, "title": "Pride and Prejudice", "author": "Jane Austen",
        "genre": "Romance · Classic", "year": 1813, "rating": 4.6, "pages": 432,
        "description": "The spirited Elizabeth Bennet navigates issues of manners, upbringing, and "
                       "marriage in Georgian England. Her relationship with the proud Mr. Darcy is one "
                       "of literature's greatest love stories — witty, warm, and utterly timeless.",
        "from": "#134e5e", "to": "#71b280", "emoji": "🌸",
        "cover": _cover("9780141439518"),
    },
    {
        "rank": 9, "title": "The Power of Now", "author": "Eckhart Tolle",
        "genre": "Self-help · Spirituality", "year": 1997, "rating": 4.4, "pages": 236,
        "description": "A guide to spiritual enlightenment, this bestseller teaches the importance of "
                       "living in the present moment. Tolle blends ancient wisdom with modern psychology "
                       "to show how releasing the past and future can transform your consciousness.",
        "from": "#0F2027", "to": "#4286f4", "emoji": "🌀",
        "cover": _cover("9781577314806"),
    },
    {
        "rank": 10, "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling",
        "genre": "Fantasy · Adventure", "year": 1997, "rating": 4.9, "pages": 309,
        "description": "Harry Potter discovers on his eleventh birthday that he is a wizard and is "
                       "admitted to Hogwarts School of Witchcraft and Wizardry. The beginning of a "
                       "magical saga that has captured the hearts of millions worldwide.",
        "from": "#360033", "to": "#0b8793", "emoji": "⚡",
        "cover": _cover("9780590353427"),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BADGE = {
    1: "linear-gradient(135deg,#FFD700,#FFA500)",
    2: "linear-gradient(135deg,#C0C0C0,#9E9E9E)",
    3: "linear-gradient(135deg,#CD7F32,#A0522D)",
}


def _stars(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def _card(i: int, b: dict) -> str:
    badge = _BADGE.get(b["rank"], "rgba(0,0,0,0.45)")
    genre_short = b["genre"].split("·")[0].strip()
    return f"""
<div class="ab-card" data-card="{i}" onclick="selectBook({i})">
  <div class="ab-rank" style="background:{badge};">{b['rank']}</div>
  <div class="ab-cover" style="background:linear-gradient(160deg,{b['from']},{b['to']});">
    <img class="ab-cover-img" src="{b['cover']}" alt=""
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="ab-emoji" style="display:none;">{b['emoji']}</span>
    <div class="ab-cover-foot">
      <p class="ab-ctitle">{b['title']}</p>
      <p class="ab-cauthor">{b['author']}</p>
    </div>
  </div>
  <div class="ab-card-meta">
    <span class="ab-cstars">{_stars(b['rating'])}</span>
    <span class="ab-cgenre">{genre_short}</span>
  </div>
</div>"""


def _detail(i: int, b: dict) -> str:
    display = "flex" if i == 0 else "none"
    return f"""
<div class="ab-detail" data-dp="{i}" style="display:{display};">
  <div class="ab-det-cover" style="background:linear-gradient(160deg,{b['from']},{b['to']});">
    <span class="ab-det-rank">#{b['rank']}</span>
    <img class="ab-det-img" src="{b['cover']}" alt=""
         onerror="this.style.display='none';this.nextElementSibling.style.display='block';" />
    <span class="ab-det-emoji" style="display:none;">{b['emoji']}</span>
  </div>
  <div class="ab-det-info">
    <h2 class="ab-det-title">{b['title']}</h2>
    <p class="ab-det-author">by {b['author']}</p>
    <div class="ab-det-chips">
      <span class="ab-chip">{b['genre']}</span>
      <span class="ab-chip">📅 {b['year']}</span>
      <span class="ab-chip">📄 {b['pages']} pages</span>
    </div>
    <p class="ab-det-stars">{_stars(b['rating'])}
      <span class="ab-det-rnum">&nbsp;{b['rating']:.1f} / 5.0</span>
    </p>
    <p class="ab-det-desc">{b['description']}</p>
  </div>
</div>"""


def render_page() -> str:
    grid    = "".join(_card(i, b)   for i, b in enumerate(BOOKS))
    details = "".join(_detail(i, b) for i, b in enumerate(BOOKS))
    return f"""
<div class="ab-wrap">

  <!-- Header -->
  <div class="ab-header">
    <div class="ab-logo">📚</div>
    <h1 class="ab-h1">Top 10 Books Library</h1>
    <p class="ab-sub">Discover the most recommended books of the moment</p>
  </div>

  <!-- Shelf label -->
  <div class="ab-shelf-row">
    <span class="ab-shelf-lbl">⭐&nbsp; Top Picks</span>
    <span class="ab-shelf-hr"></span>
  </div>

  <!-- Grid -->
  <div class="ab-grid">{grid}</div>

  <!-- Detail panel -->
  <div class="ab-det-wrap">
    <p class="ab-det-heading">Book Details</p>
    {details}
  </div>

  <!-- Footer -->
  <p class="ab-footer">
    Top 10 Books Library &nbsp;·&nbsp; EADA Business School
    &nbsp;·&nbsp; Deep Learning &amp; LLM Project
  </p>

</div>

<script>
function selectBook(idx) {{
  document.querySelectorAll('[data-dp]').forEach(function(el) {{ el.style.display = 'none'; }});
  document.querySelectorAll('[data-card]').forEach(function(el) {{ el.classList.remove('ab-selected'); }});
  var dp   = document.querySelector('[data-dp="'   + idx + '"]');
  var card = document.querySelector('[data-card="' + idx + '"]');
  if (dp)   dp.style.display = 'flex';
  if (card) card.classList.add('ab-selected');
  if (dp)   dp.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}
document.addEventListener('DOMContentLoaded', function() {{ selectBook(0); }});
</script>
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── page ── */
.gradio-container {
    background: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

/* ── wrapper ── */
.ab-wrap {
    padding: 0 4px 32px;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
}

/* ── header ── */
.ab-header { text-align: center; padding: 36px 0 24px; }
.ab-logo   { font-size: 3.6em; margin-bottom: 8px;
             filter: drop-shadow(0 4px 14px rgba(0,122,255,0.28)); }
.ab-h1     { margin: 0 0 6px; font-size: 2.2em; font-weight: 700;
             color: #1D1D1F; letter-spacing: -0.6px; }
.ab-sub    { margin: 0; color: #86868B; font-size: 1em; }

/* ── shelf label ── */
.ab-shelf-row { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.ab-shelf-lbl { font-size: 0.88em; font-weight: 600; color: #1D1D1F; white-space: nowrap; }
.ab-shelf-hr  { flex: 1; height: 1px; background: #D1D1D6; }

/* ── grid ── */
.ab-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px 16px;
    margin-bottom: 28px;
}

/* ── card ── */
.ab-card {
    cursor: pointer;
    border-radius: 4px 12px 12px 4px;
    overflow: hidden;
    box-shadow: 6px 8px 22px rgba(0,0,0,0.20), -2px 0 5px rgba(0,0,0,0.10);
    transition: transform 0.28s ease, box-shadow 0.28s ease;
    position: relative;
    transform: perspective(600px) rotateY(-4deg);
    background: #fff;
    outline: 3px solid transparent;
    outline-offset: 3px;
}
.ab-card:hover {
    transform: perspective(600px) rotateY(-13deg) translateX(-5px) scale(1.04);
    box-shadow: 14px 16px 40px rgba(0,0,0,0.28), -4px 0 10px rgba(0,0,0,0.15);
}
.ab-selected {
    outline: 3px solid #007AFF !important;
    transform: perspective(600px) rotateY(-13deg) translateX(-5px) scale(1.04) !important;
    box-shadow: 14px 16px 40px rgba(0,0,0,0.28), -4px 0 10px rgba(0,0,0,0.15) !important;
}

/* ── rank badge ── */
.ab-rank {
    position: absolute; top: 8px; left: 8px; z-index: 2;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.73em; font-weight: 700; color: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.30);
}

/* ── cover ── */
.ab-cover { position: relative; padding-top: 145%; overflow: hidden; }
.ab-cover-img {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    pointer-events: none;
}
.ab-emoji {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -60%);
    font-size: 3.2em;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
    pointer-events: none;
}
.ab-cover-foot {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, rgba(0,0,0,0.72) 0%, transparent 100%);
    padding: 28px 10px 10px;
}
.ab-ctitle {
    margin: 0; font-size: 0.77em; font-weight: 700; color: #fff;
    line-height: 1.25;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.ab-cauthor {
    margin: 2px 0 0; font-size: 0.67em; color: rgba(255,255,255,0.75);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── card footer ── */
.ab-card-meta {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px 7px; background: #fff;
}
.ab-cstars { color: #FF9F0A; font-size: 0.70em; letter-spacing: 0.4px; }
.ab-cgenre { font-size: 0.64em; color: #8E8E93; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; max-width: 58%; }

/* ── detail wrap ── */
.ab-det-wrap {
    background: #fff; border-radius: 20px;
    padding: 22px 26px 26px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
}
.ab-det-heading {
    margin: 0 0 16px; font-size: 0.78em; font-weight: 600;
    color: #8E8E93; text-transform: uppercase; letter-spacing: 0.9px;
}

/* ── detail panel ── */
.ab-detail { display: none; align-items: flex-start; gap: 28px; }

/* ── detail cover ── */
.ab-det-cover {
    flex-shrink: 0; width: 175px; height: 258px;
    border-radius: 4px 14px 14px 4px;
    display: flex; align-items: center; justify-content: center;
    position: relative;
    box-shadow: 8px 12px 32px rgba(0,0,0,0.28), -3px 0 8px rgba(0,0,0,0.15);
    transform: perspective(600px) rotateY(-8deg);
}
.ab-det-rank {
    position: absolute; top: 10px; left: 12px; z-index: 2;
    font-size: 1em; font-weight: 800;
    color: rgba(255,255,255,0.9); text-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
.ab-det-img {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    border-radius: 4px 14px 14px 4px;
}
.ab-det-emoji {
    font-size: 5em;
    filter: drop-shadow(0 6px 12px rgba(0,0,0,0.3));
}

/* ── detail info ── */
.ab-det-info  { flex: 1; padding-top: 4px; }
.ab-det-title {
    margin: 0 0 4px; font-size: 1.55em; font-weight: 700;
    color: #1D1D1F; line-height: 1.2; letter-spacing: -0.4px;
}
.ab-det-author { margin: 0 0 12px; font-size: 0.94em; color: #6E6E73; font-weight: 500; }
.ab-det-chips  { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
.ab-chip {
    background: #F2F2F7; color: #3A3A3C;
    border-radius: 980px; padding: 4px 13px;
    font-size: 0.78em; font-weight: 500;
}
.ab-det-stars  { margin: 0 0 14px; font-size: 1em; color: #FF9F0A; }
.ab-det-rnum   { color: #3A3A3C; font-size: 0.87em; font-weight: 600; }
.ab-det-desc   { margin: 0; font-size: 0.93em; color: #3A3A3C; line-height: 1.62; }

/* ── footer ── */
.ab-footer {
    text-align: center; color: #8E8E93; font-size: 0.78em; margin-top: 20px;
}

/* ── responsive ── */
@media (max-width: 920px) {
    .ab-grid { grid-template-columns: repeat(3, 1fr) !important; }
}
@media (max-width: 580px) {
    .ab-grid   { grid-template-columns: repeat(2, 1fr) !important; }
    .ab-detail { flex-direction: column !important; }
    .ab-det-cover { width: 140px !important; height: 206px !important; }
    .ab-h1 { font-size: 1.6em !important; }
}
"""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

with gr.Blocks(title="Top 10 Books Library", css=CSS) as demo:
    gr.HTML(render_page())

demo.launch()
