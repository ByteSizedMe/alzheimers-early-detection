"""Presentation layer for the dashboard: palette, CSS, and HTML fragments.

Kept separate from ``app.py`` so the Streamlit flow stays readable — the app
module reads as a sequence of steps, and every string of markup lives here.

Design intent: a clinical review tool, not a consumer app. Dark ground so the
MRI greyscale carries the visual weight, one accent (teal) for interface
chrome, and the four stage colours reserved exclusively for diagnosis — a
colour that means "stage" never appears on a button or a border.
"""

from __future__ import annotations

# Stage palette: a deliberate green-to-red progression so severity is legible
# before the label is read. Each stage carries a matched dim tone for fills.
STAGE_COLORS = {
    "NC": "#3FB950",
    "EMCI": "#D9A521",
    "LMCI": "#E8843C",
    "AD": "#F04E4E",
}
STAGE_FILLS = {
    "NC": "rgba(63, 185, 80, 0.14)",
    "EMCI": "rgba(217, 165, 33, 0.14)",
    "LMCI": "rgba(232, 132, 60, 0.14)",
    "AD": "rgba(240, 78, 78, 0.14)",
}
ACCENT = "#2AA0B5"
INK = "#E6EDF3"
MUTED = "#8B9CAD"
SURFACE = "#16202C"
BORDER = "rgba(255, 255, 255, 0.09)"

CSS = f"""
<style>
  /* ---------------------------------------------------------- foundations */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
  #MainMenu, footer {{ visibility: hidden; }}

  /* ----------------------------------------------------------------- hero */
  .hero {{ margin-bottom: 1.6rem; }}
  .hero h1 {{
    font-size: 2.35rem; font-weight: 700; letter-spacing: -0.021em;
    margin: 0 0 0.35rem 0; line-height: 1.15;
    background: linear-gradient(96deg, {INK} 20%, {ACCENT} 130%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero p {{ color: {MUTED}; font-size: 0.96rem; margin: 0; max-width: 68ch; line-height: 1.55; }}
  .pill {{
    display: inline-block; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.09em; text-transform: uppercase;
    padding: 0.24rem 0.6rem; border-radius: 999px; margin-bottom: 0.75rem;
    color: {ACCENT}; background: rgba(42, 160, 181, 0.12);
    border: 1px solid rgba(42, 160, 181, 0.3);
  }}

  /* ---------------------------------------------------------------- cards */
  .card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 1.15rem 1.3rem; margin-bottom: 1rem;
  }}
  .card-label {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 0.7rem;
  }}

  /* -------------------------------------------------------------- verdict */
  .verdict {{
    display: flex; align-items: center; gap: 1.5rem;
    border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
    border: 1px solid {BORDER};
  }}
  .verdict-stage {{ font-size: 2.6rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }}
  .verdict-desc {{ font-size: 1.02rem; color: {INK}; margin-top: 0.3rem; }}
  .verdict-sub {{ font-size: 0.82rem; color: {MUTED}; margin-top: 0.25rem; }}
  .verdict-spacer {{ flex: 1; }}
  .verdict-metric {{ text-align: right; padding-left: 1.5rem; }}
  .verdict-metric .v {{ font-size: 1.55rem; font-weight: 700; line-height: 1.1; }}
  .verdict-metric .k {{
    font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: {MUTED}; margin-top: 0.15rem;
  }}

  /* ---------------------------------------------------------- stage track */
  .track {{ display: flex; gap: 5px; margin: 0.35rem 0 0.55rem 0; }}
  .track-seg {{
    flex: 1; height: 7px; border-radius: 4px;
    background: rgba(255, 255, 255, 0.07);
  }}
  .track-labels {{ display: flex; gap: 5px; }}
  .track-labels span {{
    flex: 1; font-size: 0.72rem; letter-spacing: 0.04em;
    color: {MUTED}; text-align: center;
  }}

  /* --------------------------------------------------------- probability */
  .prob-row {{ display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.8rem; }}
  .prob-name {{ width: 58px; font-weight: 600; font-size: 0.92rem; flex-shrink: 0; }}
  .prob-track {{
    flex: 1; height: 9px; border-radius: 5px;
    background: rgba(255, 255, 255, 0.06); overflow: hidden;
  }}
  .prob-fill {{ height: 100%; border-radius: 5px; }}
  .prob-value {{
    width: 62px; text-align: right; font-variant-numeric: tabular-nums;
    font-size: 0.88rem; font-weight: 600; flex-shrink: 0;
  }}
  .prob-desc {{ font-size: 0.74rem; color: {MUTED}; margin: -0.55rem 0 0.75rem 74px; }}

  /* -------------------------------------------------------------- images */
  .scan {{
    width: 100%; border-radius: 11px; display: block;
    border: 1px solid {BORDER};
  }}
  .scan-caption {{ font-size: 0.76rem; color: {MUTED}; margin-top: 0.55rem; line-height: 1.45; }}

  /* --------------------------------------------------------------- audit */
  .audit {{ display: flex; gap: 1.6rem; flex-wrap: wrap; }}
  .audit-item .k {{
    font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED};
  }}
  .audit-item .v {{ font-size: 1.08rem; font-weight: 600; margin-top: 0.1rem; }}
  .chip {{
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 0.22rem 0.6rem; border-radius: 999px;
  }}
  .chip-ok {{ color: {STAGE_COLORS["NC"]}; background: {STAGE_FILLS["NC"]}; }}
  .chip-warn {{ color: {STAGE_COLORS["AD"]}; background: {STAGE_FILLS["AD"]}; }}

  /* --------------------------------------------------------- empty state */
  .empty {{
    border: 1px dashed rgba(255, 255, 255, 0.16); border-radius: 16px;
    padding: 2.4rem 2rem; text-align: center; background: rgba(255, 255, 255, 0.015);
  }}
  .empty h3 {{ margin: 0 0 0.5rem 0; font-size: 1.12rem; font-weight: 600; }}
  .empty p {{ color: {MUTED}; margin: 0 auto 0.4rem auto; max-width: 56ch; font-size: 0.9rem; line-height: 1.6; }}
  .empty code {{
    background: rgba(255, 255, 255, 0.07); padding: 0.16rem 0.42rem;
    border-radius: 5px; font-size: 0.84rem; color: {ACCENT};
  }}

  /* ------------------------------------------------------------- sidebar */
  section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  .side-title {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED};
    margin: 0.4rem 0 0.6rem 0;
  }}
  .status {{
    border-radius: 9px; padding: 0.6rem 0.75rem; font-size: 0.8rem;
    display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem;
  }}
  .status-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
  .disclaimer {{
    font-size: 0.72rem; color: {MUTED}; line-height: 1.5;
    border-top: 1px solid {BORDER}; padding-top: 0.9rem; margin-top: 1.4rem;
  }}
</style>
"""


def _html(markup: str) -> str:
    """Flatten generated markup to a single line before Streamlit renders it.

    Streamlit passes these fragments through a Markdown renderer, which treats
    four-space-indented lines as a code block -- so a nicely indented HTML
    template silently renders as literal source text instead of an element.
    Stripping per-line indentation and the newlines removes the ambiguity.
    """
    # Joined with a space, not "": prose wrapped across source lines would
    # otherwise lose the word break ("Grad-CAM attribution" -> "Grad-CAMattribution").
    return " ".join(line.strip() for line in markup.strip().splitlines())


def hero() -> str:
    return _html(f"""
    <div class="hero">
      <span class="pill">MRI stage classification</span>
      <h1>Early Detection of Alzheimer's</h1>
      <p>VGG19 transfer-learning classifier over structural MRI slices, with Grad-CAM
      attribution so every prediction can be checked rather than trusted.</p>
    </div>
    """)


def confidence_ring(fraction: float, color: str, size: int = 92) -> str:
    """An SVG progress ring — a bar would imply a scale this number lacks."""
    radius = size / 2 - 7
    circumference = 2 * 3.14159 * radius
    filled = circumference * max(0.0, min(fraction, 1.0))
    return _html(f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none"
              stroke="rgba(255,255,255,0.08)" stroke-width="7"/>
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none"
              stroke="{color}" stroke-width="7" stroke-linecap="round"
              stroke-dasharray="{filled} {circumference}"
              transform="rotate(-90 {size/2} {size/2})"/>
      <text x="50%" y="50%" text-anchor="middle" dy="0.36em"
            fill="{INK}" font-size="18" font-weight="700"
            font-family="system-ui, sans-serif">{fraction:.0%}</text>
    </svg>
    """)


def verdict(stage: str, description: str, confidence: float, impairment: float,
            explained: str) -> str:
    color = STAGE_COLORS.get(stage, ACCENT)
    fill = STAGE_FILLS.get(stage, "rgba(42,160,181,0.12)")
    return _html(f"""
    <div class="verdict" style="background:{fill}; border-color:{color}40;">
      {confidence_ring(confidence, color)}
      <div>
        <div class="verdict-stage" style="color:{color};">{stage}</div>
        <div class="verdict-desc">{description}</div>
        <div class="verdict-sub">Attribution shown for <b>{explained}</b></div>
      </div>
      <div class="verdict-spacer"></div>
      <div class="verdict-metric">
        <div class="v" style="color:{color};">{impairment:.1%}</div>
        <div class="k">Any impairment</div>
      </div>
    </div>
    """)


def stage_track(stages, predicted: str) -> str:
    """Four segments along the ordinal progression, lit up to the prediction.

    The stages are ordered, so showing position on the path communicates more
    than a single label: a reader sees how far along the sequence this sits.
    """
    try:
        index = list(stages).index(predicted)
    except ValueError:
        index = -1

    segments = ""
    for i, name in enumerate(stages):
        if i <= index:
            color = STAGE_COLORS.get(name, ACCENT)
            opacity = "1" if i == index else "0.4"
            segments += f'<div class="track-seg" style="background:{color}; opacity:{opacity};"></div>'
        else:
            segments += '<div class="track-seg"></div>'

    labels = ""
    for i, name in enumerate(stages):
        if i == index:
            color = STAGE_COLORS.get(name, ACCENT)
            labels += f'<span style="color:{color}; font-weight:700;">{name}</span>'
        else:
            labels += f"<span>{name}</span>"

    return _html(f"""
    <div class="card">
      <div class="card-label">Stage progression</div>
      <div class="track">{segments}</div>
      <div class="track-labels">{labels}</div>
    </div>
    """)


def probability_rows(entries, predicted: str) -> str:
    rows = ""
    for entry in entries:
        name = entry["class"]
        probability = entry["probability"]
        color = STAGE_COLORS.get(name, ACCENT)
        weight = "700" if name == predicted else "500"
        # A sliver of fill keeps a 0% row from reading as a rendering failure.
        width = max(probability * 100, 1.2)
        rows += f"""
        <div class="prob-row">
          <div class="prob-name" style="color:{color};">{name}</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{width}%; background:{color};"></div>
          </div>
          <div class="prob-value" style="font-weight:{weight};">{probability:.1%}</div>
        </div>
        <div class="prob-desc">{entry['description']}</div>
        """
    return _html(f'<div class="card"><div class="card-label">Stage probabilities</div>{rows}</div>')


def scan_panel(label: str, image_b64: str, caption: str) -> str:
    return _html(f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <img class="scan" src="data:image/png;base64,{image_b64}"/>
      <div class="scan-caption">{caption}</div>
    </div>
    """)


def audit_panel(focus: dict, threshold: float = 0.40) -> str:
    """Where the model looked, and whether that is anatomically plausible."""
    central = focus.get("central_share", 0.0)
    healthy = central >= threshold
    chip = (
        '<span class="chip chip-ok">Attending to tissue</span>'
        if healthy
        else '<span class="chip chip-warn">Attention off-tissue</span>'
    )
    note = (
        ""
        if healthy
        else f"""<div class="scan-caption" style="margin-top:0.9rem;">
        Under {threshold:.0%} of attribution falls in the central region. On a registered
        slice that usually means the model is reading background or an acquisition
        artefact rather than anatomy — treat this prediction as unverified.</div>"""
    )
    return _html(f"""
    <div class="card">
      <div class="card-label">Attention audit &nbsp; {chip}</div>
      <div class="audit">
        <div class="audit-item"><div class="k">Central mass</div>
          <div class="v">{central:.0%}</div></div>
        <div class="audit-item"><div class="k">Dominant quadrant</div>
          <div class="v">{str(focus.get('dominant_quadrant', '—')).replace('_', ' ')}</div></div>
        <div class="audit-item"><div class="k">Activated area</div>
          <div class="v">{focus.get('activated_fraction', 0):.0%}</div></div>
      </div>
      {note}
    </div>
    """)


def empty_state(sample_hint: str) -> str:
    return _html(f"""
    <div class="empty">
      <h3>Upload an MRI slice to begin</h3>
      <p>Drop a PNG or JPEG above. The model returns a stage, calibrated-looking
      probabilities for all four stages, and a Grad-CAM map showing which regions
      drove the result.</p>
      <p>{sample_hint}</p>
    </div>
    """)


def status_badge(ok: bool, message: str) -> str:
    color = STAGE_COLORS["NC"] if ok else STAGE_COLORS["EMCI"]
    background = STAGE_FILLS["NC"] if ok else STAGE_FILLS["EMCI"]
    return _html(f"""
    <div class="status" style="background:{background}; color:{color};">
      <div class="status-dot" style="background:{color};"></div>{message}
    </div>
    """)
