import json, re, subprocess
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# pull the same content the deck uses (single source of truth)
raw = subprocess.check_output(["node", "-e", "const c=require('/tmp/claude-0/deck/content.js'); console.log(JSON.stringify(c))"]).decode()
C = json.loads(raw); PEOPLE, NOTES = C["PEOPLE"], C["NOTES"]
titles = {t["n"]: t["title"] for t in json.load(open("/tmp/claude-0/deck/titles.json"))}
WPM = 140
words = lambda t: len(t.split())

doc = Document()
sec = doc.sections[0]; sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(12); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(8); st.paragraph_format.line_spacing = 1.2
for name, size, color in (("Title", 26, "3B2A44"), ("Heading 1", 18, "5B3A6B"), ("Heading 2", 13, "3B2A44")):
    s = doc.styles[name]; s.font.name = "Cambria"; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = RGBColor.from_string(color)
    s.element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)

def para(text, bold=False, italic=False, size=None, color=None, style=None, after=None):
    p = doc.add_paragraph(style=style); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

doc.add_paragraph("Presentation scripts", style="Title")
para("Early Detection of Alzheimer's from MRI and Handwriting", italic=True, size=14, color="6B5A72", after=14)

def person_words(p):
    n = [s for s in range(p["slides"][0], p["slides"][1] + 1)]
    return sum(words(NOTES[str(s)]["text"]) for s in n)

# order table
para("Speaking order", bold=True, size=14, color="3B2A44", after=4)
tbl = doc.add_table(rows=1, cols=5); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER; tbl.style = "Table Grid"
for i, h in enumerate(["#", "Presenter", "Topic", "Slides", "About"]):
    c = tbl.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, "5B3A6B")
for k, p in enumerate(PEOPLE, 1):
    extra = words(NOTES["1"]["text"]) if p["key"] == "MS" else 0
    w = person_words(p) + extra
    sl = f"{p['slides'][0]}–{p['slides'][1]}" + (" (+1)" if p["key"] == "MS" else "")
    row = tbl.add_row().cells
    for i, v in enumerate([str(k), p["name"], p["role"], sl, f"{w} words, about {round(w / WPM * 60 / 5) * 5} sec"]):
        row[i].text = v
        for pp in row[i].paragraphs:
            for r in pp.runs: r.font.size = Pt(11)
    if k % 2 == 0:
        for c in row: shade(c, "F1ECF4")
doc.add_paragraph()
para("How to use this document", bold=True, size=14, color="3B2A44", after=4)
for t in [
    "Each script is written to be spoken, not read, and every presenter has the same length: about 115 words, roughly 50 seconds. Put it in your own words once you know it.",
    "Timing assumes about 140 words per minute. The same text is in the speaker notes of the matching slides.",
    "Every script ends by handing over to the next person by name. Keep those hand-offs; they make the talk feel like one story.",
    "Say the synthetic-data point in your own voice if you can: all data are synthetic, so the numbers describe the pipeline, not patients.",
    "If you are asked something outside your part, name the teammate who covered it.",
]:
    p = doc.add_paragraph(style="List Bullet"); p.add_run(t)

for k, p in enumerate(PEOPLE, 1):
    doc.add_page_break()
    doc.add_paragraph(f"{k}. {p['name']}", style="Heading 1")
    slides = list(range(p["slides"][0], p["slides"][1] + 1))
    shown = ([1] if p["key"] == "MS" else []) + slides
    w = sum(words(NOTES[str(s)]["text"]) for s in shown)
    secs = round(w / WPM * 60 / 5) * 5
    para(f"{p['role']}   ·   slides {p['slides'][0]}–{p['slides'][1]}" + ("  (plus the opening on slide 1)" if p["key"] == "MS" else "") + f"   ·   {w} words, about {secs} seconds", italic=True, color="6B5A72", after=10)
    for s in shown:
        doc.add_paragraph(f"Slide {s}: {titles[s]}", style="Heading 2")
        para(NOTES[str(s)]["text"])
    # cue box
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"; c = t.rows[0].cells[0]; shade(c, "F1ECF4"); c.text = ""
    nxt = PEOPLE[k]["name"].split()[0] if k < len(PEOPLE) else "questions and the live demo"
    r = c.paragraphs[0].add_run("Your cue to hand over: "); r.bold = True
    c.paragraphs[0].add_run(f"finish by naming {nxt}." if k < len(PEOPLE) else "finish by inviting questions and offering the live demo.")

# closing
doc.add_page_break()
doc.add_paragraph("Closing and live demo", style="Heading 1")
para("Slide 16   ·   Malhar leads, everyone answers questions   ·   about 0:15 plus Q&A", italic=True, color="6B5A72", after=10)
doc.add_paragraph(f"Slide 16: {titles[16]}", style="Heading 2")
para(NOTES["16"]["text"])
para("Suggested live demo (about two minutes): upload the healthy sample and show the verdict; drag the attribution slider; ask the counterfactual by choosing AD under Explain; open the Handwriting tab, draw a spiral, press Analyse, and point at the speed trace and the feature bars. If the live drawing is awkward, press Elevated to replay a simulated drawing.", after=8)

# footer note
doc.add_page_break()
doc.add_paragraph("Quick facts to have in your head", style="Heading 1")
facts = [
    "All data are synthetic: MRI slices are generated phantoms, drawings are simulated. Results describe the pipeline, not patients.",
    "The released checkpoint reproduces the reported test split exactly: 95.35% accuracy, macro F1 0.9653, identical confusion matrix (43 slices).",
    "On 1,065 fresh slices: 97.8% accuracy, macro F1 0.978 (95% CI 0.968 to 0.986). All 23 errors are between neighbouring stages.",
    "Early-detection recall 0.992: 5 of 615 impaired slices called normal.",
    "Grad-CAM audit flags a map if less than 40% of it falls in the central half of the frame. It flagged 0 of 710 maps.",
    "Noise (sigma 0.10) takes macro F1 from 0.987 to 0.648; brightness or contrast shifts cut early-detection recall to about 0.89.",
    "Handwriting: 9 features, logistic regression, AUC 0.962 on 600 held-out simulated drawings; at 4 px of noise accuracy falls to 64.8% while AUC stays 0.928.",
    "Latency on 4 CPU cores: handwriting 10 ms, MRI prediction 111 ms, with Grad-CAM 293 ms. About 9.4 predictions or 3.6 explanations per second per worker.",
    "Hot-deck imputation keeps variance near 1.0 (mean imputation: 0). ADASYN: no measurable downstream gain (change of -0.002 in macro F1, 14 of 30 splits ahead).",
]
for f in facts:
    p = doc.add_paragraph(style="List Bullet"); p.add_run(f)
doc.save("/tmp/claude-0/deck/Presentation_Scripts.docx")
print("docx saved")
