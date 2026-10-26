const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("/root/.claude/skills/synced/f24784cc-22de-42f7-8dc6-e517398cb020_be301eb6-2ddc-4e52-a70e-b49ab56c98ab/pptx/scripts/apply_theme.js");
const { PEOPLE, NOTES } = require("./content.js");

const RES = "/home/user/alzheimers-early-detection/report/benchmarks/results/";
const mri = JSON.parse(fs.readFileSync(RES + "mri.json"));
const tab = JSON.parse(fs.readFileSync(RES + "tab.json"));
const hw = JSON.parse(fs.readFileSync(RES + "hw.json"));
const lat = JSON.parse(fs.readFileSync(RES + "lat.json"));
const SHOTS = "/tmp/claude-0/deck/shots/";
const SAMPLES = "/home/user/alzheimers-early-detection/data/samples/";

// ------------------------------------------------------------------ theme
const THEME = {
  name: "NeuroAD Mauve",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "F4F1F4", lt1: "0E0A0F", dk2: "B3A9B4", lt2: "1E1720",
    accent1: "C99BDB", accent2: "F43F5E", accent3: "34D399", accent4: "FBBF24", accent5: "FB923C", accent6: "8E7F91",
    hlink: "C99BDB", folHlink: "8E7F91",
  },
};
// hex copies for the few options that only accept hex (charts, shadows)
const HEX = { text: "F4F1F4", muted: "B3A9B4", dim: "8E7F91", bg: "0E0A0F", card: "1E1720", orchid: "C99BDB", grid: "3A303C",
  nc: "34D399", emci: "FBBF24", lmci: "FB923C", ad: "F43F5E" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Early Detection of Alzheimer's from MRI and Handwriting";
pres.author = PEOPLE.map((p) => p.name).join(", ");
const C = pres.SchemeColor;
const W = 13.333, L = 0.6, CW = 12.13;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE", background: { path: "/tmp/claude-0/deck/bg_title.png" },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.bg },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: L, y: 0.95, w: CW, h: 0.8, fontSize: 34, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Early detection of Alzheimer's  ·  MRI + handwriting  ·  research prototype, synthetic data", options: { x: L, y: 7.02, w: 9, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.02, w: 0.63, h: 0.3, fontSize: 10, color: HEX.dim, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSE", background: { path: "/tmp/claude-0/deck/bg_close.png" }, objects: [],
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function card(slide, x, y, w, h, name, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: opts.fill || C.background2, transparency: opts.transparency ?? 12 },
    line: { color: opts.lineColor || "FFFFFF", width: 0.75, transparency: opts.lineTransparency ?? 88 }, objectName: name,
  });
}
function T(slide, text, o) { slide.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top" }, o)); }
function badge(slide, person, extra) {
  // topic label only: slides carry no presenter names (they are listed in the script document)
  T(slide, (extra || person.role).toUpperCase(), { x: L, y: 0.42, w: 10.5, h: 0.4, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, valign: "middle", objectName: "Topic label" });
}
function framed(slide, img, x, y, w, h, name, alt) {
  card(slide, x - 0.08, y - 0.08, w + 0.16, h + 0.16, name + " frame", { transparency: 0, lineTransparency: 80 });
  slide.addImage({ path: img, x, y, w, h, altText: alt, objectName: name });
}
function bigStat(slide, x, y, w, h, value, label, color, name, valSize = 40) {
  card(slide, x, y, w, h, name);
  T(slide, value, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: h * 0.52, fontSize: valSize, bold: true, color: color || C.accent1, fontFace: "+mj-lt", valign: "middle", objectName: name + " value" });
  T(slide, label, { x: x + 0.25, y: y + h * 0.58, w: w - 0.5, h: h * 0.38, fontSize: 13, color: C.text2, objectName: name + " label" });
}
function chartStyle(extra) {
  return Object.assign({
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
    dataLabelColor: HEX.text, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, showLegend: false, showValue: true,
    titleColor: HEX.text, titleFontFace: "+mj-lt", titleFontSize: 15, valAxisLineShow: false, catAxisLineShow: true,
  }, extra);
}
const f1 = (x) => x.toFixed(3);

// =================================================================== slides
(async () => {
  const ico = {
    brain: await icon(fa.FaBrain, "0E0A0F"), eye: await icon(fa.FaEye, "0E0A0F"), db: await icon(fa.FaDatabase, "0E0A0F"), pen: await icon(fa.FaPenNib, "0E0A0F"),
    check: await icon(fa.FaCheckCircle, HEX.nc), warn: await icon(fa.FaExclamationTriangle, HEX.emci), bolt: await icon(fa.FaBolt, "0E0A0F"),
    layers: await icon(fa.FaLayerGroup, "0E0A0F"),
  };

  // ---------------------------------------------------------------- 1 title
  {
    pres.addSection({ title: "Opening" });
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
    T(s, "RESEARCH PROTOTYPE  ·  SYNTHETIC DATA", { x: L, y: 0.7, w: 8, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 4, objectName: "Kicker" });
    T(s, "Early Detection of Alzheimer's from MRI and Handwriting", { x: L, y: 1.25, w: 8.2, h: 2.35, fontSize: 46, bold: true, fontFace: "+mj-lt", valign: "top", objectName: "Deck title" });
    T(s, "Design and analysis of an explainable multimodal prototype", { x: L, y: 3.7, w: 7.6, h: 0.9, fontSize: 22, color: C.text2, objectName: "Deck subtitle" });
    // names
    const names = PEOPLE;
    const cw = 2.92, ch = 0.62, gx = 0.15, gy = 0.17, y0 = 5.2;
    names.forEach((p, i) => {
      const row = i < 4 ? 0 : 1, col = i < 4 ? i : i - 4;
      const x = L + col * (cw + gx), y = y0 + row * (ch + gy);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, rectRadius: 0.31, fill: { color: C.background2, transparency: 18 }, line: { color: "FFFFFF", width: 0.75, transparency: 82 }, objectName: "Name chip " + p.initials });
      s.addShape(pres.shapes.OVAL, { x: x + 0.1, y: y + 0.1, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "Chip dot " + p.initials });
      T(s, p.initials, { x: x + 0.1, y: y + 0.1, w: 0.42, h: 0.42, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", objectName: "Chip initials " + p.initials });
      T(s, p.name, { x: x + 0.62, y, w: cw - 0.7, h: ch, fontSize: 13, bold: true, valign: "middle", objectName: "Chip name " + p.initials });
    });
    s.addNotes(NOTES[1].text);
  }

  // helper to open a person's section
  const openSection = (p) => pres.addSection({ title: p.name });
  const P = Object.fromEntries(PEOPLE.map((p) => [p.key, p]));

  // ---------------------------------------------------------------- 3 Malhar: problem
  {
    openSection(P.MS);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.MS.name });
    badge(s, P.MS); s.addText("Early detection needs evidence you can check", { placeholder: "title" });
    const stages = [["NC", "Normal cognition", HEX.nc], ["EMCI", "Early MCI", HEX.emci], ["LMCI", "Late MCI", HEX.lmci], ["AD", "Alzheimer's", HEX.ad]];
    const pw = 2.78, gap = 0.34;
    stages.forEach((st, i) => {
      const x = L + i * (pw + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.95, w: pw, h: 0.62, rectRadius: 0.31, fill: { color: st[2], transparency: 82 }, line: { color: st[2], width: 1.25 }, objectName: "Stage pill " + st[0] });
      T(s, [{ text: st[0], options: { bold: true, color: st[2] } }, { text: "   " + st[1], options: { color: C.text1 } }], { x, y: 1.95, w: pw, h: 0.62, fontSize: 15, align: "center", valign: "middle", objectName: "Stage label " + st[0] });
      if (i < 3) T(s, "›", { x: x + pw, y: 1.95, w: gap, h: 0.62, fontSize: 24, color: C.text2, align: "center", valign: "middle", objectName: "Stage arrow " + (i + 1) });
    });
    // problems
    T(s, "THREE PROBLEMS", { x: L, y: 2.95, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Problems label" });
    const probs = ["Early MRI changes are subtle", "A bare label gives a clinician nothing to check", "Records have gaps, and healthy cases dominate"];
    probs.forEach((t, i) => {
      const y = 3.35 + i * 1.12;
      card(s, L, y, 5.7, 0.95, "Problem card " + (i + 1));
      s.addShape(pres.shapes.OVAL, { x: L + 0.22, y: y + 0.24, w: 0.47, h: 0.47, fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 }, objectName: "Problem number dot " + (i + 1) });
      T(s, String(i + 1), { x: L + 0.22, y: y + 0.24, w: 0.47, h: 0.47, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", objectName: "Problem number " + (i + 1) });
      T(s, t, { x: L + 0.9, y, w: 4.6, h: 0.95, fontSize: 16, valign: "middle", objectName: "Problem text " + (i + 1) });
    });
    // answers
    T(s, "OUR ANSWER", { x: 6.85, y: 2.95, w: 5.8, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Answer label" });
    const ans = [[ico.brain, "Stage classifier", "VGG19, four ordered stages"], [ico.eye, "Explanation", "Grad-CAM map + attention audit"], [ico.db, "Data pipeline", "Hot-deck imputation, ADASYN"], [ico.pen, "Second signal", "Spiral handwriting kinematics"]];
    ans.forEach((a, i) => {
      const col = i % 2, row = Math.floor(i / 2), x = 6.85 + col * 3.0, y = 3.35 + row * 1.7;
      card(s, x, y, 2.83, 1.55, "Answer card " + (i + 1));
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "Answer icon circle " + (i + 1) });
      s.addImage({ data: a[0], x: x + 0.33, y: y + 0.33, w: 0.29, h: 0.29, altText: a[1] + " icon", objectName: "Answer icon " + (i + 1) });
      T(s, a[1], { x: x + 0.2, y: y + 0.85, w: 2.45, h: 0.3, fontSize: 15, bold: true, objectName: "Answer title " + (i + 1) });
      T(s, a[2], { x: x + 0.2, y: y + 1.15, w: 2.45, h: 0.35, fontSize: 13, color: C.text2, objectName: "Answer text " + (i + 1) });
    });
    s.addNotes(NOTES[2].text);
  }

  // ---------------------------------------------------------------- 4 Malhar: architecture
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.MS.name });
    badge(s, P.MS); s.addText("One app, two signals, every answer explained", { placeholder: "title" });
    const bx = (x, y, w, h, t1, t2, name, fillC) => {
      card(s, x, y, w, h, name, fillC ? { fill: C.accent1, transparency: 85, lineColor: HEX.orchid, lineTransparency: 50 } : {});
      T(s, t1, { x: x + 0.1, y: y + 0.08, w: w - 0.2, h: 0.32, fontSize: 14, bold: true, align: "center", objectName: name + " title" });
      T(s, t2, { x: x + 0.1, y: y + 0.42, w: w - 0.2, h: h - 0.48, fontSize: 12, color: C.text2, align: "center", objectName: name + " detail" });
    };
    const arrow = (x, y1, y2, name) => s.addShape(pres.shapes.LINE, { x, y: y1, w: 0, h: y2 - y1, line: { color: HEX.orchid, width: 1.75, endArrowType: "triangle" }, objectName: name });
    bx(L, 1.95, 6.1, 0.85, "Browser", "MRI tab · handwriting canvas · results", "Arch browser");
    arrow(L + 3.05, 2.8, 3.1, "Arch arrow 1");
    bx(L, 3.1, 6.1, 0.85, "FastAPI service", "size limits · validation · typed errors · /health", "Arch api");
    arrow(L + 1.5, 3.95, 4.25, "Arch arrow 2"); arrow(L + 4.6, 3.95, 4.25, "Arch arrow 3");
    bx(L, 4.25, 2.95, 1.2, "MRI predictor", "VGG19 · probabilities · Grad-CAM", "Arch mri", true);
    bx(L + 3.15, 4.25, 2.95, 1.2, "Handwriting analyzer", "9 features · logistic regression", "Arch handwriting", true);
    arrow(L + 1.475, 5.45, 5.72, "Arch arrow 4"); arrow(L + 4.625, 5.45, 5.72, "Arch arrow 5");
    bx(L, 5.72, 2.95, 0.72, "Checkpoint", "best_model.pt · 132 MB", "Arch checkpoint");
    bx(L + 3.15, 5.72, 2.95, 0.72, "Saved model", "trains itself in about 6 s", "Arch hw model");
    framed(s, SHOTS + "home.png", 7.25, 1.98, 5.4, 3.375, "Prototype home screenshot", "Screenshot of the running prototype: upload zone and labelled sample scans");
    T(s, "The running prototype: upload a slice or click a labelled sample", { x: 7.25, y: 5.5, w: 5.4, h: 0.3, fontSize: 12, color: C.text2, objectName: "Screenshot caption" });
    card(s, 7.17, 5.95, 5.56, 0.78, "Synthetic data note", { fill: C.accent4, transparency: 88, lineColor: HEX.emci, lineTransparency: 55 });
    s.addImage({ data: ico.warn, x: 7.4, y: 6.17, w: 0.34, h: 0.34, altText: "Warning icon", objectName: "Warning icon" });
    T(s, "All data are synthetic: scores describe the pipeline, not patients", { x: 7.9, y: 5.95, w: 4.7, h: 0.78, fontSize: 14, bold: true, valign: "middle", objectName: "Synthetic data text" });
    s.addNotes(NOTES[3].text);
  }

  // ---------------------------------------------------------------- 5 Sathwik: data
  {
    openSection(P.SB);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.SB.name });
    badge(s, P.SB); s.addText("Why we generate our own data", { placeholder: "title" });
    const th = [["NC_normal-cognition_01.png", "NC", HEX.nc], ["EMCI_early-MCI_01.png", "EMCI", HEX.emci], ["LMCI_late-MCI_01.png", "LMCI", HEX.lmci], ["AD_alzheimers_01.png", "AD", HEX.ad]];
    th.forEach((t, i) => {
      const x = L + i * 1.58;
      s.addImage({ path: SAMPLES + t[0], x, y: 1.95, w: 1.45, h: 1.45, altText: "Phantom slice, stage " + t[1], objectName: "Phantom " + t[1] });
      T(s, t[1], { x, y: 3.45, w: 1.45, h: 0.3, fontSize: 14, bold: true, color: t[2], align: "center", objectName: "Phantom label " + t[1] });
    });
    card(s, L, 3.95, 6.2, 2.85, "Data bullets");
    const b = [["ADNI access takes weeks: a generator lets the whole pipeline run today", true], ["As the stage advances: ventricles larger, cortex thinner, hippocampus smaller", true], ["Clinical table of 7 measures; gaps rise with stage, CSF amyloid most often missing", false]];
    s.addText(b.map((x, i) => ({ text: x[0], options: { bullet: true, breakLine: i < b.length - 1, paraSpaceAfter: 10 } })), { x: L + 0.3, y: 4.15, w: 5.6, h: 2.45, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "middle", objectName: "Data bullet list" });
    card(s, 7.15, 1.95, 5.58, 4.85, "Class chart card");
    T(s, [{ text: "3.2 : 1", options: { bold: true, color: C.accent1, fontFace: "+mj-lt", fontSize: 30 } }, { text: "   class imbalance in the 285-slice cohort", options: { color: C.text2, fontSize: 14 } }], { x: 7.45, y: 2.1, w: 5.0, h: 0.6, valign: "middle", objectName: "Imbalance callout" });
    s.addChart(pres.charts.BAR, [{ name: "Slices", labels: ["NC", "EMCI", "LMCI", "AD"], values: [120, 74, 54, 37] }], chartStyle({
      x: 7.3, y: 2.8, w: 5.3, h: 3.9, barDir: "col", chartColors: [HEX.nc, HEX.emci, HEX.lmci, HEX.ad], barGapWidthPct: 45, valAxisHidden: true, valGridLine: { style: "none" },
      dataLabelPosition: "outEnd", dataLabelFontSize: 16, catAxisLabelFontSize: 14, showTitle: false,
    }));
    s.addNotes(NOTES[4].text);
  }

  // ---------------------------------------------------------------- 6 Sathwik: imputation + ADASYN
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.SB.name });
    badge(s, P.SB); s.addText("Hot-deck kept the spread; ADASYN did not help", { placeholder: "title" });
    const im = tab.imputation, ad = tab.adasyn;
    card(s, L, 1.95, 5.95, 4.85, "Imputation card");
    s.addChart(pres.charts.BAR, [{ name: "Variance ratio", labels: ["Column mean", "KNN (k=5)", "Class mean*", "Hot-deck, global", "Hot-deck, class"], values: [im["column mean"].var_ratio.mean, im["KNN (k=5)"].var_ratio.mean, im["class mean (oracle label)"].var_ratio.mean, im["hot-deck k=5, global"].var_ratio.mean, im["hot-deck k=5, class"].var_ratio.mean].map((v) => +v.toFixed(2)) }],
      chartStyle({ x: L + 0.15, y: 2.05, w: 5.65, h: 3.0, barDir: "bar", chartColors: [HEX.dim, HEX.dim, HEX.dim, HEX.orchid, HEX.orchid], barGapWidthPct: 40, showTitle: true, title: "Imputed variance ÷ true variance (ideal: 1.0)", valAxisMinVal: 0, valAxisMaxVal: 1.25, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", catAxisOrientation: "maxMin" }));
    T(s, [{ text: "Mean imputation: ", options: { bold: true } }, { text: "variance 0 and AD hippocampal volume 1.47 sd too large. " }, { text: "Hot-deck: ", options: { bold: true, color: C.accent1 } }, { text: "variance ≈ 1.0, bias −0.05 sd, at the cost of noisier single guesses." }], { x: L + 0.3, y: 5.2, w: 5.4, h: 1.45, fontSize: 14, objectName: "Imputation takeaway" });
    card(s, 6.78, 1.95, 5.95, 4.85, "ADASYN card");
    const order = ["no resampling", "class-weighted loss", "random oversampling", "ADASYN (beta=1)"];
    s.addChart(pres.charts.BAR, [{ name: "Macro F1", labels: ["None", "Class weights", "Random oversampling", "ADASYN"], values: order.map((k) => +ad[k].macro_f1.mean.toFixed(3)) }],
      chartStyle({ x: 6.93, y: 2.05, w: 5.65, h: 3.0, barDir: "bar", chartColors: [HEX.dim, HEX.dim, HEX.dim, HEX.orchid], barGapWidthPct: 40, showTitle: true, title: "Downstream macro F1, 30 splits", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin" }));
    const d = tab.adasyn_paired_vs_none["ADASYN (beta=1)"];
    T(s, [{ text: "ADASYN balances exactly ", options: { bold: true } }, { text: `(+${tab.adasyn_balance.synthetic_rows} synthetic rows, 3.26 → 1.00) ` }, { text: "but macro F1 moves ", options: {} }, { text: `${d.mean_delta_macro_f1.toFixed(3)} ± ${d.sd.toFixed(3)}`, options: { bold: true, color: C.accent1 } }, { text: `, ahead in ${d.wins} of ${d.n} splits: no measurable gain.` }], { x: 7.08, y: 5.2, w: 5.4, h: 1.45, fontSize: 14, objectName: "ADASYN takeaway" });
    T(s, "* Class mean uses the true label, so it is only a reference.", { x: L + 0.3, y: 6.5, w: 5.4, h: 0.25, fontSize: 10, color: C.text2, objectName: "Footnote" });
    s.addNotes(NOTES[5].text);
  }

  // ---------------------------------------------------------------- 7 Harit: model
  {
    openSection(P.HR);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.HR.name });
    badge(s, P.HR); s.addText("A pretrained VGG19, fine-tuned where it matters", { placeholder: "title" });
    const boxes = [["Input", "224 × 224 MRI slice", null], ["Conv blocks 1–4", "Frozen · ImageNet filters", "frozen"], ["Conv block 5", "Fine-tuned on MRI", "train"], ["New head", "25088 → 512 → 256 → 4 · dropout", "new"], ["Output", "4 stage probabilities", null]];
    const bw = 2.25, gap = 0.22;
    boxes.forEach((b, i) => {
      const x = L + i * (bw + gap);
      const opt = b[2] === "train" ? { fill: C.accent1, transparency: 70, lineColor: HEX.orchid, lineTransparency: 20 } : b[2] === "new" ? { fill: C.accent1, transparency: 88, lineColor: HEX.orchid, lineTransparency: 45 } : b[2] === "frozen" ? { fill: C.accent6, transparency: 70, lineTransparency: 70 } : {};
      card(s, x, 1.95, bw, 1.55, "VGG box " + (i + 1), opt);
      T(s, b[0], { x: x + 0.12, y: 2.1, w: bw - 0.24, h: 0.4, fontSize: 16, bold: true, align: "center", objectName: "VGG title " + (i + 1) });
      T(s, b[1], { x: x + 0.12, y: 2.55, w: bw - 0.24, h: 0.85, fontSize: 13, color: C.text2, align: "center", objectName: "VGG detail " + (i + 1) });
      if (i < 4) T(s, "›", { x: x + bw, y: 2.4, w: gap, h: 0.5, fontSize: 22, color: C.text2, align: "center", valign: "middle", objectName: "VGG arrow " + (i + 1) });
    });
    bigStat(s, L, 3.85, 3.9, 1.55, "33.0 M", "parameters in total", C.accent1, "Stat total");
    bigStat(s, L + 4.12, 3.85, 3.9, 1.55, "22.4 M", "trainable: block 5 + new head", C.accent1, "Stat trainable");
    bigStat(s, L + 8.24, 3.85, 3.89, 1.55, "4", "ordered stages: NC < EMCI < LMCI < AD", C.accent1, "Stat classes");
    card(s, L, 5.65, CW, 1.15, "ReLU note");
    s.addShape(pres.shapes.OVAL, { x: L + 0.3, y: 5.97, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "ReLU note icon circle" });
    s.addImage({ data: ico.layers, x: L + 0.43, y: 6.1, w: 0.24, h: 0.24, altText: "Layers icon", objectName: "ReLU note icon" });
    T(s, [{ text: "A small detail that matters: ", options: { bold: true } }, { text: "in-place ReLUs are replaced with normal ones, because in-place layers overwrite the activations that Grad-CAM must read." }], { x: L + 1.05, y: 5.65, w: 10.8, h: 1.15, fontSize: 15, valign: "middle", objectName: "ReLU note text" });
    s.addNotes(NOTES[6].text);
  }

  // ---------------------------------------------------------------- 8 Harit: training
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.HR.name });
    badge(s, P.HR); s.addText("How it was trained, and what we evaluated", { placeholder: "title" });
    card(s, L, 1.95, 6.6, 4.85, "Recipe card");
    T(s, "TRAINING RECIPE", { x: L + 0.3, y: 2.15, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Recipe label" });
    const rec = ["AdamW: backbone 1e-4, new head ten times higher", "Label smoothing 0.05, inverse-frequency class weights", "Cosine decay, up to 25 epochs, early stopping (patience 6)", "Checkpoint chosen by validation macro F1, not accuracy", "Mild augmentation: flip, 7° rotation, 3% shift, 10% jitter"];
    s.addText(rec.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < rec.length - 1, paraSpaceAfter: 22 } })), { x: L + 0.3, y: 2.65, w: 6.0, h: 4.0, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Recipe list" });
    bigStat(s, 7.4, 1.95, 2.62, 1.5, "Epoch 1", "selected checkpoint", C.accent1, "Stat epoch", 32);
    bigStat(s, 10.12, 1.95, 2.61, 1.5, "1.000", "validation macro F1", C.accent1, "Stat val F1", 32);
    card(s, 7.4, 3.65, 5.33, 1.75, "Saturated note", { fill: C.accent4, transparency: 88, lineColor: HEX.emci, lineTransparency: 55 });
    s.addImage({ data: ico.warn, x: 7.65, y: 3.9, w: 0.34, h: 0.34, altText: "Warning icon", objectName: "Saturated warning icon" });
    T(s, [{ text: "Validation is saturated. ", options: { bold: true } }, { text: "A perfect score at epoch 1 on 43 slices means model selection carried almost no information." }], { x: 8.2, y: 3.75, w: 4.35, h: 1.55, fontSize: 15, valign: "middle", objectName: "Saturated text" });
    card(s, 7.4, 5.6, 5.33, 1.2, "Not retrained note");
    T(s, [{ text: "We did not retrain. ", options: { bold: true, color: C.accent1 } }, { text: "Every result that follows comes from running the released checkpoint." }], { x: 7.7, y: 5.6, w: 4.8, h: 1.2, fontSize: 15, valign: "middle", objectName: "Not retrained text" });
    s.addNotes(NOTES[7].text);
  }

  // ---------------------------------------------------------------- 9 Shreya: headline results
  {
    openSection(P.SJ);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.SJ.name });
    badge(s, P.SJ); s.addText("Reproduced exactly, then tested on fresh data", { placeholder: "title" });
    const e1 = mri.e1_readme_split, pool = mri.e2_fresh_pooled, ps = mri.e2_fresh_per_seed;
    card(s, L, 1.95, 5.5, 2.35, "Reproduction card");
    T(s, "REPRODUCED", { x: L + 0.3, y: 2.1, w: 3, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Repro label" });
    T(s, (e1.accuracy * 100).toFixed(2) + "%", { x: L + 0.3, y: 2.4, w: 4.9, h: 0.9, fontSize: 48, bold: true, color: C.accent1, fontFace: "+mj-lt", valign: "middle", objectName: "Repro value" });
    T(s, `43-slice test split · macro F1 ${e1.macro_f1.toFixed(4)} · kappa ${e1.quadratic_kappa.toFixed(4)} · confusion matrix identical to the documentation`, { x: L + 0.3, y: 3.35, w: 4.9, h: 0.85, fontSize: 13, color: C.text2, objectName: "Repro detail" });
    card(s, L, 4.5, 5.5, 2.3, "Fresh cohorts card");
    T(s, "FRESH, UNSEEN COHORTS", { x: L + 0.3, y: 4.65, w: 4.5, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Fresh label" });
    T(s, (pool.accuracy * 100).toFixed(1) + "%", { x: L + 0.3, y: 4.95, w: 4.9, h: 0.9, fontSize: 48, bold: true, color: C.accent1, fontFace: "+mj-lt", valign: "middle", objectName: "Fresh value" });
    T(s, `1,065 slices from 3 new seeds · macro F1 ${pool.macro_f1.toFixed(3)} (95% CI ${pool.ci95.macro_f1[0].toFixed(3)}–${pool.ci95.macro_f1[1].toFixed(3)})`, { x: L + 0.3, y: 5.9, w: 4.9, h: 0.8, fontSize: 13, color: C.text2, objectName: "Fresh detail" });
    card(s, 6.35, 1.95, 6.38, 4.85, "Accuracy chart card");
    s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["Reported split (n=43)", "Seed 101", "Seed 102", "Seed 103", "Pooled (n=1,065)"], values: [e1.accuracy, ps["101"].accuracy, ps["102"].accuracy, ps["103"].accuracy, pool.accuracy].map((v) => +v.toFixed(3)) }],
      chartStyle({ x: 6.5, y: 2.05, w: 6.08, h: 4.65, barDir: "col", chartColors: [HEX.dim, HEX.orchid, HEX.orchid, HEX.orchid, HEX.nc], barGapWidthPct: 40, showTitle: true, title: "Accuracy by test set", valAxisMinVal: 0, valAxisMaxVal: 1.2, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0%", dataLabelFontSize: 14, catAxisLabelFontSize: 11 }));
    s.addNotes(NOTES[8].text);
  }

  // ---------------------------------------------------------------- 10 Shreya: errors
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.SJ.name });
    badge(s, P.SJ); s.addText("Every error is between neighbouring stages", { placeholder: "title" });
    const pool = mri.e2_fresh_pooled, cm = pool.confusion_matrix, cls = ["NC", "EMCI", "LMCI", "AD"], col = [HEX.nc, HEX.emci, HEX.lmci, HEX.ad];
    card(s, L, 1.95, 6.5, 4.85, "Matrix card");
    T(s, "Pooled confusion matrix · rows are the true stage", { x: L + 0.3, y: 2.1, w: 5.9, h: 0.3, fontSize: 14, bold: true, objectName: "Matrix heading" });
    const cell = (txt, o = {}) => ({ text: String(txt), options: Object.assign({ align: "center", valign: "middle", fontSize: 18, color: HEX.text, bold: false }, o) });
    const rows = [[cell("True \\ Pred", { fontSize: 12, color: HEX.muted }), ...cls.map((c, i) => cell(c, { bold: true, color: col[i], fontSize: 15 }))]];
    cm.forEach((r, i) => rows.push([cell(cls[i], { bold: true, color: col[i], fontSize: 15 }), ...r.map((v, j) => {
      const fill = v === 0 ? "241B26" : i === j ? "6F4E80" : (Math.abs(i - j) === 1 ? "8A3A52" : "B3203A");
      return cell(v, { fill: { color: fill }, bold: v > 0 && i !== j });
    })]));
    s.addTable(rows, { x: L + 0.3, y: 2.6, w: 5.9, colW: [1.5, 1.1, 1.1, 1.1, 1.1], rowH: 0.62, border: { type: "solid", color: HEX.bg, pt: 2 }, objectName: "Confusion matrix" });
    T(s, "14 early-MCI slices read as late MCI · 5 as normal · 3 normal as early MCI · 1 AD as late MCI", { x: L + 0.3, y: 5.85, w: 5.9, h: 0.85, fontSize: 13, color: C.text2, objectName: "Matrix note" });
    card(s, 7.3, 1.95, 5.43, 2.55, "Recall chart card");
    s.addChart(pres.charts.BAR, [{ name: "Recall", labels: cls, values: cls.map((c) => +pool.per_class[c].recall.toFixed(3)) }],
      chartStyle({ x: 7.4, y: 2.0, w: 5.23, h: 2.45, barDir: "bar", chartColors: col, barGapWidthPct: 35, showTitle: true, title: "Recall by stage", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin", catAxisLabelFontSize: 13 }));
    bigStat(s, 7.3, 4.7, 1.72, 2.1, "0.992", "early-detection recall", C.accent1, "Stat early recall", 26);
    bigStat(s, 9.11, 4.7, 1.72, 2.1, "23", "errors, all adjacent", C.accent1, "Stat errors", 26);
    bigStat(s, 10.92, 4.7, 1.81, 2.1, "98.6 vs 91.8", "accuracy vs mean confidence (%)", C.accent1, "Stat calibration", 20);
    s.addNotes(NOTES[9].text);
  }

  // ---------------------------------------------------------------- 11 Taneesha: Grad-CAM
  {
    openSection(P.TP);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.TP.name });
    badge(s, P.TP); s.addText("Explanations you can check, not just trust", { placeholder: "title" });
    const ih = 4.15, iw = ih * (1106 / 1312);
    framed(s, SHOTS + "mri_attribution.png", L + 0.08, 2.03, iw, ih, "Attribution screenshot", "Screenshot: scan beside its Grad-CAM attribution map with a draggable divider");
    framed(s, SHOTS + "mri_counterfactual.png", L + 0.08 + iw + 0.35, 2.03, iw, ih, "Counterfactual screenshot", "Screenshot: Grad-CAM map for the class AD on a healthy scan, almost entirely cold");
    T(s, "Drag to compare: scan vs map", { x: L + 0.08, y: 6.3, w: iw, h: 0.45, fontSize: 12, color: C.text2, objectName: "Caption 1" });
    T(s, "Counterfactual: “what would make this AD?”", { x: L + 0.08 + iw + 0.35, y: 6.3, w: iw, h: 0.45, fontSize: 12, color: C.text2, objectName: "Caption 2" });
    const x0 = 8.3, w0 = 4.43;
    const steps = [["Weight", "Gradient of the stage score weights the last conv feature maps"], ["Map", "Weighted sum + ReLU: warm regions pushed the score up"], ["Audit", "Central share of the map below 40% flags the prediction"]];
    steps.forEach((st, i) => {
      const y = 1.95 + i * 1.34;
      card(s, x0, y, w0, 1.2, "Step card " + (i + 1));
      s.addShape(pres.shapes.OVAL, { x: x0 + 0.2, y: y + 0.36, w: 0.48, h: 0.48, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "Step dot " + (i + 1) });
      T(s, String(i + 1), { x: x0 + 0.2, y: y + 0.36, w: 0.48, h: 0.48, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", objectName: "Step number " + (i + 1) });
      T(s, [{ text: st[0] + "  ", options: { bold: true, color: C.accent1 } }, { text: st[1] }], { x: x0 + 0.85, y, w: w0 - 1.05, h: 1.2, fontSize: 14, valign: "middle", objectName: "Step text " + (i + 1) });
    });
    const g = mri.e3_gradcam;
    bigStat(s, x0, 6.0, w0, 0.8, `0 of ${g.n} maps flagged`, "", C.accent1, "Audit stat", 18);
    s.addNotes(NOTES[10].text);
  }

  // ---------------------------------------------------------------- 12 Taneesha: robustness
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.TP.name });
    badge(s, P.TP); s.addText("Clean-data scores do not survive degraded images", { placeholder: "title" });
    const rb = mri.e2b_robustness;
    const keys = [["clean", "Clean"], ["rotate_10deg", "Rotation 10°"], ["shift_20px", "Shift 20 px"], ["contrast_x0.6", "Contrast × 0.6"], ["brightness_+0.2", "Brightness +0.2"], ["blur_sigma_2", "Blur σ = 2"], ["jpeg_q30", "JPEG quality 30"], ["gauss_noise_0.05", "Noise σ = 0.05"], ["gauss_noise_0.10", "Noise σ = 0.10"], ["gauss_noise_0.20", "Noise σ = 0.20"]];
    const vals = keys.map((k) => +rb[k[0]].macro_f1.toFixed(3));
    card(s, L, 1.95, 7.0, 4.85, "Robustness chart card");
    s.addChart(pres.charts.BAR, [{ name: "Macro F1", labels: keys.map((k) => k[1]), values: vals }],
      chartStyle({ x: L + 0.15, y: 2.05, w: 6.7, h: 4.65, barDir: "bar", chartColors: vals.map((v) => (v >= 0.95 ? HEX.nc : v >= 0.8 ? HEX.emci : HEX.ad)), barGapWidthPct: 30, showTitle: true, title: "Macro F1 on 355 slices after each degradation", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin", catAxisLabelFontSize: 12 }));
    const x0 = 7.9, w0 = 4.83;
    bigStat(s, x0, 1.95, w0, 1.5, "0.987 → 0.648", "macro F1 with Gaussian noise σ = 0.10", C.accent2, "Noise stat", 30);
    bigStat(s, x0, 3.6, w0, 1.5, "≈ 0.89", "early-detection recall under brightness or contrast shifts: impaired read as normal", C.accent4, "Recall stat", 32);
    framed(s, SHOTS + "mri_misread_verdict.png", x0 + 0.08, 5.41, w0 - 0.16, (w0 - 0.16) * (388 / 2248), "Misread screenshot", "Screenshot: a true early-MCI slice predicted as normal cognition with 78 percent confidence");
    T(s, "A real miss from the prototype: true EMCI read as NC", { x: x0, y: 6.45, w: w0, h: 0.35, fontSize: 12, color: C.text2, objectName: "Misread caption" });
    s.addNotes(NOTES[11].text);
  }

  // ---------------------------------------------------------------- 13 Nithyashree: handwriting capture
  {
    openSection(P.NH);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.NH.name });
    badge(s, P.NH); s.addText("Handwriting: the pen's motion, not the picture", { placeholder: "title" });
    const iw = 5.3, ih = iw * (1258 / 1520);
    framed(s, SHOTS + "hw_live_canvas.png", L + 0.08, 2.03, iw, ih, "Live canvas screenshot", "Screenshot: a spiral drawn live on the handwriting canvas");
    T(s, "A live drawing: every pen point is stored with its timestamp", { x: L + 0.08, y: 2.03 + ih + 0.15, w: iw, h: 0.3, fontSize: 12, color: C.text2, objectName: "Canvas caption" });
    const x0 = 7.05;
    T(s, "NINE MEASUREMENTS FROM (x, y, t)", { x: x0, y: 1.95, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Features label" });
    const feats = [["Drawing time", "seconds"], ["Mean pen speed", "path per second"], ["Speed variability", "stop-start motion"], ["Stalled time", "hesitation"], ["Pen-up time", "pen lifted"], ["Speed peaks", "fluency"], ["Tremor", "4–12 Hz wobble"], ["Spiral irregularity", "radial residual"], ["Drawing size", "micrographia"]];
    feats.forEach((f, i) => {
      const c = i % 3, r = Math.floor(i / 3), x = x0 + c * 1.9, y = 2.35 + r * 1.2;
      card(s, x, y, 1.78, 1.08, "Feature card " + (i + 1));
      T(s, f[0], { x: x + 0.12, y: y + 0.12, w: 1.54, h: 0.5, fontSize: 14, bold: true, objectName: "Feature name " + (i + 1) });
      T(s, f[1], { x: x + 0.12, y: y + 0.66, w: 1.54, h: 0.3, fontSize: 12, color: C.text2, objectName: "Feature unit " + (i + 1) });
    });
    card(s, x0, 6.0, 5.68, 0.8, "Scale note");
    T(s, "Lengths are relative to the canvas diagonal, so screen size does not matter. A mouse or finger gives no pressure.", { x: x0 + 0.2, y: 6.0, w: 5.3, h: 0.8, fontSize: 13, valign: "middle", objectName: "Scale note text" });
    s.addNotes(NOTES[12].text);
  }

  // ---------------------------------------------------------------- 14 Nithyashree: classifier
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.NH.name });
    badge(s, P.NH); s.addText("An interpretable score, with one big caveat", { placeholder: "title" });
    const ih = 4.5, iw = ih * (1102 / 1504);
    framed(s, SHOTS + "hw_elevated_trace.png", L + 0.08, 2.03, iw, ih, "Speed trace screenshot", "Screenshot: the drawing coloured by pen speed");
    framed(s, SHOTS + "hw_elevated_features.png", L + 0.08 + iw + 0.3, 2.03, iw, ih, "Feature contributions screenshot", "Screenshot: bar chart of each feature's contribution to the score");
    const x0 = 7.75, w0 = 4.98, h1 = hw.h1_heldout;
    bigStat(s, x0, 1.95, 2.4, 1.45, h1.auc.toFixed(3), "AUC · 600 held-out drawings", C.accent1, "AUC stat", 34);
    bigStat(s, x0 + 2.58, 1.95, 2.4, 1.45, (h1.accuracy * 100).toFixed(1) + "%", "accuracy at threshold 0.5", C.accent1, "Accuracy stat", 34);
    card(s, x0, 3.55, w0, 2.7, "Noise chart card");
    const rob = hw.h6_robustness;
    s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["Clean", "0.5 px", "1 px", "2 px", "4 px"], values: [rob.clean, rob["position_noise_0.5px"], rob["position_noise_1.0px"], rob["position_noise_2.0px"], rob["position_noise_4.0px"]].map((r) => +r.accuracy.toFixed(3)) }],
      chartStyle({ x: x0 + 0.1, y: 3.6, w: w0 - 0.2, h: 2.6, barDir: "col", chartColors: [HEX.orchid, HEX.orchid, HEX.orchid, HEX.orchid, HEX.ad], barGapWidthPct: 40, showTitle: true, title: "Accuracy as coordinate noise grows", titleFontSize: 14, valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0%", catAxisLabelFontSize: 12 }));
    card(s, x0, 6.4, w0, 0.45, "Caveat card", { fill: C.accent4, transparency: 88, lineColor: HEX.emci, lineTransparency: 55 });
    T(s, "Trained on simulated drawings: it learned our assumptions", { x: x0 + 0.15, y: 6.4, w: w0 - 0.3, h: 0.45, fontSize: 13, bold: true, valign: "middle", objectName: "Caveat text" });
    s.addNotes(NOTES[13].text);
  }

  // ---------------------------------------------------------------- 15 Arshi: performance
  {
    openSection(P.AG);
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.AG.name });
    badge(s, P.AG); s.addText("Responsive on four CPU cores, with one bottleneck", { placeholder: "title" });
    card(s, L, 1.95, 5.6, 3.1, "Latency card");
    s.addChart(pres.charts.BAR, [{ name: "Median ms", labels: ["Handwriting score", "MRI prediction", "Prediction + Grad-CAM"], values: [lat.hw_analyze_total.median_ms, lat.predict.median_ms, lat.explain.median_ms].map((v) => +v.toFixed(0)) }],
      chartStyle({ x: L + 0.1, y: 2.0, w: 5.4, h: 3.0, barDir: "bar", chartColors: [HEX.orchid, HEX.orchid, HEX.orchid], barGapWidthPct: 40, showTitle: true, title: "Median latency in milliseconds (4 CPU cores)", valAxisMinVal: 0, valAxisMaxVal: 360, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", catAxisOrientation: "maxMin", catAxisLabelFontSize: 13 }));
    const c = lat.http_concurrency;
    bigStat(s, L, 5.25, 1.8, 1.55, c.predict_1.req_per_s.toFixed(1), "predictions / s per worker", C.accent1, "Stat predictions", 30);
    bigStat(s, L + 1.9, 5.25, 1.8, 1.55, c.explain_1.req_per_s.toFixed(1), "explanations / s per worker", C.accent1, "Stat explanations", 30);
    bigStat(s, L + 3.8, 5.25, 1.8, 1.55, lat.server_cold_start_s.toFixed(1) + " s", "cold start to ready", C.accent1, "Stat cold start", 28);
    const iw = 5.6;
    framed(s, SHOTS + "mri_ad_full.png", 12.73 - iw + 0.08, 2.03, iw - 0.16, (iw - 0.16) * (1800 / 2880), "Interface screenshot", "Screenshot of a finished result: verdict, stage track, attribution slider and probabilities");
    T(s, "The finished interface (real screenshot)", { x: 12.73 - iw + 0.08, y: 2.03 + (iw - 0.16) * (1800 / 2880) + 0.14, w: iw - 0.16, h: 0.3, fontSize: 12, color: C.text2, objectName: "Interface caption" });
    card(s, 12.73 - iw, 6.05, iw, 0.8, "Bottleneck note", { fill: C.accent4, transparency: 88, lineColor: HEX.emci, lineTransparency: 55 });
    T(s, [{ text: "Bottleneck: ", options: { bold: true } }, { text: "more clients do not add throughput; each request runs a blocking forward pass in one worker." }], { x: 12.73 - iw + 0.15, y: 6.05, w: iw - 0.3, h: 0.8, fontSize: 13, valign: "middle", objectName: "Bottleneck text" });
    s.addNotes(NOTES[14].text);
  }

  // ---------------------------------------------------------------- 16 Arshi: limits and next
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: P.AG.name });
    badge(s, P.AG); s.addText("What we can claim, and what comes next", { placeholder: "title" });
    card(s, L, 1.95, 5.9, 2.5, "Claim card");
    T(s, "WHAT WE CAN CLAIM", { x: L + 0.3, y: 2.1, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.nc || HEX.nc, charSpacing: 3, objectName: "Claim label" });
    const yes = ["Works end to end and reproduces exactly", "Handwriting explanation is exact; MRI map is checkable", "Fast enough to demo on a laptop"];
    yes.forEach((t, i) => { s.addImage({ data: ico.check, x: L + 0.3, y: 2.6 + i * 0.58, w: 0.3, h: 0.3, altText: "Check icon", objectName: "Claim icon " + (i + 1) }); T(s, t, { x: L + 0.75, y: 2.55 + i * 0.58, w: 4.9, h: 0.4, fontSize: 15, valign: "middle", objectName: "Claim text " + (i + 1) }); });
    card(s, L, 4.6, 5.9, 2.2, "Cannot claim card");
    T(s, "WHAT WE CANNOT CLAIM", { x: L + 0.3, y: 4.75, w: 5, h: 0.3, fontSize: 12, bold: true, color: HEX.emci, charSpacing: 3, objectName: "Cannot label" });
    const no = ["Clinical performance: all data are synthetic", "Robustness: noise and blur break it", "A trained multimodal model: the combined score is an average"];
    no.forEach((t, i) => { s.addImage({ data: ico.warn, x: L + 0.3, y: 5.22 + i * 0.5, w: 0.3, h: 0.3, altText: "Warning icon", objectName: "Cannot icon " + (i + 1) }); T(s, t, { x: L + 0.75, y: 5.17 + i * 0.5, w: 4.9, h: 0.4, fontSize: 14, valign: "middle", objectName: "Cannot text " + (i + 1) }); });
    card(s, 6.75, 1.95, 5.98, 4.85, "Next steps card");
    T(s, "NEXT STEPS, IN ORDER", { x: 7.05, y: 2.1, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3, objectName: "Next label" });
    const nxt = ["Real data (ADNI, DARWIN) with subject-level splits", "Calibration and thresholds for the cost of a miss", "Robustness training: noise, blur, compression", "Faithfulness tests for the explanations", "A learned fusion once paired data exists", "Engineering: browser tests, CI, pinned versions"];
    nxt.forEach((t, i) => {
      const y = 2.55 + i * 0.7;
      s.addShape(pres.shapes.OVAL, { x: 7.05, y: y + 0.08, w: 0.44, h: 0.44, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "Next dot " + (i + 1) });
      T(s, String(i + 1), { x: 7.05, y: y + 0.08, w: 0.44, h: 0.44, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", objectName: "Next number " + (i + 1) });
      T(s, t, { x: 7.7, y, w: 4.85, h: 0.6, fontSize: 15, valign: "middle", objectName: "Next text " + (i + 1) });
    });
    s.addNotes(NOTES[15].text);
  }

  // ---------------------------------------------------------------- 17 closing
  {
    pres.addSection({ title: "Closing" });
    const s = pres.addSlide({ masterName: "CLOSE", sectionTitle: "Closing" });
    T(s, "Thank you", { x: L, y: 1.5, w: 8, h: 1.3, fontSize: 60, bold: true, fontFace: "+mj-lt", valign: "middle", objectName: "Closing title" });
    T(s, "Questions, and a live demo", { x: L, y: 2.85, w: 8, h: 0.7, fontSize: 26, color: C.accent1, objectName: "Closing subtitle" });
    const cw = 2.85, ch = 0.55, gx = 0.17, gy = 0.17, y0 = 4.35;
    PEOPLE.forEach((p, i) => {
      const row = i < 4 ? 0 : 1, col = i < 4 ? i : i - 4, x = L + col * (cw + gx), y = y0 + row * (ch + gy);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, rectRadius: 0.27, fill: { color: C.background2, transparency: 18 }, line: { color: "FFFFFF", width: 0.75, transparency: 82 }, objectName: "Closing chip " + p.initials });
      T(s, p.name, { x: x + 0.2, y, w: cw - 0.3, h: ch, fontSize: 14, bold: true, valign: "middle", objectName: "Closing name " + p.initials });
    });
    T(s, "Research prototype · not a medical device · all data synthetic", { x: L, y: 6.75, w: 9, h: 0.3, fontSize: 12, color: C.text2, objectName: "Closing disclaimer" });
    T(s, "github.com/ByteSizedMe/alzheimers-early-detection", { x: L, y: 7.05, w: 9, h: 0.3, fontSize: 12, color: C.accent1, objectName: "Closing link" });
    s.addNotes(NOTES[16].text);
  }

  const out = "/tmp/claude-0/deck/Alzheimers_Presentation.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})().catch((e) => { console.error(e); process.exit(1); });
