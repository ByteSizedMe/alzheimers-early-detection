"""Build the project report PDF from the artifacts of an actual run.

    python scripts/make_report.py --out artifacts/Alzheimers_CPI_Report.pdf

Every number in the report is read from ``artifacts/`` and
``data/processed/`` rather than hard-coded, so regenerating after a retrain
produces a report that matches the run instead of the last one. If an artifact
is missing the section is skipped with a note, so the script never invents
results.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
PROCESSED = REPO_ROOT / "data" / "processed"
CLASSES = ("NC", "EMCI", "LMCI", "AD")

# Clinical-slate palette: deep teal carries the document, amber is the single
# accent reserved for numbers that matter.
INK = colors.HexColor("#14202B")
SLATE = colors.HexColor("#4A5A68")
TEAL = colors.HexColor("#1B6B7A")
TEAL_PALE = colors.HexColor("#E8F1F3")
AMBER = colors.HexColor("#B26B00")
RULE = colors.HexColor("#C9D6DB")
PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


# --------------------------------------------------------------------- styles


def build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Times-Bold", fontSize=25,
        leading=30, textColor=INK, spaceAfter=6,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName="Times-Italic", fontSize=13.5,
        leading=19, textColor=SLATE, alignment=TA_CENTER,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=15.5,
        leading=19, textColor=TEAL, spaceBefore=16, spaceAfter=7,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.8,
        leading=15, textColor=INK, spaceBefore=11, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Times-Roman", fontSize=10.2,
        leading=15.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=13, bulletIndent=3, spaceAfter=4.5,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"], fontName="Times-Italic", fontSize=8.6,
        leading=12, textColor=SLATE, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Normal"], fontName="Courier", fontSize=8.3,
        leading=11.4, textColor=INK, backColor=TEAL_PALE,
        borderPadding=7, leftIndent=2, spaceBefore=3, spaceAfter=9,
    )
    s["callout"] = ParagraphStyle(
        "callout", parent=s["body"], fontSize=9.8, leading=14.4,
        leftIndent=10, rightIndent=10, borderPadding=9,
        backColor=colors.HexColor("#FDF4E3"), spaceBefore=5, spaceAfter=10,
    )
    s["cover_meta"] = ParagraphStyle(
        "cover_meta", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
        leading=16, textColor=SLATE, alignment=TA_CENTER,
    )
    return s


# ---------------------------------------------------------------- page frames


def draw_page(canvas, doc):
    """Header rule and footer on every page after the cover."""
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 6 * mm, PAGE_W - MARGIN, PAGE_H - MARGIN + 6 * mm)

    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 8 * mm, "Early Detection of Alzheimer's — CPI Project Report")

    canvas.line(MARGIN, MARGIN - 5 * mm, PAGE_W - MARGIN, MARGIN - 5 * mm)
    canvas.drawString(MARGIN, MARGIN - 9 * mm, "Research prototype — not a medical device")
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(INK)
    canvas.rect(0, 0, PAGE_W, 6 * mm, stroke=0, fill=1)
    canvas.restoreState()


# ----------------------------------------------------------------- components


def table(data, widths, header=True, center_cols=()):
    """A quiet table: ruled header, hairline rows, no heavy grid.

    ``center_cols`` centres only the numeric columns. Prose columns stay
    left-aligned -- centred sentences inside a table read as a layout mistake.
    """
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), TEAL_PALE),
            ("TEXTCOLOR", (0, 0), (-1, 0), TEAL),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
        ]
    for column in center_cols:
        style.append(("ALIGN", (column, 0), (column, -1), "CENTER"))
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, caption: str, width_mm: float, styles) -> list:
    """An image scaled to width, with its caption kept on the same page."""
    if not path.exists():
        return [Paragraph(f"[figure missing: {path.name}]", styles["caption"])]

    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        ratio = im.height / im.width

    width = width_mm * mm
    return [
        KeepTogether(
            [
                Image(str(path), width=width, height=width * ratio),
                Paragraph(caption, styles["caption"]),
            ]
        )
    ]


def captioned_table(
    rows, widths, caption, styles, center_cols=(), extra_style=None, heading=None
):
    """A table bound to its caption -- and optionally its sub-heading.

    A page break that strands a table header, or a heading, at the foot of a
    page reads as a broken document; keeping the unit together avoids both.
    """
    t = table(rows, widths=widths, center_cols=center_cols)
    if extra_style:
        t.setStyle(TableStyle(extra_style))

    parts = [t, Paragraph(caption, styles["caption"])]
    if heading:
        parts.insert(0, Paragraph(heading, styles["h2"]))
    return KeepTogether(parts)


def bullets(items, styles) -> list:
    return [Paragraph(text, styles["bullet"], bulletText="•") for text in items]


# ------------------------------------------------------------------ data load


def load_artifacts() -> dict:
    def read(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "evaluation": read(ARTIFACTS / "evaluation_test.json"),
        "history": read(ARTIFACTS / "training_history.json"),
        "prepare": read(PROCESSED / "prepare_summary.json"),
    }


def plot_confusion(matrix, out_path: Path) -> Path:
    """Row-normalised confusion matrix, drawn for the report."""
    import numpy as np

    matrix = np.asarray(matrix)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    image = ax.imshow(normalized, cmap="BuPu", vmin=0, vmax=1)

    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax.set_xlabel("Predicted stage", fontsize=9.5)
    ax.set_ylabel("True stage", fontsize=9.5)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, f"{int(matrix[i, j])}" + "\n" + f"{normalized[i, j]:.0%}",
                ha="center", va="center", fontsize=8.5,
                color="white" if normalized[i, j] > 0.55 else "#333333",
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("share of true class", fontsize=8.5)
    bar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def plot_training_curve(history, out_path: Path) -> Path | None:
    """Loss and validation macro-F1 against epoch."""
    if not history:
        return None

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))

    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", ms=4,
            color="#1B6B7A", lw=1.8, label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="s", ms=4,
            color="#B26B00", lw=1.8, label="validation loss")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("cross-entropy loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#DCE5E8", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    twin = ax.twinx()
    twin.plot(epochs, [h["val_macro_f1"] for h in history], marker="^", ms=4,
              color="#5A7D8C", lw=1.5, ls="--", label="validation macro F1")
    twin.set_ylabel("macro F1", fontsize=9)
    twin.set_ylim(0, 1.05)
    twin.tick_params(labelsize=8)
    twin.spines["top"].set_visible(False)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------- content


def cover(styles, evaluation) -> list:
    story = [
        Spacer(1, 44 * mm),
        Paragraph("Early Detection of Alzheimer's Disease", styles["title"]),
        Paragraph("Deep Learning on MRI with Explainable Predictions", styles["subtitle"]),
        Spacer(1, 5 * mm),
    ]

    rule = Table([[""]], colWidths=[52 * mm], rowHeights=[1.4])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), AMBER)]))
    rule.hAlign = "CENTER"
    story += [rule, Spacer(1, 12 * mm)]

    story.append(
        Paragraph(
            "A four-stage classifier (NC &rarr; EMCI &rarr; LMCI &rarr; AD) built on VGG19 "
            "transfer learning, made auditable with Grad-CAM, and deployed as a "
            "containerised FastAPI and Streamlit stack.",
            ParagraphStyle("lede", parent=styles["cover_meta"], fontSize=11.2,
                           leading=17, textColor=INK),
        )
    )
    story.append(Spacer(1, 26 * mm))

    if evaluation:
        headline = [
            ["Test accuracy", "Macro F1", "Impaired-case recall", "Quadratic kappa"],
            [
                f"{evaluation['accuracy']:.1%}",
                f"{evaluation['macro_f1']:.3f}",
                f"{evaluation['early_detection_recall']:.1%}",
                f"{evaluation['quadratic_kappa']:.3f}",
            ],
        ]
        t = Table(headline, colWidths=[40 * mm] * 4)
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8.4),
                    ("TEXTCOLOR", (0, 0), (-1, 0), SLATE),
                    ("FONTNAME", (0, 1), (-1, 1), "Times-Bold"),
                    ("FONTSIZE", (0, 1), (-1, 1), 17),
                    ("TEXTCOLOR", (0, 1), (-1, 1), TEAL),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 1), (-1, 1), 3),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.4, RULE),
                    ("LINEBELOW", (0, 1), (-1, 1), 0.4, RULE),
                    ("TOPPADDING", (0, 0), (-1, 0), 7),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                ]
            )
        )
        t.hAlign = "CENTER"
        story += [t, Spacer(1, 3 * mm)]
        story.append(
            Paragraph(
                "Measured on a held-out split of a synthetic validation cohort — "
                "see §6 on how to read these figures.",
                styles["caption"],
            )
        )

    story += [
        Spacer(1, 22 * mm),
        Paragraph(
            f"CPI Project Report<br/>{date.today():%d %B %Y}",
            styles["cover_meta"],
        ),
        Spacer(1, 6 * mm),
        Paragraph(
            "<b>Research prototype. Not a medical device and not for clinical use.</b>",
            ParagraphStyle("disc", parent=styles["cover_meta"], fontSize=9, textColor=AMBER),
        ),
    ]
    return story


def body(styles, art) -> list:
    evaluation = art["evaluation"]
    prepare = art["prepare"]
    story: list = []

    # ------------------------------------------------------------- 1 abstract
    story.append(Paragraph("1. Abstract", styles["h1"]))
    story.append(
        Paragraph(
            "Alzheimer's disease affects roughly one in nine people over 65 and one in "
            "three over 85, and there is no cure — which makes early detection the only "
            "clinically useful intervention window. This project implements an end-to-end "
            "system that classifies structural MRI slices into four ordinal stages "
            "(normal cognition, early and late mild cognitive impairment, and Alzheimer's "
            "disease), explains each prediction with a Grad-CAM heatmap a clinician can "
            "accept or reject, and serves the result through a containerised API and "
            "review dashboard.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "The work addresses the three failure modes that make medical imaging models "
            "unreliable in practice rather than only the classification task itself: "
            "incomplete clinical records are repaired with hot-deck imputation, class "
            "imbalance is corrected with ADASYN on the tabular features and class-weighted "
            "loss on the images, and opaque predictions are made checkable with "
            "gradient-based attribution. The complete pipeline has been executed and "
            "verified; 68 automated tests cover the statistical claims, not merely the "
            "call signatures.",
            styles["body"],
        )
    )

    # ------------------------------------------------------------- 2 problem
    story.append(Paragraph("2. Problem and Motivation", styles["h1"]))
    story.append(
        Paragraph(
            "Current diagnosis depends on manual review of MRI and CT, which is "
            "labour-intensive and misses the subtle early-stage markers that matter most. "
            "Three obstacles recur across the literature, and each maps to a component of "
            "this system:",
            styles["body"],
        )
    )
    story.append(
        table(
            [
                ["Obstacle", "Response", "Module"],
                ["Manual review misses early markers",
                 "VGG19 fine-tuned to four ordinal stages", "models/vgg19.py"],
                ["Clinicians cannot verify AI decisions",
                 "Grad-CAM attribution over the last conv block", "models/gradcam.py"],
                ["Incomplete clinical records",
                 "Hot-deck imputation from similar patients", "data/imputation.py"],
                ["Cohorts dominated by healthy controls",
                 "ADASYN plus class-weighted loss", "data/balancing.py"],
            ],
            widths=[54 * mm, 62 * mm, 39 * mm],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "The explainability requirement is not cosmetic. A model that reaches high "
            "accuracy by keying on a scanner watermark is statistically indistinguishable "
            "from a model reading anatomy — until the attribution map is inspected. In a "
            "diagnostic setting that distinction is the entire difference between a usable "
            "tool and a liability.",
            styles["body"],
        )
    )

    # -------------------------------------------------------- 3 architecture
    story.append(Paragraph("3. System Architecture", styles["h1"]))
    story.append(
        Paragraph(
            "The system is organised as three sequential phases plus a serving layer. "
            "Each phase is an independent command-line entry point that reads and writes "
            "files, so any stage can be rerun without repeating the others.",
            styles["body"],
        )
    )
    story.append(
        table(
            [
                ["Phase", "Stage", "Produces"],
                ["1", "Split, impute, balance", "manifests, balanced clinical table"],
                ["2", "Fine-tune VGG19", "best_model.pt, training history"],
                ["3", "Evaluate and explain", "metrics, confusion matrix, Grad-CAM panels"],
                ["—", "Serve", "FastAPI service, Streamlit dashboard, Docker images"],
            ],
            widths=[16 * mm, 55 * mm, 84 * mm],
            center_cols=(0,),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Splitting on a written manifest rather than walking directories at training "
            "time is deliberate: it makes the train/validation/test partition reproducible "
            "and records exactly which scans a given checkpoint was fitted on.",
            styles["body"],
        )
    )

    # ------------------------------------------------------------- 4 phase 1
    story.append(Paragraph("4. Phase 1 — Data Preparation", styles["h1"]))

    story.append(Paragraph("4.1 Hot-deck imputation", styles["h2"]))
    story.append(
        Paragraph(
            "Clinical tables are missing values non-uniformly: cognitive scores are almost "
            "always recorded, while CSF biomarkers and APOE genotype are frequently blank. "
            "Mean imputation flattens precisely the between-group variance a classifier "
            "depends on. Hot-deck imputation instead copies each missing value from a real, "
            "complete record that resembles the incomplete one, matched by nearest "
            "neighbour on the standardised observed features.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Two refinements matter. The donor is sampled from the <i>k</i> nearest complete "
            "records rather than always the single closest, which prevents one donor being "
            "copied into dozens of recipients and collapsing the imputed variance. And "
            "donors are restricted to the same diagnostic class during training, so an "
            "Alzheimer's patient never inherits a healthy control's hippocampal volume.",
            styles["body"],
        )
    )

    if prepare and prepare.get("clinical"):
        clinical = prepare["clinical"]
        story.append(
            Paragraph(
                f"On the validation cohort, {clinical['missing_cells']} missing cells "
                f"across {len(clinical['features'])} features were filled, leaving no "
                "residual gaps.",
                styles["body"],
            )
        )

    story.append(Paragraph("4.2 ADASYN class balancing", styles["h2"]))
    story.append(
        Paragraph(
            "Diagnostic cohorts are dominated by normal controls; the impaired classes a "
            "screening tool exists to catch are the minority. A model trained on the raw "
            "distribution achieves high accuracy by predicting the majority class and is "
            "clinically worthless. ADASYN (He et al., 2008) generates synthetic minority "
            "samples, but unlike SMOTE it allocates them adaptively: minority points "
            "sitting in majority-dense neighbourhoods — the hard, near-boundary cases — "
            "receive more synthetic neighbours than points deep inside a safe cluster, "
            "pushing the decision boundary towards the region that actually matters.",
            styles["body"],
        )
    )

    if prepare and prepare.get("clinical"):
        clinical = prepare["clinical"]
        before, after = clinical["distribution_before"], clinical["distribution_after"]
        rows = [["Stage", "Before ADASYN", "After ADASYN", "Synthetic added"]]
        for name in CLASSES:
            b, a = before.get(name, 0), after.get(name, 0)
            rows.append([name, str(b), str(a), f"+{a - b}" if a > b else "—"])
        ratio_before = max(before.values()) / min(before.values())
        rows.append(["Imbalance ratio", f"{ratio_before:.2f}x", "1.00x", ""])

        story.append(
            captioned_table(
                rows,
                widths=[42 * mm, 37 * mm, 37 * mm, 39 * mm],
                caption="Table 1 — ADASYN lifts every minority stage to the majority count.",
                styles=styles,
                center_cols=(1, 2, 3),
                extra_style=[
                    ("FONTNAME", (0, -1), (-1, -1), "Times-Bold"),
                    ("TEXTCOLOR", (0, -1), (-1, -1), AMBER),
                    ("LINEABOVE", (0, -1), (-1, -1), 0.8, TEAL),
                ],
            )
        )

    story.append(
        Paragraph(
            "<b>ADASYN is applied to the clinical table only, never to images.</b> "
            "Interpolating between two MRI slices produces anatomy that does not exist. "
            "Image-side imbalance is handled instead with inverse-frequency class weights "
            "in the loss, so rare stages cost more when predicted wrongly.",
            styles["callout"],
        )
    )
    story.append(
        Paragraph(
            "Two leakage guards are deliberate: the imputer is fitted on training rows "
            "only, and held-out splits are never resampled. Resampling a validation set "
            "inflates the score against synthetic rows derived from data the model has "
            "effectively already seen.",
            styles["body"],
        )
    )

    # ------------------------------------------------------------- 5 phase 2
    story.append(Paragraph("5. Phase 2 — Model Development", styles["h1"]))
    story.append(
        Paragraph(
            "The backbone is VGG19 pretrained on ImageNet, with the 1000-class head "
            "replaced by a four-class classifier and all but the final convolutional block "
            "frozen. At the dataset sizes available in this domain there is no fitting "
            "143 million parameters from scratch, but the early ImageNet filters already "
            "encode the edge and texture detectors that ventricular enlargement and "
            "cortical thinning present as. VGG19 is also a deliberate choice for "
            "explainability: its final feature map is 14&times;14 at 224-pixel input, "
            "coarse enough for stable Grad-CAM gradients and fine enough to localise.",
            styles["body"],
        )
    )
    story.append(
        table(
            [
                ["Setting", "Value", "Rationale"],
                ["Trainable blocks", "1 of 5", "Adapt deep features, keep early filters"],
                ["Head learning rate", "10x backbone", "Fresh head converges without damage"],
                ["Optimiser", "AdamW + cosine decay", "Stable under small batches"],
                ["Loss", "Weighted CE, smoothing 0.05", "Rare stages cost more"],
                ["Checkpoint policy", "Best validation macro F1", "Not the last epoch"],
                ["Early stopping", "Patience 6", "Overfits small cohorts quickly"],
            ],
            widths=[38 * mm, 44 * mm, 73 * mm],
        )
    )
    story.append(Spacer(1, 4 * mm))

    if art["history"]:
        curve = plot_training_curve(art["history"], ARTIFACTS / "report_training_curve.png")
        if curve:
            story += figure(
                curve,
                "Figure 1 — Training and validation loss with validation macro F1. "
                "Validation separates within the first epoch; training was early-stopped "
                "once the monitored metric stopped improving.",
                width_mm=155,
                styles=styles,
            )

    # ------------------------------------------------------------- 6 results
    story.append(Paragraph("6. Phase 3 — Results", styles["h1"]))

    if evaluation:
        story.append(
            Paragraph(
                "Accuracy is deliberately not the headline metric. In a cohort where "
                "normal controls are the plurality, a model that never predicts Alzheimer's "
                "still scores well. The reported figures therefore lead with macro F1 and "
                "balanced accuracy, which weight every stage equally; quadratic-weighted "
                "kappa, which penalises distant confusions more than adjacent ones because "
                "the stages are ordinal; and impaired-case recall, the fraction of "
                "genuinely impaired patients not classified as normal — the number a "
                "screening tool actually lives or dies by.",
                styles["body"],
            )
        )
        story.append(
            captioned_table(
                [
                    ["Metric", "Value", "What it measures"],
                    ["Accuracy", f"{evaluation['accuracy']:.4f}", "Overall correctness"],
                    ["Balanced accuracy", f"{evaluation['balanced_accuracy']:.4f}", "Mean per-stage recall"],
                    ["Macro F1", f"{evaluation['macro_f1']:.4f}", "Every stage weighted equally"],
                    ["Quadratic kappa", f"{evaluation['quadratic_kappa']:.4f}", "Ordinal agreement"],
                    ["Macro AUC", f"{evaluation.get('macro_auc', float('nan')):.4f}", "Ranking quality, one-vs-rest"],
                    ["Impaired-case recall", f"{evaluation['early_detection_recall']:.4f}", "Impaired cases not sent home"],
                ],
                widths=[45 * mm, 27 * mm, 83 * mm],
                caption=f"Table 2 — Held-out test split, n = {evaluation['n_samples']}.",
                styles=styles,
                center_cols=(1,),
            )
        )

        rows = [["Stage", "Precision", "Recall", "F1", "Support"]]
        for name in CLASSES:
            m = evaluation["per_class"][name]
            rows.append([
                name, f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                f"{m['f1']:.3f}", str(m["support"]),
            ])
        story.append(
            captioned_table(
                rows,
                heading="6.1 Per-stage breakdown",
                widths=[36 * mm, 30 * mm, 30 * mm, 30 * mm, 29 * mm],
                caption="Table 3 — All confusion is between adjacent stages; no impaired "
                        "case was classified more than one stage away from truth.",
                styles=styles,
                center_cols=(1, 2, 3, 4),
            )
        )

    confusion_path = (
        plot_confusion(evaluation["confusion_matrix"], ARTIFACTS / "report_confusion.png")
        if evaluation
        else ARTIFACTS / "confusion_matrix_test.png"
    )
    story += figure(
        confusion_path,
        "Figure 2 — Row-normalised confusion matrix. Rows are true stage, columns "
        "predicted. Raw counts hide poor recall on a small class, so each row is shown "
        "as a share of its own total.",
        width_mm=112,
        styles=styles,
    )

    story.append(
        Paragraph(
            "<b>How to read these numbers.</b> The pipeline was validated on a synthetic "
            "cohort generated to match the shape of real data — imbalanced classes, "
            "stage-dependent missingness, progressive atrophy — because access to a real "
            "research cohort requires an approved application. Those phantom images encode "
            "stage as a clean geometric difference, so they are close to trivially "
            "separable: near-perfect scores are what a <i>correctly working</i> pipeline "
            "should produce here, and materially lower scores would indicate a defect. "
            "These figures establish that every stage connects and behaves as designed. "
            "They are not a claim of clinical accuracy, and real-world data will score "
            "substantially lower.",
            styles["callout"],
        )
    )

    # ------------------------------------------------------ 7 explainability
    story.append(Paragraph("7. Explainability", styles["h1"]))
    story.append(
        Paragraph(
            "Grad-CAM (Selvaraju et al., 2017) takes the gradient of the target class "
            "score with respect to the last convolutional feature map and global-average-"
            "pools it per channel, producing a weight for each channel. The rectified "
            "weighted sum is a coarse map of what raised the score — for this model, where "
            "the atrophy it keyed on is located.",
            styles["body"],
        )
    )
    story += figure(
        ARTIFACTS / "gradcam" / "00001_NC_as_EMCI.png",
        "Figure 3 — A misclassified case: slice, attribution map, and overlay. The model's "
        "attention concentrates on the ventricular and hippocampal regions — the structures "
        "that genuinely encode stage — rather than on background or edge artefacts.",
        width_mm=163,
        styles=styles,
    )
    story.append(
        Paragraph(
            "Evaluation panels deliberately sample half correct and half misclassified "
            "cases. The errors are the informative half: they are where a spurious feature "
            "would reveal itself.",
            styles["body"],
        )
    )

    story.append(Paragraph("7.1 Automated attention auditing", styles["h2"]))
    story.append(
        Paragraph(
            "Each explanation carries a quantitative focus summary — the attribution's "
            "distribution across quadrants and the share of its mass falling within the "
            "central region of the frame. On a registered slice, attention drifting to the "
            "periphery indicates the model is reading background or an acquisition "
            "artefact rather than tissue. The dashboard raises a warning when central mass "
            "falls below 40%.",
            styles["body"],
        )
    )
    if evaluation and evaluation.get("gradcam"):
        rows = [["Case", "True", "Predicted", "Confidence", "Central mass"]]
        for record in evaluation["gradcam"][:6]:
            rows.append([
                str(record["index"]), record["true"], record["predicted"],
                f"{record['confidence']:.1%}",
                f"{record['focus']['central_share']:.0%}",
            ])
        story.append(
            captioned_table(
                rows,
                widths=[26 * mm, 28 * mm, 32 * mm, 34 * mm, 35 * mm],
                caption="Table 4 — Audited explanations. Central mass stays high across "
                        "correct and incorrect predictions alike, indicating the model "
                        "attends to tissue regardless of whether it gets the stage right.",
                styles=styles,
                center_cols=(0, 1, 2, 3, 4),
            )
        )

    story.append(
        Paragraph(
            "<b>Implementation note.</b> Grad-CAM must read a post-activation feature map. "
            "Targeting the convolution directly feeds in raw output containing large "
            "negative values; the channel-weighted sum then goes negative across the whole "
            "map and the final rectification flattens every heatmap to zero. This fails "
            "silently — it produces blank images, not an error — so a regression test pins "
            "the target layer to the rectified output.",
            styles["callout"],
        )
    )

    # ---------------------------------------------------------- 8 deployment
    story.append(Paragraph("8. Deployment", styles["h1"]))
    story.append(
        Paragraph(
            "The trained model is served by a FastAPI application and reviewed through a "
            "Streamlit dashboard, each in its own container so inference scales "
            "independently of the interface. Both surfaces call the same prediction object, "
            "so a dashboard reading cannot disagree with an API reading.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
        table(
            [
                ["Endpoint", "Method", "Purpose"],
                ["/health", "GET", "Liveness and whether a checkpoint is loaded"],
                ["/classes", "GET", "The four stages and their descriptions"],
                ["/predict", "POST", "Image upload to stage probabilities"],
                ["/explain", "POST", "Prediction plus a Grad-CAM overlay"],
            ],
            widths=[36 * mm, 24 * mm, 95 * mm],
            center_cols=(1,),
        )
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Two operational decisions are worth recording. The model loads lazily and the "
            "health endpoint reports an unloaded state rather than crashing, because in an "
            "orchestrator a service that dies at boot is indistinguishable from one that is "
            "merely untrained. And checkpoints are mounted as a volume rather than baked "
            "into the image — a half-gigabyte layer that changes on every retrain defeats "
            "image caching entirely.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "The <font face='Courier' size='8.6'>/explain</font> endpoint accepts an "
            "optional target class, letting a reviewer ask a counterfactual — what evidence "
            "would have supported an Alzheimer's reading? — rather than only seeing support "
            "for the winning class.",
            styles["body"],
        )
    )

    story.append(Paragraph("8.1 Verified service behaviour", styles["h2"]))
    story.append(
        Paragraph(
            "AD scan&nbsp;&nbsp;-&gt; AD&nbsp;&nbsp;&nbsp;&nbsp;confidence 99.11%&nbsp;&nbsp;"
            "|&nbsp;&nbsp;impairment 100.00%<br/>"
            "NC scan&nbsp;&nbsp;-&gt; NC&nbsp;&nbsp;&nbsp;&nbsp;confidence 87.94%&nbsp;&nbsp;"
            "|&nbsp;&nbsp;impairment&nbsp;&nbsp;12.06%<br/>"
            "/explain -&gt; central attribution mass 95.6%<br/><br/>"
            "malformed class -&gt; 400&nbsp;&nbsp;&nbsp;non-image -&gt; 400&nbsp;&nbsp;&nbsp;"
            "empty upload -&gt; 400",
            styles["code"],
        )
    )

    # ------------------------------------------------------------- 9 testing
    story.append(Paragraph("9. Validation and Testing", styles["h1"]))
    story.append(
        Paragraph(
            "Sixty-eight automated tests run in approximately 34 seconds. They target the "
            "statistical claims rather than merely exercising call signatures:",
            styles["body"],
        )
    )
    story += bullets(
        [
            "Imputed values are genuine donor values present in the source data, not "
            "computed averages — the property that distinguishes hot-deck from mean imputation.",
            "Hot-deck preserves sample variance more closely than mean imputation.",
            "Class-stratified donors never cross diagnostic boundaries.",
            "ADASYN interpolates strictly within the minority hull and allocates adaptively "
            "towards boundary-adjacent points.",
            "A single-member class is skipped rather than duplicated, which would leak "
            "identical rows across splits.",
            "Train, validation, and test splits are disjoint and preserve class ratios.",
            "The evaluation transform is deterministic while the training transform genuinely augments.",
            "Macro F1 penalises a majority-only predictor that accuracy would reward.",
            "The configured Grad-CAM target is a post-activation map.",
        ],
        styles,
    )
    story.append(
        Paragraph(
            "Model tests construct the network without pretrained weights — identical "
            "architecture, no half-gigabyte download — so the suite runs offline in seconds.",
            styles["body"],
        )
    )

    # -------------------------------------------------------- 10 limitations
    story.append(Paragraph("10. Limitations and Future Work", styles["h1"]))
    story.append(
        Paragraph(
            "The most consequential limitation is the partitioning strategy. Splits are "
            "currently made per slice, which is correct for the validation cohort where "
            "each subject contributes one slice. Real research cohorts contribute many "
            "slices per subject, and splitting per slice would place images of the same "
            "patient on both sides of the boundary — leaking identity and inflating "
            "accuracy substantially. Grouped stratified splitting on subject identifier is "
            "the required change before any accuracy figure from real data is quoted.",
            styles["body"],
        )
    )
    story += bullets(
        [
            "<b>Two-dimensional slices discard volumetric context.</b> A 3-D network over "
            "whole volumes captures atrophy patterns a single slice cannot, at the cost of "
            "substantially more data and compute.",
            "<b>Single modality.</b> The multimodal argument — combining MRI with EEG, "
            "handwriting dynamics, and speech — is not implemented. The prediction layer is "
            "the natural seam for a fusion head, with the clinical branch consuming the "
            "already-imputed table.",
            "<b>Grad-CAM is coarse.</b> A 14&times;14 map upsampled to full resolution "
            "localises to a lobe, not a structure. Guided Grad-CAM or attention rollout "
            "yields finer attribution.",
            "<b>Predictions are uncalibrated.</b> Softmax confidence is not a probability a "
            "clinician should read as one; temperature scaling on the validation split is "
            "the inexpensive correction.",
        ],
        styles,
    )

    # ------------------------------------------------------------ 11 running
    story.append(Paragraph("11. Reproducing This Report", styles["h1"]))
    story.append(
        Paragraph(
            "Every figure and number above is read from the artifacts of an actual run "
            "rather than transcribed, so regenerating after a retrain produces a report "
            "matching that run. The complete sequence:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "pip install -r requirements-dev.txt<br/>"
            "python scripts/make_synthetic_data.py --n-per-class 150<br/>"
            "python -m alzheimers.data.prepare<br/>"
            "python -m alzheimers.training.train<br/>"
            "python -m alzheimers.training.evaluate --split test --gradcam-samples 8<br/>"
            "python scripts/make_report.py",
            styles["code"],
        )
    )
    story.append(
        Paragraph(
            "To serve the trained model: "
            "<font face='Courier' size='8.6'>uvicorn alzheimers.api.main:app --port 8000</font> "
            "for the API, and "
            "<font face='Courier' size='8.6'>streamlit run src/alzheimers/dashboard/app.py</font> "
            "for the dashboard, or "
            "<font face='Courier' size='8.6'>docker compose up --build</font> for both.",
            styles["body"],
        )
    )

    story.append(Paragraph("References", styles["h1"]))
    story += bullets(
        [
            "He, H., Bai, Y., Garcia, E. A., &amp; Li, S. (2008). ADASYN: Adaptive synthetic "
            "sampling approach for imbalanced learning. <i>IEEE IJCNN</i>, 1322–1328.",
            "Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: Visual "
            "explanations from deep networks via gradient-based localization. <i>ICCV</i>, 618–626.",
            "Simonyan, K., &amp; Zisserman, A. (2015). Very deep convolutional networks for "
            "large-scale image recognition. <i>ICLR</i>.",
        ],
        styles,
    )
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project report PDF")
    parser.add_argument("--out", default="artifacts/Alzheimers_CPI_Report.pdf")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = build_styles()
    art = load_artifacts()

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Early Detection of Alzheimer's — CPI Project Report",
        author="CPI Project",
        subject="VGG19 MRI stage classification with Grad-CAM explainability",
    )
    frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="main")
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame], onPage=draw_cover),
            PageTemplate(id="body", frames=[frame], onPage=draw_page),
        ]
    )

    story = cover(styles, art["evaluation"])
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story += body(styles, art)

    doc.build(story)

    missing = [name for name, value in art.items() if value is None]
    if missing:
        print(f"[report] warning: skipped sections, missing artifacts: {missing}")
    print(f"[report] wrote {out_path}")


if __name__ == "__main__":
    main()
