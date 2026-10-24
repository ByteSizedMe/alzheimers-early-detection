# Early Detection of Alzheimer's — The AI Frontier

Implementation of the CPI project deck: a four-stage MRI classifier
(**NC → EMCI → LMCI → AD**) built on VGG19 transfer learning, made auditable
with Grad-CAM, and served as a containerised FastAPI + Streamlit stack.

> **Research prototype. Not a medical device, and not for clinical use.**

> **On hosting:** the interface needs a Python backend running PyTorch, so it
> cannot be served from GitHub Pages — Pages is static-only, and every
> prediction would fail. Run it locally with the commands below, or deploy the
> included `Dockerfile` to any host that runs containers.

---

## Why the pieces are what they are

The deck names four problems and four answers. Each one is a module here:

| Problem from the deck | What the code does | Where |
|---|---|---|
| Manual MRI review misses subtle early markers | VGG19 fine-tuned to 4 ordinal stages | [`models/vgg19.py`](src/alzheimers/models/vgg19.py) |
| AI decisions clinicians can't verify | Grad-CAM heatmaps over the last conv block | [`models/gradcam.py`](src/alzheimers/models/gradcam.py) |
| Incomplete clinical records | Hot-deck imputation from similar patients | [`data/imputation.py`](src/alzheimers/data/imputation.py) |
| Imbalanced cohorts (mostly healthy controls) | ADASYN oversampling + class-weighted loss | [`data/balancing.py`](src/alzheimers/data/balancing.py) |

Two design choices worth stating up front, because they are what keep the
numbers honest:

**Accuracy is not the headline metric.** A cohort that is ~42% normal controls
lets a model that never predicts AD score well. The reported metrics lead with
macro F1, balanced accuracy, quadratic-weighted kappa (the stages are ordinal —
calling AD "normal" should cost more than confusing EMCI with LMCI), and
`early_detection_recall`: the fraction of impaired patients not sent home.

**ADASYN runs on the clinical table, never on images.** Interpolating between
two MRI slices produces anatomy that does not exist. Image-side imbalance is
handled with inverse-frequency class weights in the loss instead.

---

## Quick start

```bash
git clone https://github.com/Harit117/alzheimers-early-detection.git
cd alzheimers-early-detection
pip install -r requirements-dev.txt
```

Grab the trained checkpoint (126 MB, shipped as a release asset because it
exceeds GitHub's file limit), then start the service:

```bash
python scripts/fetch_model.py
uvicorn alzheimers.api.main:app --port 8000
```

Open **http://localhost:8000**. Fourteen labelled sample scans ship in
[`data/samples/`](data/samples/) and appear in the interface as clickable
thumbnails, so there is nothing to download or generate before trying it.

Prefer to train your own? `make all` builds a synthetic cohort, runs the full
pipeline, and writes a fresh checkpoint in a few minutes on CPU.

`streamlit` is only needed for the dashboard; the pipeline, API, and tests
run without it.

No ADNI access yet? Generate a stand-in cohort with the same shape — imbalanced
classes, stage-dependent missingness, progressive atrophy in the images:

```bash
python scripts/make_synthetic_data.py --n-per-class 150
```

Then run the three phases from the deck's roadmap:

```bash
python -m alzheimers.data.prepare
python -m alzheimers.training.train
python -m alzheimers.training.evaluate --split test --gradcam-samples 8
```

`make all` runs all four steps. Every command takes `--config` to point at a
different YAML.

---

## Phase 1 — Data and preprocessing

```bash
python -m alzheimers.data.prepare
```

Splits the MRI set into stratified train/val/test manifests, hot-deck imputes
the clinical table, and ADASYN-balances the training rows. Sample output on the
synthetic cohort:

```
[images] found 285 slices: {'NC': 120, 'EMCI': 74, 'LMCI': 54, 'AD': 37}
[clinical] missing values: 290 cells
           csf_abeta                109  (38.2%)
           apoe4                     68  (23.9%)
[clinical] hot-deck imputed; residual NaN: 0
[clinical] train distribution {'NC': 86, 'EMCI': 53, 'LMCI': 35, 'AD': 26} (imbalance 3.31x)
[clinical] ADASYN -> {'AD': 86, 'EMCI': 86, 'LMCI': 86, 'NC': 86} (+144 synthetic)
```

**Hot-deck imputation** fills each gap with a real value copied from a similar
complete record, rather than a column mean. The donor is drawn from the *k*
nearest complete records (not always the single closest), which stops one donor
being copied into dozens of recipients and collapsing the imputed variance —
there is a test pinning exactly that property. Donors are matched within the
same diagnostic class during training, so an AD patient never inherits a healthy
control's hippocampal volume.

**ADASYN** (He et al., 2008) generates synthetic minority samples, weighting
generation towards minority points that sit in majority-dense neighbourhoods —
the hard, near-boundary cases. It is implemented directly on
`sklearn.neighbors`, so the project carries no `imbalanced-learn` dependency.

Two leakage guards are deliberate: the imputer is fitted on training rows only,
and validation/test sets are never resampled. Resampling a held-out split
inflates the score against synthetic rows the model has effectively already
seen.

---

## Phase 2 — Model development

```bash
python -m alzheimers.training.train --epochs 25
```

VGG19 pretrained on ImageNet, with the classifier replaced by a 4-class head
and all but the last conv block frozen. On an ADNI-scale dataset there is no
fitting 143M parameters from scratch, but the early ImageNet filters already
detect the edges and textures that ventricular enlargement and cortical
thinning present as.

Training details that matter:

- **Discriminative learning rates** — the fresh head runs at 10× the backbone's
  rate, so it converges without wrecking the pretrained features.
- **Class-weighted cross-entropy** with label smoothing, cosine LR decay, AdamW.
- **Checkpoint on best validation macro F1**, not the last epoch. These models
  overfit small cohorts within a handful of epochs.
- **Early stopping** on the same metric.

Writes `artifacts/best_model.pt`, `training_history.json`, and `test_metrics.json`.

---

## Phase 3 — Evaluation and Grad-CAM

```bash
python -m alzheimers.training.evaluate --split test --gradcam-samples 8
```

Produces a row-normalised confusion matrix, per-class precision/recall/F1, and
Grad-CAM panels in `artifacts/gradcam/` — slice, heatmap, and overlay side by
side, labelled with the true and predicted stage.

The panel sampler deliberately takes **half correct and half misclassified**
cases. The errors are the informative half: a model that hits 92% by keying on
a scanner watermark looks identical to a good one until you see where it looked.

Each panel carries a `focus` summary — the CAM's quadrant distribution and how
much of its mass falls in the central region of the frame. On a registered
slice, attention drifting to the edges means the model is reading background or
an acquisition artefact, not anatomy. The dashboard raises a warning when
central mass drops below 40%.

---

## Verified run (synthetic cohort)

The full pipeline was run end to end on a 285-slice synthetic cohort
(CPU, 12 epochs, early-stopped at 7):

```
[model] VGG19 on cpu -- 22,417,156 trainable / 33,002,308 total
[model] class weights {'NC': 0.495, 'EMCI': 0.815, 'LMCI': 1.093, 'AD': 1.598}

=== test split ===
accuracy 0.9535   macro F1 0.9653   quadratic kappa 0.9783
impaired-case recall 0.9600
                    -- predicted --
            NC   EMCI   LMCI     AD
NC          17      1      0      0
EMCI         1     11      0      0
LMCI         0      0      8      0
AD           0      0      0      5
```

**Read these numbers as a plumbing check, not a result.** The phantom images
encode stage as a clean geometric difference in ventricle size, so they are
close to trivially separable — near-perfect scores are what a *working*
pipeline should produce here, and anything lower would indicate a bug. Real
ADNI data will score substantially lower. What the run does establish is that
every stage connects: imputation fills 290 missing cells, ADASYN balances
3.31x imbalance to even, the model trains and early-stops on validation macro
F1, and the Grad-CAM maps land on the ventricular and hippocampal regions the
generator actually varies — which is the signal the model *should* be reading.

The live API was verified against the same checkpoint:

```
AD scan  -> AD    conf 99.11%  | impairment 100.00%
NC scan  -> NC    conf 87.94%  | impairment  12.06%
/explain -> central CAM mass 95.6%   (attention on tissue, not background)
```

88 tests pass in ~50s.

---

## Serving

```bash
uvicorn alzheimers.api.main:app --reload --port 8000
```

That one command serves both the API and the review interface at
**http://localhost:8000** — a bespoke single-page front end, no build step and no
framework. Drag a slice in, or click one of the labelled samples it discovers on
your Desktop.

Its centrepiece is a **draggable comparison slider**: scan on the left,
attribution on the right, wiped against each other rather than shown as two
separate pictures you have to mentally align. Alongside it sit the ordinal stage
track, per-stage probabilities, and the attention audit. Interface source is in
[`api/static/`](src/alzheimers/api/static/).

A Streamlit dashboard also remains, for anyone who prefers it:

```bash
streamlit run src/alzheimers/dashboard/app.py          # http://localhost:8501
```

Want scans to try it on? This copies a labelled set to your Desktop:

```bash
python scripts/export_samples.py
```

Files are drawn from the held-out test split and named by their true stage
(`AD_alzheimers_01.png`), so the folder is self-checking — the dashboard either
agrees with the filename or it does not. A couple of known-misclassified slices
are included as `MISREAD_*`: a demo where everything is correct teaches a
reviewer nothing about the failure modes.

| Endpoint | Purpose |
|---|---|
| `GET /` | The review interface |
| `GET /api` | Service metadata (the JSON that used to live at `/`) |
| `GET /samples` | Labelled sample scans the interface offers as one-click inputs |
| `GET /health` | Liveness, and whether a checkpoint is actually loaded |
| `GET /classes` | The four stages and their descriptions |
| `POST /predict` | Multipart image upload → class probabilities |
| `POST /explain` | Same, plus a base64 Grad-CAM overlay |
| `POST /handwriting/analyze` | Timestamped spiral-drawing strokes → risk indicator, per-feature contributions, speed-coloured trace |
| `GET /handwriting/demo/{profile}` | A simulated drawing (`typical`, `borderline`, `elevated`) the interface can replay |

`/explain` takes an optional `target_class` so a reviewer can ask a
counterfactual — *what would have made this AD?* — rather than only seeing
evidence for the winning class.

The model loads lazily and `/health` reports `model_loaded: false` when no
checkpoint exists, instead of the container crash-looping. In an orchestrator,
a service that dies at boot looks identical to one that is merely untrained.

The dashboard can run against the API or load a checkpoint in-process; both
paths call the same `Predictor`, so they cannot disagree.

Its layout is deliberately ordered — verdict, then where the model looked, then
the numbers. A confidence score shown alone invites the reviewer to accept it;
the attribution map placed directly beside it invites them to check it. An
attention-audit panel states how much of the Grad-CAM mass falls on tissue and
flags the prediction as unverified below 40%. Presentation lives in
[`dashboard/styles.py`](src/alzheimers/dashboard/styles.py), the Streamlit flow
in [`dashboard/app.py`](src/alzheimers/dashboard/app.py), and the dark clinical
theme in `.streamlit/config.toml`.

### Docker

```bash
docker compose up --build
```

Two images — API and dashboard — so inference scales independently of the UI.
Checkpoints are mounted as a volume rather than baked in; a 550 MB layer that
changes on every retrain defeats image caching.

---

## Handwriting modality (prototype)

The deck's multimodal argument names handwriting as a second signal. The
**Handwriting** tab in the interface adds it: the person draws a spiral from the
centre outward, and the browser records every pen point with its timestamp.

```bash
uvicorn alzheimers.api.main:app --port 8000      # then open the Handwriting tab
python -m alzheimers.handwriting.model            # retrain the handwriting model
```

**Pipeline.** [`handwriting/features.py`](src/alzheimers/handwriting/features.py)
turns a trace into nine scale-free kinematic features: drawing time, mean pen
speed, speed variability, stalled time, pen-up time, velocity peaks per unit
path (fluency), 4–12 Hz tremor amplitude, spiral irregularity, and drawing size.
[`handwriting/model.py`](src/alzheimers/handwriting/model.py) scores them with a
regularised logistic regression. A linear model is the point: every prediction
splits exactly into per-feature contributions (a test checks they reproduce the
model's logit), which the interface shows as a diverging bar chart, next to the
drawing recoloured by pen speed. It plays the role Grad-CAM plays for the MRI.

**Demoing it.** Draw live with a mouse, finger or stylus, or press
*Typical / Borderline / Elevated* to replay a simulated drawing in real time and
score it. After an MRI has been analysed in the same session, the handwriting
result also shows an illustrative combined view of the two indicators.

**What this is not.** No handwriting dataset ships here, so the classifier is
trained on drawings from [`handwriting/synthetic.py`](src/alzheimers/handwriting/synthetic.py).
That simulator encodes the *direction* of changes reported in the handwriting
literature (slower, more variable, more stalls and lifts, more tremor, a smaller
and wobblier spiral), not their size, and gives every subject their own baseline
so the classes overlap. The cross-validated accuracy (about 92%, AUC about 0.98)
therefore describes the simulator and says nothing about patients. A live mouse
drawing is also not a pen tablet: there is no pressure or in-air trajectory. To
make this real, record a labelled cohort (or use DARWIN, Cilia et al. 2022),
swap it in for `make_cohort`, and keep the feature code unchanged.

---

## Testing

```bash
python -m pytest -q
```

The suite covers the claims rather than just the call signatures: that imputed
values are genuine donor values and not averages, that hot-deck preserves
variance better than mean imputation, that ADASYN interpolates within the
minority hull and allocates adaptively, that the eval transform is
deterministic while the train transform actually augments, that splits are
disjoint, and that macro F1 punishes a majority-only predictor that accuracy
would reward.

Model tests build VGG19 with `pretrained=False` — identical architecture, no
548 MB download, so the suite runs offline in seconds.

---

## Using real ADNI data

1. Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu/) and download the
   T1-weighted MRI collection plus the clinical tables.
2. Preprocess volumes to 2-D slices (skull-strip, register to MNI space, take
   axial slices through the hippocampal region) into class-named folders:

   ```
   data/raw/{NC,EMCI,LMCI,AD}/*.png
   data/raw/clinical.csv        # numeric columns + a `label` column
   ```

3. Point `paths.raw_dir` and `paths.clinical_csv` at them and rerun Phase 1.

Nothing else changes — the manifest layer is why the pipeline does not care
whether the images are synthetic or real.

**Split by patient, not by slice.** Multiple slices from one subject landing on
both sides of a split leaks identity and inflates accuracy dramatically. The
current `stratified_split` splits on rows because the synthetic cohort has one
slice per subject; with real ADNI data, group the manifest by `subject_id`
first (`sklearn.model_selection.StratifiedGroupKFold`). This is the single most
important change when the real data arrives.

---

## Layout

```
configs/default.yaml              every knob, one file
scripts/make_synthetic_data.py    runnable stand-in cohort
scripts/export_samples.py         labelled sample scans for the Desktop
scripts/make_report.py            project report PDF, built from real artifacts
src/alzheimers/
├── config.py                     dotted-path config access
├── inference.py                  shared Predictor for API and dashboard
├── data/
│   ├── imputation.py             hot-deck
│   ├── balancing.py              ADASYN
│   ├── datasets.py               manifests, transforms, class weights
│   └── prepare.py                Phase 1 CLI
├── models/
│   ├── vgg19.py                  backbone + head + checkpointing
│   └── gradcam.py                explanations
├── training/
│   ├── train.py                  Phase 2 CLI
│   ├── evaluate.py               Phase 3 CLI
│   └── metrics.py                ordinal-aware metrics
├── handwriting/
│   ├── features.py               pen-trace kinematics
│   ├── synthetic.py              spiral simulator (stand-in cohort)
│   └── model.py                  logistic regression + explanations
├── api/
│   ├── main.py                   FastAPI
│   └── static/                   the review interface (HTML/CSS/JS, no build)
└── dashboard/
    ├── app.py                    Streamlit flow
    └── styles.py                 palette, CSS, HTML fragments
```

---

## Known limitations

- **Slice-level, not subject-level.** See the ADNI note above. Fix before
  quoting any accuracy figure.
- **2-D slices discard 3-D context.** A 3-D CNN over whole volumes captures
  atrophy patterns a single slice cannot; it also needs far more data.
- **Multimodal is partial.** MRI and handwriting are both implemented, but they
  are separate models joined only by an illustrative equal-weight average in the
  interface: no learned fusion, because no dataset pairs both signals for the same
  subjects. EEG and speech from the deck are not implemented. `Predictor` is the
  seam where a fusion head would attach.
- **Handwriting is trained on simulated drawings.** See the handwriting section.
- **Grad-CAM is coarse.** A 14×14 map upsampled to 224×224 localises to a lobe,
  not a structure. Guided Grad-CAM or attention rollout gives finer maps.
- **No calibration.** Softmax confidence is not a probability a clinician should
  read as one. Temperature scaling on the validation split is the cheap fix.
