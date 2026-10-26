"""MRI classifier benchmarks: reproduction, fresh-cohort generalisation, robustness, Grad-CAM audit."""
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
import io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageFilter
sys.path.insert(0, str(REPO / "src"))
from alzheimers import CLASSES
from alzheimers.config import Config
from alzheimers.data.datasets import scan_image_directory, stratified_split
from alzheimers.inference import Predictor
from alzheimers.training.metrics import compute_metrics, early_detection_recall

B = Path(__file__).resolve().parents[1]
OUT = B / "results" / "mri.json"
cfg = Config.load()
pred = Predictor(checkpoint=str(REPO / "artifacts/best_model.pt"))
IDX = {c: i for i, c in enumerate(CLASSES)}
rng = np.random.default_rng(0)

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def probs_for(images):
    out = []
    for i in range(0, len(images), 16):
        batch = torch.stack([pred.transform(im.convert("RGB")) for im in images[i:i+16]])
        with torch.no_grad():
            out.append(torch.softmax(pred.model(batch), 1).numpy())
    return np.concatenate(out)

def ece(prob, y, bins=15):
    conf = prob.max(1); hit = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)

def summarise(prob, y, boot=2000):
    yhat = prob.argmax(1)
    m = compute_metrics(y, yhat, prob, CLASSES)
    m["early_detection_recall"] = early_detection_recall(y, yhat, CLASSES)
    m["ece"] = ece(prob, y)
    m["nll"] = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())
    onehot = np.eye(len(CLASSES))[y]
    m["brier"] = float(((prob - onehot) ** 2).sum(1).mean())
    m["n"] = int(len(y))
    # percentile bootstrap, resampling slices
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "quadratic_kappa", "early_detection_recall"]
    draws = {k: [] for k in keys}
    for _ in range(boot):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2: continue
        mm = compute_metrics(y[s], yhat[s], None, CLASSES)
        for k in keys[:4]: draws[k].append(mm[k])
        draws["early_detection_recall"].append(early_detection_recall(y[s], yhat[s], CLASSES))
    m["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
    return m

def load(manifest):
    return [Image.open(p).convert("RGB") for p in manifest["path"]], manifest["label"].map(IDX).to_numpy()

res = {"meta": {"checkpoint_metadata": {k: v for k, v in pred.metadata.items() if k != "config"}}}

# ---- E1: reproduce the README test split --------------------------------------
man = scan_image_directory(B / "data/seed42")
splits = stratified_split(man, float(cfg["data.val_fraction"]), float(cfg["data.test_fraction"]), int(cfg["seed"]))
res["e1_split_sizes"] = {k: len(v) for k, v in splits.items()}
imgs, y = load(splits["test"])
res["e1_readme_split"] = summarise(probs_for(imgs), y)
log("E1 done", res["e1_split_sizes"], res["e1_readme_split"]["accuracy"], res["e1_readme_split"]["macro_f1"])

# ---- E2: fresh independent cohorts ---------------------------------------------
fresh_probs, fresh_y, per_seed, fresh_imgs = [], [], {}, {}
for seed in (101, 102, 103):
    imgs, y = load(scan_image_directory(B / f"data/seed{seed}"))
    p = probs_for(imgs); per_seed[seed] = summarise(p, y, boot=500)
    fresh_probs.append(p); fresh_y.append(y); fresh_imgs[seed] = imgs
    log("E2 seed", seed, per_seed[seed]["accuracy"], per_seed[seed]["macro_f1"])
P = np.concatenate(fresh_probs); Y = np.concatenate(fresh_y)
res["e2_fresh_per_seed"] = per_seed
res["e2_fresh_pooled"] = summarise(P, Y)
log("E2 pooled", res["e2_fresh_pooled"]["accuracy"], res["e2_fresh_pooled"]["macro_f1"])

# ---- E2b: robustness (seed 101) ------------------------------------------------
base_imgs, base_y = fresh_imgs[101], fresh_y[0]
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def noise(im, s):
    a = np.asarray(im, float) / 255; a = np.clip(a + np.random.default_rng(1).normal(0, s, a.shape), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))
def bright(im, d):
    a = np.clip(np.asarray(im, float) / 255 + d, 0, 1); return Image.fromarray((a * 255).astype(np.uint8))
def contrast(im, f):
    a = np.asarray(im, float) / 255; a = np.clip((a - a.mean()) * f + a.mean(), 0, 1); return Image.fromarray((a * 255).astype(np.uint8))
perturb = {
    "clean": lambda im: im,
    "gauss_noise_0.05": lambda im: noise(im, 0.05),
    "gauss_noise_0.10": lambda im: noise(im, 0.10),
    "gauss_noise_0.20": lambda im: noise(im, 0.20),
    "blur_sigma_2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "jpeg_q30": lambda im: jpeg(im, 30),
    "brightness_+0.2": lambda im: bright(im, 0.2),
    "contrast_x0.6": lambda im: contrast(im, 0.6),
    "rotate_10deg": lambda im: im.rotate(10, fillcolor=(0, 0, 0)),
    "shift_20px": lambda im: im.transform(im.size, Image.AFFINE, (1, 0, 20, 0, 1, 0), fillcolor=(0, 0, 0)),
}
res["e2b_robustness"] = {}
for name, fn in perturb.items():
    p = probs_for([fn(im) for im in base_imgs]); yh = p.argmax(1)
    m = compute_metrics(base_y, yh, None, CLASSES)
    res["e2b_robustness"][name] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "early_detection_recall": early_detection_recall(base_y, yh, CLASSES)}
    log("E2b", name, res["e2b_robustness"][name])
json.dump(res, open(OUT, "w"), indent=1)

# ---- E3: Grad-CAM focus audit (seeds 101, 102) -----------------------------------
rows = []
for si, seed in enumerate((101, 102)):
    for im, y, p in zip(fresh_imgs[seed], fresh_y[si], fresh_probs[si]):
        out = pred.explain(im, None)
        rows.append({"seed": seed, "y": int(y), "pred": int(p.argmax()), "central": out["focus"]["central_share"],
                     "activated": out["focus"]["activated_fraction"], "quadrant": out["focus"]["dominant_quadrant"]})
    log("E3 seed", seed, "done")
df = pd.DataFrame(rows); df["correct"] = df.y == df.pred
res["e3_gradcam"] = {
    "n": int(len(df)),
    "central_mass_mean": float(df.central.mean()), "central_mass_median": float(df.central.median()),
    "central_mass_p05": float(df.central.quantile(0.05)), "central_mass_min": float(df.central.min()),
    "flagged_below_0.40": int((df.central < 0.40).sum()),
    "central_correct_mean": float(df[df.correct].central.mean()),
    "central_wrong_mean": float(df[~df.correct].central.mean()) if (~df.correct).any() else None,
    "n_wrong": int((~df.correct).sum()),
    "activated_fraction_by_true_class": {CLASSES[c]: float(df[df.y == c].activated.mean()) for c in range(4)},
}
json.dump(res, open(OUT, "w"), indent=1)
log("ALL DONE")
