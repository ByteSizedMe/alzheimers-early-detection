"""Handwriting benchmarks: held-out performance, baselines, ablation, severity response, robustness."""
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, str(REPO / "src"))
from alzheimers.handwriting.features import FEATURE_NAMES, extract_features
from alzheimers.handwriting.model import _transform, train, TIERS
from alzheimers.handwriting.synthetic import generate_drawing, make_cohort, CANVAS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
B = Path(__file__).resolve().parents[1]
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
rng_b = np.random.default_rng(5)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=1000))

# ---- training cohort (the shipped recipe: 300 per class, seed 0) and the shipped CV figures
t0 = time.time(); bundle = train(300, 0); train_seconds = time.time() - t0
Xtr, ytr, str_ = make_cohort(300, 0)
res = {"train_wall_seconds": train_seconds, "cv_shipped_recipe": bundle["metrics"], "n_train": int(len(ytr))}

# ---- held-out cohort with retained strokes (seed 999); identical sampling procedure to make_cohort
def cohort_strokes(n_per_class, seed):
    rng = np.random.default_rng(seed); out = []
    for label, (mu, sd) in enumerate([(0.15, 0.12), (0.60, 0.20)]):
        for _ in range(n_per_class):
            sev = float(np.clip(rng.normal(mu, sd), 0, 1))
            try:
                s = generate_drawing(sev, rng); extract_features(s, CANVAS)
            except ValueError: continue
            out.append((s, label, sev))
    return out
held = cohort_strokes(300, 999)
yh = np.array([l for _, l, _ in held]); sev = np.array([v for _, _, v in held])
def feats(strokes_list, canvas=CANVAS):
    rows = []
    for s in strokes_list:
        try: f = extract_features(s, canvas); rows.append([f[n] for n in FEATURE_NAMES])
        except ValueError: rows.append([np.nan] * len(FEATURE_NAMES))
    return np.array(rows)
Xh = feats([s for s, _, _ in held]); assert not np.isnan(Xh).any()
Xm, ym, _ = make_cohort(300, 999); assert np.allclose(Xm, Xh), "held-out features must match make_cohort"
res["n_heldout"] = int(len(yh))

pipe = bundle["pipeline"]
def evaluate(p, y, boot=2000):
    pred = (p >= 0.5).astype(int)
    m = {"accuracy": float((pred == y).mean()), "sensitivity": float((pred[y == 1] == 1).mean()),
         "specificity": float((pred[y == 0] == 0).mean()), "auc": float(roc_auc_score(y, p))}
    d = {k: [] for k in m}
    for _ in range(boot):
        s = rng_b.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2: continue
        pr = pred[s]; ys = y[s]
        d["accuracy"].append((pr == ys).mean()); d["sensitivity"].append((pr[ys == 1] == 1).mean())
        d["specificity"].append((pr[ys == 0] == 0).mean()); d["auc"].append(roc_auc_score(ys, p[s]))
    m["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
    return m
p_h = pipe.predict_proba(_transform(Xh))[:, 1]
res["h1_heldout"] = evaluate(p_h, yh); log("H1", {k: round(v, 3) for k, v in res["h1_heldout"].items() if k != "ci95"})
tn = lambda p: float(np.mean([next(l for lim, l in TIERS if q < lim) for q in p] == np.array(["Moderate"] * len(p))))
tiers = np.array([next(l for lim, l in TIERS if q < lim) for q in p_h])
res["h5_tiers"] = {g: {t: float(np.mean(tiers[yh == gi] == t)) for t in ("Low", "Moderate", "High")} for gi, g in ((0, "typical"), (1, "impaired"))}
res["h5_severity_spearman"] = float(spearmanr(p_h, sev).statistic)
res["h5_moderate_share_overall"] = float(np.mean(tiers == "Moderate"))

# ---- baselines
res["h2_baselines"] = {"majority_class_accuracy": float(max(np.mean(yh == 0), np.mean(yh == 1)))}
single = {}
for i, n in enumerate(FEATURE_NAMES):
    a = roc_auc_score(yh, Xh[:, i]); single[n] = float(max(a, 1 - a))
res["h2_single_feature_auc"] = single
gb = GradientBoostingClassifier(random_state=0).fit(_transform(Xtr), ytr); pg = gb.predict_proba(_transform(Xh))[:, 1]
res["h2_gradient_boosting"] = {"accuracy": float(((pg >= 0.5) == yh).mean()), "auc": float(roc_auc_score(yh, pg))}
log("H2 GB", res["h2_gradient_boosting"], "best single", max(single, key=single.get), round(max(single.values()), 3))

# ---- coefficients (standardised) of the shipped-recipe model
coef = pipe.named_steps["logisticregression"].coef_[0]
res["h4_coefficients"] = {n: float(c) for n, c in zip(FEATURE_NAMES, coef)}

# ---- drop-one-feature ablation (retrain on training cohort, score held-out)
abl = {}
for i, n in enumerate(FEATURE_NAMES):
    keep = [j for j in range(len(FEATURE_NAMES)) if j != i]
    m = lr().fit(_transform(Xtr)[:, keep], ytr); p = m.predict_proba(_transform(Xh)[:, keep])[:, 1]
    abl[n] = {"auc": float(roc_auc_score(yh, p)), "accuracy": float(((p >= 0.5) == yh).mean())}
res["h3_drop_one"] = abl
log("H3 done")

# ---- robustness on held-out strokes (model fixed; only the input is degraded)
def score(list_of_strokes, canvas=CANVAS):
    X = feats(list_of_strokes, canvas); ok = ~np.isnan(X).any(1)
    p = np.full(len(X), np.nan); p[ok] = pipe.predict_proba(_transform(X[ok]))[:, 1]
    return p, ok
def metrics_ok(p, ok):
    y = yh[ok]; pr = (p[ok] >= 0.5).astype(int)
    return {"accuracy": float((pr == y).mean()), "auc": float(roc_auc_score(y, p[ok])), "rejected": int((~ok).sum())}
rob = {"clean": metrics_ok(p_h, np.ones(len(p_h), bool))}
def subsample(k): return [[s[::k] for s in strokes] for strokes, _, _ in held]
for k in (2, 4):
    rob[f"subsample_x{k}"] = metrics_ok(*score(subsample(k)))
rng = np.random.default_rng(3)
for sg in (0.5, 1.0, 2.0, 4.0):
    noisy = [[np.column_stack([s[:, 0] + rng.normal(0, sg, len(s)), s[:, 1] + rng.normal(0, sg, len(s)), s[:, 2]]) for s in strokes] for strokes, _, _ in held]
    rob[f"position_noise_{sg}px"] = metrics_ok(*score(noisy))
for st in (2.0, 5.0):
    jit = []
    for strokes, _, _ in held:
        jit.append([np.column_stack([s[:, 0], s[:, 1], np.maximum.accumulate(s[:, 2] + rng.normal(0, st, len(s)))]) for s in strokes])
    rob[f"timestamp_jitter_{int(st)}ms"] = metrics_ok(*score(jit))
for frac in (0.5, 0.75):
    cut = []
    for strokes, _, _ in held:
        allp = sum(len(s) for s in strokes); keep = int(allp * frac); out = []
        for s in strokes:
            if keep <= 0: break
            out.append(s[:keep]); keep -= len(s)
        cut.append(out)
    rob[f"truncated_{int(frac * 100)}pct"] = metrics_ok(*score(cut))
res["h6_robustness"] = rob
for k, v in rob.items(): log("H6", k, v)

# ---- scale invariance: same drawing at 2x coordinates on a 2x canvas
f1 = feats([s for s, _, _ in held[:100]])
f2 = feats([[s * np.array([2.0, 2.0, 1.0]) for s in strokes] for strokes, _, _ in held[:100]], (CANVAS[0] * 2, CANVAS[1] * 2))
rel = np.abs(f1 - f2) / np.maximum(np.abs(f1), 1e-9)
res["h6_scale_invariance_max_relative_diff"] = {n: float(rel[:, i].max()) for i, n in enumerate(FEATURE_NAMES)}
# ---- descriptive statistics of the simulated drawings
npts = np.array([sum(len(s) for s in strokes) for strokes, _, _ in held])
dur = Xh[:, FEATURE_NAMES.index("duration_s")]
res["drawing_stats"] = {"points_median": float(np.median(npts)), "points_p5_p95": [float(np.percentile(npts, 5)), float(np.percentile(npts, 95))],
                        "duration_s_median": float(np.median(dur)), "strokes_median": float(np.median([len(s) for s, _, _ in held]))}
json.dump(res, open(B / "results/hw.json", "w"), indent=1); log("ALL DONE")
