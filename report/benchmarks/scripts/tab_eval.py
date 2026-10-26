"""Tabular-stage benchmarks: hot-deck imputation vs baselines, and ADASYN vs alternatives downstream."""
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import make_synthetic_data as msd
from alzheimers.data.imputation import HotDeckImputer
from alzheimers.data.balancing import adasyn, imbalance_ratio
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score

B = Path(__file__).resolve().parents[1]
CLASSES = msd.CLASSES
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def counts(n):
    largest = max(p["share"] for p in msd.STAGE_PROFILE.values())
    return {l: max(int(n * msd.STAGE_PROFILE[l]["share"] / largest), 8) for l in CLASSES}
res = {}

# ---------------- imputation benchmark: complete table, MCAR masking ------------------
saved = (dict(msd.BASE_MISSING), dict(msd.MISSING_STAGE_PENALTY))
for k in msd.BASE_MISSING: msd.BASE_MISSING[k] = 0.0
for k in msd.MISSING_STAGE_PENALTY: msd.MISSING_STAGE_PENALTY[k] = 0.0
full = msd.generate_clinical(B / "data/clin_complete.csv", counts(400), seed=7)
msd.BASE_MISSING.update(saved[0]); msd.MISSING_STAGE_PENALTY.update(saved[1])
assert full.drop(columns=["subject_id", "label"]).isna().sum().sum() == 0
FEATS = [c for c in full.columns if c not in ("subject_id", "label")]
CONT = [c for c in FEATS if c != "apoe4"]          # apoe4 is binary: RMSE is not meaningful
res["imputation_setup"] = {"rows": len(full), "features": FEATS, "scored_columns": CONT, "mask_rate": 0.20, "repeats": 30}

def run_imputers(seed):
    rng = np.random.default_rng(seed)
    tr, te = train_test_split(full, test_size=0.3, stratify=full.label, random_state=seed)
    Xtr, ytr = tr[FEATS], tr.label; Xte_true, yte = te[FEATS].copy(), te.label
    mask = rng.random(Xte_true.shape) < 0.20
    mask[:, [FEATS.index(c) for c in FEATS if c == "apoe4"]] &= True
    for r in np.where(mask.all(1))[0]: mask[r, rng.integers(len(FEATS))] = False   # keep >=1 observed
    Xte = Xte_true.mask(mask)
    out = {}
    out["column mean"] = Xte.fillna(Xtr.mean())
    cm = Xtr.groupby(ytr.values).mean()
    out["class mean (oracle label)"] = Xte.copy().apply(lambda col: col.fillna(pd.Series(cm.loc[yte.values, col.name].values, index=col.index)))
    out["KNN (k=5)"] = pd.DataFrame(KNNImputer(n_neighbors=5).fit(Xtr).transform(Xte), columns=FEATS, index=Xte.index)
    for name, kw in [("hot-deck k=1, class", dict(n_donors=1, stratify_by_class=True)),
                     ("hot-deck k=5, class", dict(n_donors=5, stratify_by_class=True)),
                     ("hot-deck k=5, global", dict(n_donors=5, stratify_by_class=False))]:
        imp = HotDeckImputer(random_state=seed, **kw).fit(Xtr, ytr)
        out[name] = imp.transform(Xte, yte if kw["stratify_by_class"] else None)
    rows = {}
    for name, filled in out.items():
        rmse, vr, ccd = [], [], []
        for c in CONT:
            j = FEATS.index(c); m = mask[:, j]
            if m.sum() < 5: continue
            truth = Xte_true[c].to_numpy()[m]; est = filled[c].to_numpy()[m]; sd = Xtr[c].std()
            rmse.append(np.sqrt(np.mean((est - truth) ** 2)) / sd)
            vr.append(est.var() / truth.var())
        # does the imputed AD-class mean of hippocampal volume stay near the truth?
        j = FEATS.index("hippo_vol"); m = mask[:, j] & (yte.values == "AD")
        bias = (filled["hippo_vol"].to_numpy()[m].mean() - Xte_true["hippo_vol"].to_numpy()[m].mean()) / Xtr["hippo_vol"].std() if m.sum() >= 3 else np.nan
        rows[name] = {"nrmse": float(np.mean(rmse)), "var_ratio": float(np.mean(vr)), "ad_hippo_bias_sd": float(bias)}
    return rows
acc = {}
for s in range(30):
    for k, v in run_imputers(s).items():
        for kk, vv in v.items(): acc.setdefault(k, {}).setdefault(kk, []).append(vv)
res["imputation"] = {k: {kk: {"mean": float(np.nanmean(vv)), "sd": float(np.nanstd(vv, ddof=1))} for kk, vv in v.items()} for k, v in acc.items()}
log("imputation done")
for k, v in res["imputation"].items(): log(f"  {k:28s} nRMSE {v['nrmse']['mean']:.3f}  var-ratio {v['var_ratio']['mean']:.3f}  AD-hippo bias {v['ad_hippo_bias_sd']['mean']:+.3f}")

# ---------------- ADASYN downstream benchmark -----------------------------------------------
clin = msd.generate_clinical(B / "data/clin_missing.csv", counts(400), seed=11)   # natural, stage-dependent missingness
res["adasyn_setup"] = {"rows": len(clin), "class_counts": clin.label.value_counts().to_dict(), "imbalance": float(imbalance_ratio(clin.label)),
                       "missing_cells_pct": float(clin[FEATS].isna().mean().mean() * 100), "repeats": 30, "holdout_fraction": 0.3}
def ros(X, y, rng):
    X, y = np.asarray(X), np.asarray(y); labs, cnt = np.unique(y, return_counts=True); top = cnt.max(); Xs, ys = [X], [y]
    for l, c in zip(labs, cnt):
        if c < top:
            idx = rng.choice(np.where(y == l)[0], top - c, replace=True); Xs.append(X[idx]); ys.append(y[idx])
    return np.concatenate(Xs), np.concatenate(ys)
def clf(cw=None): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw))
methods = ["no resampling", "class-weighted loss", "random oversampling", "ADASYN (beta=1)"]
store = {m: {"macro_f1": [], "balanced_accuracy": [], "recall_AD": [], "recall_LMCI": []} for m in methods}
for seed in range(30):
    rng = np.random.default_rng(seed)
    tr, ho = train_test_split(clin, test_size=0.3, stratify=clin.label, random_state=seed)
    imp = HotDeckImputer(n_donors=5, stratify_by_class=True, random_state=seed)
    Xtr = imp.fit_transform(tr[FEATS], tr.label); Xho = imp.transform(ho[FEATS]); ytr, yho = tr.label.to_numpy(), ho.label.to_numpy()
    fitted = {}
    fitted["no resampling"] = clf().fit(Xtr, ytr)
    fitted["class-weighted loss"] = clf("balanced").fit(Xtr, ytr)
    Xr, yr = ros(Xtr, ytr, rng); fitted["random oversampling"] = clf().fit(Xr, yr)
    Xa, ya = adasyn(Xtr.to_numpy(), ytr, n_neighbors=5, beta=1.0, random_state=seed); fitted["ADASYN (beta=1)"] = clf().fit(Xa, ya)
    for m, model in fitted.items():
        yh = model.predict(Xho)
        store[m]["macro_f1"].append(f1_score(yho, yh, average="macro")); store[m]["balanced_accuracy"].append(balanced_accuracy_score(yho, yh))
        rc = recall_score(yho, yh, labels=list(CLASSES), average=None, zero_division=0)
        store[m]["recall_AD"].append(rc[CLASSES.index("AD")]); store[m]["recall_LMCI"].append(rc[CLASSES.index("LMCI")])
res["adasyn"] = {m: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in d.items()} for m, d in store.items()}
res["adasyn_paired_vs_none"] = {}
for m in methods[1:]:
    d = np.array(store[m]["macro_f1"]) - np.array(store["no resampling"]["macro_f1"])
    res["adasyn_paired_vs_none"][m] = {"mean_delta_macro_f1": float(d.mean()), "sd": float(d.std(ddof=1)), "wins": int((d > 0).sum()), "n": len(d)}
# the balancing effect itself, on one training split
tr, _ = train_test_split(clin, test_size=0.3, stratify=clin.label, random_state=0)
imp = HotDeckImputer(n_donors=5, random_state=0); Xtr = imp.fit_transform(tr[FEATS], tr.label)
Xa, ya = adasyn(Xtr.to_numpy(), tr.label.to_numpy(), n_neighbors=5, beta=1.0, random_state=0)
res["adasyn_balance"] = {"before": pd.Series(tr.label).value_counts().to_dict(), "after": pd.Series(ya).value_counts().to_dict(),
                         "imbalance_before": float(imbalance_ratio(tr.label)), "imbalance_after": float(imbalance_ratio(ya)), "synthetic_rows": int(len(ya) - len(tr))}
log("adasyn done")
for m, d in res["adasyn"].items(): log(f"  {m:22s} macroF1 {d['macro_f1']['mean']:.3f}±{d['macro_f1']['sd']:.3f}  AD recall {d['recall_AD']['mean']:.3f}  LMCI recall {d['recall_LMCI']['mean']:.3f}")
json.dump(res, open(B / "results/tab.json", "w"), indent=1)
log("ALL DONE")
