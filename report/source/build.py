"""Turn measured JSON results into LaTeX macros and tables. No number in the paper is typed by hand."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; B = HERE.parent / "benchmarks"; REPO = HERE.parents[1]
R = B / "results"; P = HERE
mri = json.load(open(R / "mri.json")); tab = json.load(open(R / "tab.json"))
hw = json.load(open(R / "hw.json")); lat = json.load(open(R / "lat.json"))
meta = json.load(open(R / "env.json"))
CLASSES = ["NC", "EMCI", "LMCI", "AD"]

f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
def ci(c, d=3): return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"
def ms(x): return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")
macros = {}
def M(name, value): macros[name] = str(value)

# ---------------- environment / versions
M("PyVer", meta["python"]); M("TorchVer", lat["env"]["torch"]); M("SklVer", meta["sklearn"]); M("SciVer", meta["scipy"])
M("FastapiVer", meta["fastapi"]); M("UvicornVer", meta["uvicorn"])
M("CpuCount", lat["env"]["cpu_count"]); M("RamGb", f"{lat['env']['ram_gb']:.0f}"); M("TorchThreads", lat["env"]["torch_threads"])
M("StaticTotalKb", f"{lat['static_total_kb']:,.0f}"); M("StaticNoVideoKb", f"{lat['static_without_video_kb']:,.0f}")
M("ParamsTotalM", f"{lat['model']['params_total'] / 1e6:.1f}"); M("ParamsTrainM", f"{meta['params_trainable_expected'] / 1e6:.1f}")
ckm = mri["meta"]["checkpoint_metadata"]
M("CkEpochWord", {1: "the first"}.get(ckm["epoch"], f"epoch {ckm['epoch']} as the")); M("CkValScore", f"{ckm['val_score']:.3f}")
M("EValN", mri["e1_split_sizes"]["val"])

# ---------------- MRI headline
e1 = mri["e1_readme_split"]; pool = mri["e2_fresh_pooled"]
M("EOneN", e1["n"]); M("EOneAcc", f4(e1["accuracy"])); M("EOneFone", f4(e1["macro_f1"]))
M("PoolN", f"{pool['n']:,}"); M("PoolAcc", f3(pool["accuracy"])); M("PoolFone", f3(pool["macro_f1"])); M("PoolEce", f3(pool["ece"]))
M("PoolAccLo", f3(pool["ci95"]["accuracy"][0])); M("PoolAccHi", f3(pool["ci95"]["accuracy"][1]))
M("PoolFoneLo", f3(pool["ci95"]["macro_f1"][0])); M("PoolFoneHi", f3(pool["ci95"]["macro_f1"][1]))
M("PoolEarly", f3(pool["early_detection_recall"])); M("PoolKappa", f3(pool["quadratic_kappa"])); M("PoolBal", f3(pool["balanced_accuracy"]))
M("PoolNll", f3(pool["nll"])); M("PoolBrier", f3(pool["brier"])); M("PoolAuc", f3(pool["macro_auc"]))
for s in ("101", "102", "103"):
    d = mri["e2_fresh_per_seed"][s]; M(f"Acc{ {'101':'A','102':'B','103':'C'}[s] }", f3(d["accuracy"])); M(f"Fone{ {'101':'A','102':'B','103':'C'}[s] }", f3(d["macro_f1"]))
wrong = int(round((1 - pool["accuracy"]) * pool["n"])); M("PoolWrong", wrong)
cm = pool["confusion_matrix"]
# adjacent vs distant errors (distance in stage index)
adj = sum(cm[i][j] for i in range(4) for j in range(4) if abs(i - j) == 1); far = sum(cm[i][j] for i in range(4) for j in range(4) if abs(i - j) >= 2)
M("PoolAdjErr", adj); M("PoolFarErr", far)
M("EOneEce", f3(e1["ece"])); M("EOneEarly", f3(e1["early_detection_recall"]))
readme_cm = [[17, 1, 0, 0], [1, 11, 0, 0], [0, 0, 8, 0], [0, 0, 0, 5]]
M("EOneCmMatches", "identical to" if e1["confusion_matrix"] == readme_cm else "different from")

# ---------------- Grad-CAM audit
g = mri.get("e3_gradcam")
if g:
    M("GcN", g["n"]); M("GcMean", f3(g["central_mass_mean"])); M("GcMedian", f3(g["central_mass_median"])); M("GcMin", f3(g["central_mass_min"]))
    M("GcFlag", g["flagged_below_0.40"]); M("GcNWrong", g["n_wrong"])
    M("GcCorrect", f3(g["central_correct_mean"])); M("GcWrongMean", f3(g["central_wrong_mean"]) if g["central_wrong_mean"] is not None else "n/a")
    M("GcPfive", f3(g["central_mass_p05"]))

# ---------------- latency
M("PredictMedian", ms(lat["predict"]["median_ms"])); M("ExplainMedian", ms(lat["explain"]["median_ms"]))
M("PredictPNinetyFive", ms(lat["predict"]["p95_ms"])); M("ExplainPNinetyFive", ms(lat["explain"]["p95_ms"]))
M("PreprocMedian", ms(lat["preprocess"]["median_ms"])); M("ModelLoadS", f"{lat['model_load_s']:.1f}")
M("CkptMb", f"{lat['model']['checkpoint_mb']:.0f}"); M("ServerRssMb", f"{lat['server_rss_mb']:,.0f}"); M("ServerColdS", f"{lat['server_cold_start_s']:.1f}")
M("HwAnalyzeMedian", ms(lat["hw_analyze_total"]["median_ms"])); M("HwAnalyzePNinetyFive", ms(lat["hw_analyze_total"]["p95_ms"]))
M("HwFeatMedian", ms(lat["hw_extract_features"]["median_ms"])); M("HwTrainS", f"{hw['train_wall_seconds']:.1f}")
M("HwPoints", f"{lat['hw_median_points']:.0f}"); M("HwLoadMs", ms(lat["handwriting_model_load_s"] * 1000))
M("HttpPredictMedian", ms(lat["http"]["predict"]["median_ms"])); M("HttpExplainMedian", ms(lat["http"]["explain"]["median_ms"]))
M("HttpHwMedian", ms(lat["http"]["handwriting"]["median_ms"])); M("HttpHealthMedian", ms(lat["http"]["health"]["median_ms"]))
M("ThroughputBatchSixteen", f"{lat['batch_throughput']['16']['images_per_s']:.1f}"); M("ThroughputBatchOne", f"{lat['batch_throughput']['1']['images_per_s']:.1f}")
c = lat["http_concurrency"]
M("ConcPredOne", f"{c['predict_1']['req_per_s']:.1f}"); M("ConcPredFour", f"{c['predict_4']['req_per_s']:.1f}")
M("ConcExpOne", f"{c['explain_1']['req_per_s']:.2f}"); M("ConcExpFour", f"{c['explain_4']['req_per_s']:.2f}")
M("LocSrc", f"{lat['loc_nonblank']['python_src']:,}"); M("LocTests", f"{lat['loc_nonblank']['python_tests']:,}")
M("LocFront", f"{lat['loc_nonblank']['frontend_html_css_js']:,}"); M("LocScripts", f"{lat['loc_nonblank']['python_scripts']:,}")
M("TestSeconds", f"{meta['pytest_seconds']:.0f}"); M("TestCount", meta["pytest_count"])
_mx = max(lat["hw_scale_1p7_max_rel_diff"].values()); _e = int(f"{_mx:.1e}".split("e")[1]); _m = float(f"{_mx:.1e}".split("e")[0])
M("HwScaleMax", f"{_m:.1f}\\times10^{{{_e}}}" if _mx > 0 else "0")

# ---------------- handwriting
h = hw["h1_heldout"]
M("HwAuc", f3(h["auc"])); M("HwHeldN", hw["n_heldout"]); M("HwAcc", f3(h["accuracy"])); M("HwSens", f3(h["sensitivity"])); M("HwSpec", f3(h["specificity"]))
M("HwAucLo", f3(h["ci95"]["auc"][0])); M("HwAucHi", f3(h["ci95"]["auc"][1]))
M("HwAccLo", f3(h["ci95"]["accuracy"][0])); M("HwAccHi", f3(h["ci95"]["accuracy"][1]))
cv = hw["cv_shipped_recipe"]; M("HwCvAcc", f3(cv["cv_accuracy"])); M("HwCvAuc", f3(cv["cv_auc"])); M("HwTrainN", hw["n_train"])
M("HwGbAcc", f3(hw["h2_gradient_boosting"]["accuracy"])); M("HwGbAuc", f3(hw["h2_gradient_boosting"]["auc"]))
sf = hw["h2_single_feature_auc"]; best = max(sf, key=sf.get); M("HwBestSingle", f3(sf[best]))
M("HwRho", f3(hw["h5_severity_spearman"])); M("HwModShare", f"{100 * hw['h5_moderate_share_overall']:.0f}")
M("HwTypLow", f"{100 * hw['h5_tiers']['typical']['Low']:.0f}"); M("HwImpHigh", f"{100 * hw['h5_tiers']['impaired']['High']:.0f}")
M("HwTypHigh", f"{100 * hw['h5_tiers']['typical']['High']:.0f}"); M("HwImpLow", f"{100 * hw['h5_tiers']['impaired']['Low']:.0f}")
rob = hw["h6_robustness"]
M("HwNoiseFourAcc", f3(rob["position_noise_4.0px"]["accuracy"])); M("HwNoiseFourAuc", f3(rob["position_noise_4.0px"]["auc"]))
M("HwNoiseTwoAcc", f3(rob["position_noise_2.0px"]["accuracy"]))
drops = {n: hw["h1_heldout"]["auc"] - v["auc"] for n, v in hw["h3_drop_one"].items()}
M("HwMaxDrop", f"{max(drops.values()):.3f}"); M("HwMaxDropName", {"tremor_amp": "tremor"}.get(max(drops, key=drops.get), max(drops, key=drops.get)))

# ---------------- tabular
imp = tab["imputation"]
M("ImpRows", tab["imputation_setup"]["rows"]); M("AdasynRows", tab["adasyn_setup"]["rows"])
M("AdasynBefore", tab["adasyn_balance"]["imbalance_before"] and f"{tab['adasyn_balance']['imbalance_before']:.2f}")
M("AdasynSynth", tab["adasyn_balance"]["synthetic_rows"])
d = tab["adasyn_paired_vs_none"]["ADASYN (beta=1)"]
M("AdasynDelta", f"{d['mean_delta_macro_f1']:+.3f}".replace("-", "$-$")); M("AdasynDeltaSd", f"{d['sd']:.3f}"); M("AdasynWins", d["wins"])


# ---------------- extra macros
conf = json.load(open(R / "conf.json"))
M("ConfAcc", f3(conf["acc"])); M("ConfMean", f3(conf["mean_conf"])); M("ConfCorrect", f2(conf["conf_correct"])); M("ConfWrong", f2(conf["conf_wrong"])); M("ConfMedian", f3(conf["median_conf"]))
M("EOneKappa", f4(e1["quadratic_kappa"])); M("EOneAccLo", f3(e1["ci95"]["accuracy"][0])); M("EOneAccHi", f3(e1["ci95"]["accuracy"][1]))
M("EOneEarlyLo", f3(e1["ci95"]["early_detection_recall"][0])); M("EOneImpaired", int(sum(sum(r) for r in e1["confusion_matrix"][1:])))
M("EOneMissed", int(sum(e1["confusion_matrix"][i][0] for i in (1, 2, 3))))
rb = mri["e2b_robustness"]
for key, nm in [("clean", "Clean"), ("gauss_noise_0.05", "NoiseFive"), ("gauss_noise_0.10", "NoiseTen"), ("gauss_noise_0.20", "NoiseTwenty"), ("blur_sigma_2", "Blur"),
                ("jpeg_q30", "Jpeg"), ("brightness_+0.2", "Bright"), ("contrast_x0.6", "Contrast"), ("rotate_10deg", "Rotate"), ("shift_20px", "Shift")]:
    M(f"Rob{nm}F", f3(rb[key]["macro_f1"])); M(f"Rob{nm}E", f3(rb[key]["early_detection_recall"]))
M("PoolImpaired", int(sum(sum(r) for r in pool["confusion_matrix"][1:]))); M("PoolMissedImpaired", int(sum(pool["confusion_matrix"][i][0] for i in (1, 2, 3))))
M("PoolEmciLmci", pool["confusion_matrix"][1][2]); M("PoolEmciNc", pool["confusion_matrix"][1][0])
M("PoolEmciRecall", f3(pool["per_class"]["EMCI"]["recall"])); M("PoolLmciPrec", f3(pool["per_class"]["LMCI"]["precision"]))
if g:
    M("GcNWrongPct", f"{100 * g['n_wrong'] / g['n']:.1f}")


M("HttpPredOverhead", f"{lat['http']['predict']['median_ms'] - lat['predict']['median_ms']:.0f}")
M("HttpExpOverhead", f"{lat['http']['explain']['median_ms'] - lat['explain']['median_ms']:.0f}")
M("ExplainRatio", f"{lat['explain']['median_ms'] / lat['predict']['median_ms']:.1f}")
M("RssAfterLoad", f"{lat['model']['rss_after_load_mb']:.0f}"); M("RssAfterExplain", f"{lat['model']['rss_peak_after_explain_mb']:.0f}")
M("CkptBytes", f"{int(lat['model']['params_total'] * 4):,}".replace(",", "{,}")); M("CkptKb", f"{((REPO / "artifacts/best_model.pt").stat().st_size - lat['model']['params_total'] * 4) / 1e3:.0f}")
M("PngKb", f"{lat['http_payload_bytes']['mri_png'] / 1e3:.1f}"); M("HwJsonKb", f"{lat['http_payload_bytes']['handwriting_json'] / 1e3:.1f}")
M("MpFourKb", f"{lat['static_assets_kb']['bg.mp4']:.0f}"); M("WebmKb", f"{lat['static_assets_kb']['bg.webm']:.0f}")
M("PredPerS", f"{c['predict_1']['req_per_s']:.1f}"); M("ExpPerS", f"{c['explain_1']['req_per_s']:.1f}")

# ======================= tables =======================
def table(label, caption, colspec, header, rows, small=True, star=False):
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small: out.append("\\footnotesize")
    out.append(f"\\begin{{tabular}}{{{colspec}}}"); out.append("\\toprule"); out.append(header + " \\\\"); out.append("\\midrule")
    out += [r + " \\\\" for r in rows]; out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)

T = {}
# headline MRI
def mrow(name, key, fmt=f3, ci_key=None):
    a = fmt(e1[key]); b = fmt(pool[key])
    if ci_key: b += f" {ci(pool['ci95'][ci_key])}"
    return f"{name} & {a} & {b}"
rows = [mrow("Accuracy", "accuracy", ci_key="accuracy"), mrow("Balanced accuracy", "balanced_accuracy", ci_key="balanced_accuracy"),
        mrow("Macro F1", "macro_f1", ci_key="macro_f1"), mrow("Quadratic kappa", "quadratic_kappa", ci_key="quadratic_kappa"),
        mrow("Early-detection recall", "early_detection_recall", ci_key="early_detection_recall"),
        mrow("Macro AUC (one-vs-rest)", "macro_auc"), mrow("ECE (15 bins)", "ece"), mrow("NLL", "nll"), mrow("Brier score", "brier"),
        f"Slices & {e1['n']} & {pool['n']:,}"]
T["mri"] = (table("tab:mri", "Image Classifier Metrics (Measured, Released Checkpoint)", "@{}lcc@{}",
               "Metric & Reported split & Fresh cohorts, pooled [95\\% CI]", rows))
# confusion matrix + per-class (pooled)
rows = []
for i, cname in enumerate(CLASSES):
    pc = pool["per_class"][cname]
    rows.append(f"{cname} & " + " & ".join(str(v) for v in cm[i]) + f" & {pc['precision']:.3f} & {pc['recall']:.3f} & {pc['f1']:.3f}")
T["cm"] = (table("tab:cm", "Pooled Confusion Matrix on Fresh Cohorts (Measured; Rows Are True Stage)", "@{}lccccccc@{}",
               "True & NC & EMCI & LMCI & AD & Prec. & Recall & F1", rows))
# robustness
names = {"clean": "None (clean)", "gauss_noise_0.05": "Gaussian noise, $\\sigma=0.05$", "gauss_noise_0.10": "Gaussian noise, $\\sigma=0.10$",
         "gauss_noise_0.20": "Gaussian noise, $\\sigma=0.20$", "blur_sigma_2": "Gaussian blur, $\\sigma=2$ px", "jpeg_q30": "JPEG, quality 30",
         "brightness_+0.2": "Brightness $+0.2$", "contrast_x0.6": "Contrast $\\times0.6$", "rotate_10deg": "Rotation $10^\\circ$", "shift_20px": "Shift 20 px"}
rows = [f"{names[k]} & {v['macro_f1']:.3f} & {v['accuracy']:.3f} & {v['early_detection_recall']:.3f}" for k, v in mri["e2b_robustness"].items()]
T["robust"] = (table("tab:robust", "Image Robustness on the Seed-101 Cohort (Measured; 355 Slices)", "@{}lccc@{}", "Perturbation & Macro F1 & Accuracy & Early recall", rows))
# imputation
order = ["column mean", "class mean (oracle label)", "KNN (k=5)", "hot-deck k=1, class", "hot-deck k=5, class", "hot-deck k=5, global"]
lab = {"column mean": "Column mean", "class mean (oracle label)": "Class mean (uses true label)", "KNN (k=5)": "KNN imputer, $k=5$",
       "hot-deck k=1, class": "Hot-deck, $k=1$, class pool", "hot-deck k=5, class": "Hot-deck, $k=5$, class pool", "hot-deck k=5, global": "Hot-deck, $k=5$, global pool"}
rows = [f"{lab[k]} & {imp[k]['nrmse']['mean']:.3f} & {imp[k]['var_ratio']['mean']:.2f} & {imp[k]['ad_hippo_bias_sd']['mean']:+.2f}".replace('-', '$-$') for k in order]
T["impute"] = (table("tab:impute", "Imputation of Hidden Cells, 30 Repeats (Measured)", "@{}lccc@{}", "Method & nRMSE & Variance ratio & AD bias ($\\mathrm{sd}$)", rows))
# adasyn
lab2 = {"no resampling": "No resampling", "class-weighted loss": "Class-weighted loss", "random oversampling": "Random oversampling", "ADASYN (beta=1)": "ADASYN, $\\beta=1$"}
rows = [f"{lab2[k]} & {v['macro_f1']['mean']:.3f} $\\pm$ {v['macro_f1']['sd']:.3f} & {v['recall_AD']['mean']:.3f} & {v['recall_LMCI']['mean']:.3f}" for k, v in tab["adasyn"].items()]
T["adasyn"] = (table("tab:adasyn", "Downstream Effect of Balancing, 30 Splits (Measured; Mean $\\pm$ SD)", "@{}lccc@{}", "Strategy & Macro F1 & AD recall & LMCI recall", rows))
# handwriting headline
sfa = sf[best]
rows = [f"Majority class & {hw['h2_baselines']['majority_class_accuracy']:.3f} & -- & -- & 0.500",
        f"Best single feature & -- & -- & -- & {sfa:.3f}",
        f"Gradient boosting & {hw['h2_gradient_boosting']['accuracy']:.3f} & -- & -- & {hw['h2_gradient_boosting']['auc']:.3f}",
        f"Logistic reg., 5-fold CV & {cv['cv_accuracy']:.3f} & {cv['cv_sensitivity']:.3f} & {cv['cv_specificity']:.3f} & {cv['cv_auc']:.3f}",
        f"Logistic reg., held out & {h['accuracy']:.3f} & {h['sensitivity']:.3f} & {h['specificity']:.3f} & {h['auc']:.3f}"]
T["hw"] = (table("tab:hw", "Handwriting Classifier on Simulated Drawings (Measured)", "@{}p{3.55cm}cccc@{}", "Model & Acc. & Sens. & Spec. & AUC", rows))
# feature table
fl = {"duration_s": "Drawing time", "mean_speed": "Mean speed", "speed_cv": "Speed variability", "pause_ratio": "Stalled time", "in_air_ratio": "Pen-up time",
      "speed_peaks": "Speed peaks", "tremor_amp": "Tremor", "spiral_irregularity": "Spiral irregularity", "size_norm": "Drawing size"}
rows = [f"{fl[n]} & " + f"{hw['h4_coefficients'][n]:+.2f}".replace("-", "$-$") + f" & {sf[n]:.3f} & {hw['h3_drop_one'][n]['auc']:.3f}" for n in sorted(fl, key=lambda n: -abs(hw["h4_coefficients"][n]))]
T["ablate"] = (table("tab:ablate", "Handwriting Features: Weight, Single-Feature AUC, and AUC Without the Feature (Measured)", "@{}lccc@{}", "Feature & Weight & Alone & Dropped", rows))
# handwriting robustness
hn = {"clean": "None (clean)", "subsample_x2": "Every 2nd sample kept", "subsample_x4": "Every 4th sample kept", "position_noise_0.5px": "Position noise 0.5 px",
      "position_noise_1.0px": "Position noise 1 px", "position_noise_2.0px": "Position noise 2 px", "position_noise_4.0px": "Position noise 4 px",
      "timestamp_jitter_2ms": "Timestamp jitter 2 ms", "timestamp_jitter_5ms": "Timestamp jitter 5 ms", "truncated_75pct": "Drawing cut at 75\\%", "truncated_50pct": "Drawing cut at 50\\%"}
rows = [f"{hn[k]} & {rob[k]['accuracy']:.3f} & {rob[k]['auc']:.3f}" for k in hn]
T["hwrobust"] = (table("tab:hwrobust", "Handwriting Robustness on 600 Held-Out Drawings (Measured; Model Fixed)", "@{}lcc@{}", "Input degradation & Accuracy & AUC", rows))
# latency
def lrow(name, d): return f"{name} & {ms(d['median_ms'])} & {ms(d['p5_ms'])}--{ms(d['p95_ms'])} & {d['n']}"
rows = [lrow("Preprocessing (resize, normalize)", lat["preprocess"]), lrow("\\texttt{predict()}", lat["predict"]), lrow("\\texttt{explain()} (Grad-CAM)", lat["explain"]),
        lrow("HTTP \\texttt{/health}", lat["http"]["health"]), lrow("HTTP \\texttt{/predict}", lat["http"]["predict"]), lrow("HTTP \\texttt{/explain}", lat["http"]["explain"]),
        lrow("Handwriting: extract features", lat["hw_extract_features"]), lrow("Handwriting: features + score", lat["hw_analyze_total"]),
        lrow("HTTP \\texttt{/handwriting/analyze}", lat["http"]["handwriting"]), lrow("Simulate one drawing", lat["hw_simulate_one_drawing"])]
T["latency"] = (table("tab:latency", "Latency on \\CpuCount{} CPU Cores, Idle Machine (Measured; Milliseconds)", "@{}lccc@{}", "Operation & Median & 5th--95th pct. & $n$", rows))
# resources
b = lat["batch_throughput"]
rows = [f"Checkpoint file & {lat['model']['checkpoint_mb']:.0f} MB", f"Parameters, total & {lat['model']['params_total'] / 1e6:.1f} M",
        f"Model load time & {lat['model_load_s']:.1f} s", f"Server cold start to ready & {lat['server_cold_start_s']:.1f} s",
        f"Server process resident memory after the benchmark & {lat['server_rss_mb']:,.0f} MiB",
        f"Forward throughput, batch 1 & {b['1']['images_per_s']:.1f} img/s", f"Forward throughput, batch 16 & {b['16']['images_per_s']:.1f} img/s",
        f"\\texttt{{/predict}} throughput, 1 / 2 / 4 clients & {c['predict_1']['req_per_s']:.1f} / {c['predict_2']['req_per_s']:.1f} / {c['predict_4']['req_per_s']:.1f} req/s",
        f"\\texttt{{/explain}} throughput, 1 / 2 / 4 clients & {c['explain_1']['req_per_s']:.2f} / {c['explain_2']['req_per_s']:.2f} / {c['explain_4']['req_per_s']:.2f} req/s"]
T["resources"] = (table("tab:resources", "Size, Memory, and Throughput (Measured)", "@{}p{4.6cm}r@{}", "Quantity & Value", rows))
for k, v in T.items(): (P / f"tbl_{k}.tex").write_text(v)
(P / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(len(macros), "macros;", len(T), "tables")
