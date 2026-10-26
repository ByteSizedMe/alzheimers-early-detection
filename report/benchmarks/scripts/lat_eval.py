"""Latency, throughput, memory and size benchmarks. Run on an otherwise idle machine."""
import json, os, subprocess, sys, time, warnings, glob, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, psutil, requests, torch
from PIL import Image
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO / "src"))
B = Path(__file__).resolve().parents[1]
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def stats(v):
    v = np.array(v) * 1000
    return {"median_ms": float(np.median(v)), "mean_ms": float(v.mean()), "p5_ms": float(np.percentile(v, 5)), "p95_ms": float(np.percentile(v, 95)), "n": int(len(v))}
res = {}
proc = psutil.Process()
res["env"] = {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
              "python": sys.version.split()[0], "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}

# ---- model load, size, parameters
from alzheimers.inference import Predictor
from alzheimers.models.vgg19 import count_parameters
ck = REPO / "artifacts/best_model.pt"
rss0 = proc.memory_info().rss
t = time.perf_counter(); pred = Predictor(checkpoint=str(ck)); res["model_load_s"] = time.perf_counter() - t
trainable, total = count_parameters(pred.model)
res["model"] = {"checkpoint_mb": ck.stat().st_size / 1e6, "params_total": int(total), "params_trainable_as_loaded": int(trainable),
                "rss_after_load_mb": (proc.memory_info().rss - rss0) / 2**20}
imgs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(str(REPO / "data/samples/*.png")))]
res["n_sample_images"] = len(imgs)

# ---- preprocessing alone, predict, explain
pre = []
for i in range(100):
    t = time.perf_counter(); pred._to_tensor(imgs[i % len(imgs)]); pre.append(time.perf_counter() - t)
res["preprocess"] = stats(pre)
for i in range(5): pred.predict(imgs[i])
lat = []
for i in range(100):
    t = time.perf_counter(); pred.predict(imgs[i % len(imgs)]); lat.append(time.perf_counter() - t)
res["predict"] = stats(lat); log("predict", res["predict"])
for i in range(3): pred.explain(imgs[i])
lat = []
for i in range(50):
    t = time.perf_counter(); pred.explain(imgs[i % len(imgs)]); lat.append(time.perf_counter() - t)
res["explain"] = stats(lat); log("explain", res["explain"])
res["model"]["rss_peak_after_explain_mb"] = (proc.memory_info().rss - rss0) / 2**20

# ---- batched forward throughput
th = {}
for bs in (1, 4, 16):
    x = torch.stack([pred.transform(imgs[i % len(imgs)]) for i in range(bs)])
    with torch.no_grad():
        pred.model(x); t = time.perf_counter(); reps = 6 if bs > 1 else 20
        for _ in range(reps): pred.model(x)
        dt = (time.perf_counter() - t) / reps
    th[str(bs)] = {"batch_ms": dt * 1000, "images_per_s": bs / dt}
res["batch_throughput"] = th; log("throughput", th)

# ---- handwriting: per-drawing latency
from alzheimers.handwriting.features import extract_features
from alzheimers.handwriting.model import HandwritingAnalyzer
from alzheimers.handwriting.synthetic import generate_drawing, CANVAS, DEMO_SEVERITY
rng = np.random.default_rng(0)
t = time.perf_counter(); ana = HandwritingAnalyzer(); res["handwriting_model_load_s"] = time.perf_counter() - t
draws = [generate_drawing(float(rng.uniform(0, 1)), rng) for _ in range(200)]
gen = []
for _ in range(50):
    t = time.perf_counter(); generate_drawing(0.5, rng); gen.append(time.perf_counter() - t)
fe, an = [], []
for d in draws:
    t = time.perf_counter(); extract_features(d, CANVAS); fe.append(time.perf_counter() - t)
    t = time.perf_counter(); ana.analyze(d, CANVAS); an.append(time.perf_counter() - t)
res["hw_simulate_one_drawing"] = stats(gen); res["hw_extract_features"] = stats(fe); res["hw_analyze_total"] = stats(an)
res["hw_median_points"] = float(np.median([sum(len(s) for s in d) for d in draws]))
# scale invariance at a non-power-of-two factor (a x2 factor is exact in floating point and proves little)
f1 = [extract_features(d, CANVAS) for d in draws[:60]]
f2 = [extract_features([s * np.array([1.7, 1.7, 1.0]) for s in d], (CANVAS[0] * 1.7, CANVAS[1] * 1.7)) for d in draws[:60]]
res["hw_scale_1p7_max_rel_diff"] = {k: float(max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-9) for a, b in zip(f1, f2))) for k in f1[0]}
log("hw", res["hw_analyze_total"])

# ---- HTTP end to end against a real uvicorn process
env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
t_start = time.perf_counter()
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "alzheimers.api.main:app", "--port", "8010", "--log-level", "warning"], cwd=REPO, env=env)
base = "http://127.0.0.1:8010"
try:
    for _ in range(300):
        try:
            if requests.get(base + "/health", timeout=1).json().get("model_loaded"): break
        except Exception: pass
        time.sleep(0.5)
    res["server_cold_start_s"] = time.perf_counter() - t_start
    sp = psutil.Process(srv.pid)
    sample = sorted(glob.glob(str(REPO / "data/samples/*.png")))[0]; payload = open(sample, "rb").read()
    demo = requests.get(base + "/handwriting/demo/borderline").json()
    hw_body = {k: demo[k] for k in ("strokes", "canvas_width", "canvas_height")}
    res["http_payload_bytes"] = {"mri_png": len(payload), "handwriting_json": len(json.dumps(hw_body))}
    def call(kind):
        if kind == "predict": return requests.post(base + "/predict", files={"file": ("a.png", payload, "image/png")})
        if kind == "explain": return requests.post(base + "/explain", files={"file": ("a.png", payload, "image/png")})
        if kind == "handwriting": return requests.post(base + "/handwriting/analyze", json=hw_body)
        return requests.get(base + "/health")
    http = {}
    for kind, n in (("health", 100), ("predict", 50), ("explain", 30), ("handwriting", 50)):
        for _ in range(3): call(kind)
        lat = []
        for _ in range(n):
            t = time.perf_counter(); r = call(kind); lat.append(time.perf_counter() - t); assert r.status_code == 200, r.text[:200]
        http[kind] = stats(lat); log("http", kind, http[kind])
    conc = {}
    for kind, n in (("predict", 40), ("explain", 24)):
        for workers in (1, 2, 4):
            t = time.perf_counter()
            with ThreadPoolExecutor(workers) as ex: list(ex.map(lambda _: call(kind).status_code, range(n)))
            dt = time.perf_counter() - t; conc[f"{kind}_{workers}"] = {"requests": n, "seconds": dt, "req_per_s": n / dt}
        log("conc", kind, {k: round(v["req_per_s"], 2) for k, v in conc.items() if k.startswith(kind)})
    res["http"] = http; res["http_concurrency"] = conc
    res["server_rss_mb"] = sp.memory_info().rss / 2**20
finally:
    srv.terminate(); srv.wait(timeout=20)

# ---- static asset sizes served by the interface
st = REPO / "src/alzheimers/api/static"
res["static_assets_kb"] = {str(p.relative_to(st)): round(p.stat().st_size / 1024, 1) for p in sorted(st.rglob("*")) if p.is_file()}
res["static_total_kb"] = round(sum(p.stat().st_size for p in st.rglob("*") if p.is_file()) / 1024, 1)
res["static_without_video_kb"] = round(sum(p.stat().st_size for p in st.rglob("*") if p.is_file() and p.suffix not in (".mp4", ".webm", ".jpg")) / 1024, 1)

# ---- code size
def loc(paths):
    n = 0
    for p in paths:
        n += sum(1 for line in open(p, encoding="utf-8", errors="ignore") if line.strip())
    return n
res["loc_nonblank"] = {"python_src": loc((REPO / "src").rglob("*.py")), "python_tests": loc((REPO / "tests").rglob("*.py")), "python_scripts": loc((REPO / "scripts").rglob("*.py")),
                       "frontend_html_css_js": loc(p for p in st.rglob("*") if p.suffix in (".html", ".css", ".js"))}
json.dump(res, open(B / "results/lat.json", "w"), indent=1); log("ALL DONE")
