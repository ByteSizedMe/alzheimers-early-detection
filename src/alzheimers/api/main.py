"""FastAPI inference service.

    uvicorn alzheimers.api.main:app --reload --port 8000

Endpoints
---------
``GET  /health``   liveness plus whether a model is actually loaded
``GET  /classes``  the four stages and what they mean
``POST /predict``  multipart image upload -> class probabilities
``POST /explain``  same, plus a base64 Grad-CAM overlay

The model loads lazily on first use rather than at import, so ``/health`` still
answers (reporting ``model_loaded: false``) when no checkpoint exists yet.
That distinction matters in a container orchestrator: a service that crashes on
boot looks identical to one that is merely untrained.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path

from .. import CLASS_DESCRIPTIONS, __version__
from ..config import Config, REPO_ROOT
from ..inference import Predictor, load_image_bytes

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # a 224px slice is ~100 KB; 20 MB is generous

STATIC_DIR = Path(__file__).parent / "static"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_state: dict = {"predictor": None, "error": None}


def get_predictor() -> Predictor:
    """Load the model on first request; cache it (and any load failure)."""
    if _state["predictor"] is None:
        try:
            _state["predictor"] = Predictor()
            _state["error"] = None
        except Exception as exc:  # noqa: BLE001 - reported through /health
            _state["error"] = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"model unavailable: {exc}",
            ) from exc
    return _state["predictor"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the model at boot when a checkpoint exists, so the first real
    # request does not pay the load cost. A missing checkpoint is not fatal.
    try:
        _state["predictor"] = Predictor()
        print(f"[api] model ready: {_state['predictor'].checkpoint_path}")
    except Exception as exc:  # noqa: BLE001
        _state["error"] = str(exc)
        print(f"[api] starting without a model: {exc}")
    yield
    _state["predictor"] = None


app = FastAPI(
    title="Alzheimer's Early Detection API",
    description="VGG19 MRI stage classifier with Grad-CAM explanations",
    version=__version__,
    lifespan=lifespan,
)

# The dashboard runs on a different port, so the browser needs CORS. Tighten
# allow_origins to the dashboard host before this leaves a private network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    predictor = _state["predictor"]
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": predictor is not None,
        "checkpoint": str(predictor.checkpoint_path) if predictor else None,
        "device": str(predictor.device) if predictor else None,
        "error": _state["error"],
    }


@app.get("/classes")
def classes() -> dict:
    cfg = Config.load()
    return {
        "classes": [
            {"name": name, "description": CLASS_DESCRIPTIONS.get(name, name), "index": i}
            for i, name in enumerate(cfg["classes"])
        ],
        "note": "Stages are ordinal: NC < EMCI < LMCI < AD.",
    }


async def _read_image(file: UploadFile):
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(payload)} bytes, limit {MAX_UPLOAD_BYTES})",
        )
    try:
        return load_image_bytes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify one MRI slice."""
    image = await _read_image(file)
    result = get_predictor().predict(image)
    result["filename"] = file.filename
    return result


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    target_class: str | None = Query(
        None,
        description="Explain this class instead of the predicted one, e.g. 'AD'",
    ),
) -> dict:
    """Classify one MRI slice and return a Grad-CAM overlay."""
    image = await _read_image(file)
    predictor = get_predictor()

    class_index = None
    if target_class is not None:
        if target_class not in predictor.classes:
            raise HTTPException(
                status_code=400,
                detail=f"unknown class {target_class!r}; expected one of {list(predictor.classes)}",
            )
        class_index = predictor.classes.index(target_class)

    result = predictor.explain(image, class_index)
    result["filename"] = file.filename
    return result


# ------------------------------------------------------------------ frontend

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def find_samples_dir() -> Path | None:
    """Locate the exported sample scans.

    Deliberately the same search order as the Streamlit dashboard, so both
    surfaces offer whatever set the user actually generated rather than each
    keeping its own copy.
    """
    for candidate in (
        Path.home() / "OneDrive" / "Desktop" / "Alzheimers-Sample-Scans",
        Path.home() / "Desktop" / "Alzheimers-Sample-Scans",
        REPO_ROOT / "data" / "samples",
    ):
        if candidate.is_dir():
            return candidate
    return None


def list_samples() -> list[dict]:
    """Sample scans with the true stage parsed out of the filename."""
    directory = find_samples_dir()
    if directory is None:
        return []

    samples = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if stem.startswith("MISREAD_"):
            # MISREAD_true-NC_predicted-EMCI
            parts = stem.split("_")
            true_stage = parts[1].replace("true-", "") if len(parts) > 1 else "NC"
            samples.append(
                {"name": path.name, "true_stage": true_stage, "is_known_error": True}
            )
        else:
            samples.append(
                {"name": path.name, "true_stage": stem.split("_")[0], "is_known_error": False}
            )
    return samples


@app.get("/samples")
def samples() -> dict:
    """Sample scans the front end offers as one-click inputs."""
    directory = find_samples_dir()
    return {
        "location": directory.name if directory else None,
        "samples": list_samples(),
    }


@app.get("/samples/{name}")
def sample_image(name: str) -> FileResponse:
    """Serve one sample by name.

    Resolved against the listing rather than by joining the name onto a path —
    a user-supplied filename must never reach the filesystem directly.
    """
    directory = find_samples_dir()
    allowed = {entry["name"] for entry in list_samples()}
    if directory is None or name not in allowed:
        raise HTTPException(status_code=404, detail=f"no such sample: {name!r}")
    return FileResponse(directory / name)


@app.get("/api")
def api_info() -> dict:
    return {
        "service": "Alzheimer's Early Detection API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/classes", "/predict", "/explain", "/samples"],
        "disclaimer": (
            "Research prototype. Not a medical device and not for clinical use."
        ),
    }


@app.get("/")
def index() -> FileResponse:
    """The review interface. API metadata moved to /api."""
    return FileResponse(STATIC_DIR / "index.html")
