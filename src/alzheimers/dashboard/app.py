"""Streamlit dashboard for reviewing model output.

    streamlit run src/alzheimers/dashboard/app.py

Two inference modes, chosen in the sidebar:

* **API** — posts to the FastAPI service. This is the deployed path, and it
  exercises the same boundary a hospital integration would.
* **Local** — loads the checkpoint in-process. Useful while iterating on a
  model without keeping a second service alive.

Both call the same ``Predictor``, so the two paths cannot disagree.

The layout is deliberately ordered: verdict, then where the model looked, then
the numbers. A confidence score presented alone invites the reviewer to accept
it; putting the attribution map directly beside it invites them to check it.
All markup lives in ``styles.py``.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# Allow `streamlit run src/alzheimers/dashboard/app.py` from the repo root
# without requiring an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from alzheimers import CLASS_DESCRIPTIONS  # noqa: E402
from alzheimers.config import Config  # noqa: E402
from alzheimers.dashboard import styles  # noqa: E402

st.set_page_config(
    page_title="Alzheimer's Early Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model…")
def load_local_predictor(checkpoint: str):
    from alzheimers.inference import Predictor

    return Predictor(checkpoint=checkpoint)


def call_api(api_url: str, image_bytes: bytes, filename: str, target_class: str | None) -> dict:
    response = requests.post(
        f"{api_url.rstrip('/')}/explain",
        files={"file": (filename, image_bytes, "image/png")},
        params={"target_class": target_class} if target_class else None,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def check_api(api_url: str) -> tuple[bool, str]:
    try:
        health = requests.get(f"{api_url.rstrip('/')}/health", timeout=5).json()
    except requests.RequestException:
        return False, "API unreachable"
    if health.get("model_loaded"):
        return True, f"Model ready on {health.get('device', 'cpu')}"
    return False, "API up, no checkpoint loaded"


def find_sample_dir() -> Path | None:
    """Locate the exported sample scans, if the user generated them."""
    candidates = [
        Path.home() / "OneDrive" / "Desktop" / "Alzheimers-Sample-Scans",
        Path.home() / "Desktop" / "Alzheimers-Sample-Scans",
        REPO_ROOT / "data" / "samples",
    ]
    return next((path for path in candidates if path.is_dir()), None)


def encode_png(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def render_sidebar(cfg: Config) -> dict:
    with st.sidebar:
        st.markdown('<div class="side-title">Inference</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Mode", ["API service", "Local checkpoint"], index=0, label_visibility="collapsed"
        )

        api_url = str(cfg["dashboard.api_url"])
        checkpoint = str(cfg["api.checkpoint"])

        if mode == "API service":
            api_url = st.text_input("API URL", value=api_url)
            ok, message = check_api(api_url)
            st.markdown(styles.status_badge(ok, message), unsafe_allow_html=True)
        else:
            checkpoint = st.text_input("Checkpoint path", value=checkpoint)
            exists = Path(checkpoint).is_absolute() or (REPO_ROOT / checkpoint).exists()
            st.markdown(
                styles.status_badge(exists, "Checkpoint found" if exists else "Checkpoint missing"),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="side-title">Explanation</div>', unsafe_allow_html=True)
        classes = list(cfg["classes"])
        choice = st.selectbox(
            "Explain which stage?",
            ["Predicted stage"] + classes,
            help="Pick a stage to ask a counterfactual: what evidence would have "
                 "supported that reading instead?",
        )

        view = st.radio(
            "Attribution view",
            ["Overlay", "Heatmap only"],
            horizontal=True,
            help="Overlay blends the map onto the slice; heatmap shows it alone.",
        )

        st.markdown(
            '<div class="disclaimer"><b>Research prototype.</b> Not a medical device '
            "and not for clinical use. Predictions are uncalibrated — confidence is "
            "a model output, not a probability of disease.</div>",
            unsafe_allow_html=True,
        )

    return {
        "mode": mode,
        "api_url": api_url,
        "checkpoint": checkpoint,
        "target_class": None if choice == "Predicted stage" else choice,
        "view": view,
    }


def render_result(result: dict, image_bytes: bytes, settings: dict, classes) -> None:
    predicted = result["predicted_class"]

    st.markdown(
        styles.verdict(
            stage=predicted,
            description=CLASS_DESCRIPTIONS.get(predicted, ""),
            confidence=result["confidence"],
            impairment=result["impairment_probability"],
            explained=result.get("explained_class", predicted),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(styles.stage_track(classes, predicted), unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            styles.scan_panel(
                "Input slice",
                encode_png(image_bytes),
                "The uploaded scan, resized to 224&times;224 and normalised before inference.",
            ),
            unsafe_allow_html=True,
        )
    with right:
        heatmap_only = settings["view"] == "Heatmap only"
        key = "heatmap_png_base64" if heatmap_only else "overlay_png_base64"
        image_b64 = result.get(key) or result.get("overlay_png_base64")
        caption = (
            "Raw attribution map. Brighter regions contributed more to the score."
            if heatmap_only
            else "Warm regions raised the score for the explained stage."
        )
        if image_b64:
            st.markdown(
                styles.scan_panel("Grad-CAM attribution", image_b64, caption),
                unsafe_allow_html=True,
            )

    lower_left, lower_right = st.columns(2, gap="medium")
    with lower_left:
        st.markdown(
            styles.probability_rows(result["probabilities"], predicted), unsafe_allow_html=True
        )
    with lower_right:
        if result.get("focus"):
            st.markdown(styles.audit_panel(result["focus"]), unsafe_allow_html=True)

    with st.expander("Raw response"):
        st.json({k: v for k, v in result.items() if not k.endswith("_base64")})


def main() -> None:
    cfg = Config.load()
    st.markdown(styles.CSS, unsafe_allow_html=True)
    st.markdown(styles.hero(), unsafe_allow_html=True)

    settings = render_sidebar(cfg)

    uploaded = st.file_uploader(
        "Upload an MRI slice",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        sample_dir = find_sample_dir()
        hint = (
            f"Sample scans are on your Desktop in <code>{sample_dir.name}</code> — "
            "each filename states its true stage, so you can check the model against it."
            if sample_dir
            else "No sample scans yet? Run <code>python scripts/export_samples.py</code> "
                 "to put a labelled set on your Desktop."
        )
        st.markdown(styles.empty_state(hint), unsafe_allow_html=True)
        return

    image_bytes = uploaded.getvalue()

    try:
        with st.spinner("Running inference and computing attribution…"):
            if settings["mode"] == "API service":
                result = call_api(
                    settings["api_url"], image_bytes, uploaded.name, settings["target_class"]
                )
            else:
                predictor = load_local_predictor(settings["checkpoint"])
                class_index = (
                    predictor.classes.index(settings["target_class"])
                    if settings["target_class"]
                    else None
                )
                result = predictor.explain(Image.open(io.BytesIO(image_bytes)), class_index)
    except requests.RequestException as exc:
        st.error(f"API call failed: {exc}")
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_result(result, image_bytes, settings, list(cfg["classes"]))


if __name__ == "__main__":
    main()
