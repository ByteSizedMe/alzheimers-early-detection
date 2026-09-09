# Convenience wrappers. Every target is a plain command you can also run by hand.
PY ?= python
export PYTHONPATH := src

.PHONY: help setup data prepare train evaluate report samples test lint api dashboard docker clean all

help:
	@echo "setup      install dependencies"
	@echo "data       generate a synthetic cohort (no ADNI needed)"
	@echo "prepare    Phase 1: split, hot-deck impute, ADASYN balance"
	@echo "train      Phase 2: fine-tune VGG19"
	@echo "evaluate   Phase 3: metrics + Grad-CAM panels"
	@echo "report     build the project report PDF from the latest run"
	@echo "samples    copy labelled sample scans to your Desktop"
	@echo "test       run the test suite"
	@echo "api        serve the FastAPI inference service on :8000"
	@echo "dashboard  serve the Streamlit dashboard on :8501"
	@echo "all        data -> prepare -> train -> evaluate"

setup:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) scripts/make_synthetic_data.py --n-per-class 150

prepare:
	$(PY) -m alzheimers.data.prepare

train:
	$(PY) -m alzheimers.training.train

evaluate:
	$(PY) -m alzheimers.training.evaluate --split test --gradcam-samples 8

report:
	$(PY) scripts/make_report.py

samples:
	$(PY) scripts/export_samples.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

api:
	$(PY) -m uvicorn alzheimers.api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run src/alzheimers/dashboard/app.py

docker:
	docker compose up --build

clean:
	rm -rf artifacts/gradcam artifacts/*.json artifacts/*.png data/processed
	find . -name __pycache__ -type d -exec rm -rf {} +

all: data prepare train evaluate report
