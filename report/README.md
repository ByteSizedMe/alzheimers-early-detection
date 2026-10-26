# IEEE report and benchmarks

`Alzheimers_Early_Detection_IEEE_Report.pdf` is the report (IEEEtran, conference mode).

```
report/
  source/       LaTeX source. numbers.tex and tbl_*.tex are GENERATED from benchmarks/results by build.py
  benchmarks/
    scripts/    the four measurement scripts
    results/    the raw measurements the report was built from
```

## Rebuild the PDF from the stored measurements

```bash
cd report/source
python build.py                      # results/*.json  ->  numbers.tex, tbl_*.tex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs a LaTeX install with `IEEEtran` (e.g. `texlive-publishers`, `texlive-latex-extra`, `texlive-science`).
No number in the paper is typed by hand: prose uses macros from `numbers.tex`.

## Re-run the measurements

Run from the repository root with the project installed (`pip install -r requirements-dev.txt psutil requests`)
and the checkpoint fetched (`python scripts/fetch_model.py`).

```bash
# cohorts: the authors' (seed 42, 285 slices) and three fresh ones (355 slices each)
python scripts/make_synthetic_data.py --out report/benchmarks/data/seed42  --seed 42  --n-per-class 120
for s in 101 102 103; do
  python scripts/make_synthetic_data.py --out report/benchmarks/data/seed$s --seed $s --n-per-class 150
done

python report/benchmarks/scripts/tab_eval.py   # imputation + ADASYN        (~30 s)
python report/benchmarks/scripts/hw_eval.py    # handwriting                (~1 min)
python report/benchmarks/scripts/mri_eval.py   # MRI metrics, robustness, Grad-CAM audit (~15 min on 4 CPU cores)
python report/benchmarks/scripts/lat_eval.py   # latency / memory / throughput: run on an IDLE machine (~2 min)
```

`tab_eval.py` and `hw_eval.py` were re-run from this folder and reproduced every stored value exactly
(apart from wall-clock timings). The latency figures are specific to the machine they ran on
(4 vCPU Intel Xeon @ 2.1 GHz, 16 GB RAM, CPU only) and will differ elsewhere.
`results/env.json` and `results/conf.json` were produced by short ad-hoc commands (library versions and parameter
counts; a mean-confidence check on the seed-101 cohort), and `results/pytest.txt` is the timed test run.

## Scope

All data are synthetic. The figures describe the pipeline and the generators, not patients.
