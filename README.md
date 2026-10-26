# Next Day Wildfire Spread Prediction

Tabular AML (Phase 1) and spatial deep learning (Phase 2) for predicting next-day fire masks on the [Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread/data) dataset (Huot et al., 2022).

## Setup

Python 3.14.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs OpenMP: `brew install libomp`.

Tested on macOS 26.6 (Apple M3, 8 GB RAM) with Python 3.14.8: the full pipeline and both notebooks run with no extra steps beyond `libomp`. On this machine the steps took longer than the times below (data 166 s, features 54 s, train 23 min, baselines 98 s, ablation 5 min).

Download the Kaggle dataset and put the 19 `*.tfrecord` files in `data/`.

## Layout

```
data/              raw TFRecords + processed files (not in git)
src/
  data.py          TFRecords -> data/processed/{train,eval,test}.npz, load_split()
  features.py      pixel table + engineered features
  evaluate.py      shared metrics -> reports/results.csv
  train.py         XGBoost / LightGBM tuning + final test run
  baselines.py     persistence, logistic regression, RF, raw XGBoost
  ablation.py      feature-group ablation
notebooks/
  01_eda.ipynb         EDA figures
  02_evaluation.ipynb  curves, probability maps, SHAP, error analysis
  notes_dataset.md     dataset notes from the paper
docs/
  data_dictionary.md   channels, units, sources, value ranges
paper/             IEEE LaTeX report (main.tex, sections/, references.bib)
reports/
  figures/         plots
  checks/          confidence intervals, calibration, learning curves, reproducibility check
  preds/           saved test predictions (not in git)
```

## How to run

Approximate runtimes on a laptop CPU:

```bash
python -m src.data        # ~11 s
python -m src.features    # ~30 s
python -m src.train       # ~19 min
python -m src.baselines   # ~80 s
python -m src.ablation    # ~3.5 min
```

Then the notebooks:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_evaluation.ipynb
```

## Results

Test split (1,689 patches, 6.73M labelled pixels, 1.25% fire). Precision / recall / F1 at the decision threshold chosen on the eval split. Bold = best per column.

| Model | PR AUC | ROC AUC | Precision | Recall | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.106 | 0.633 | 0.357 | 0.273 | 0.309 | 0.273 | 0.273 |
| Logistic reg. (raw) | 0.146 | 0.798 | **0.378** | 0.266 | 0.312 | 0.281 | 0.319 |
| Random forest (raw) | 0.157 | 0.816 | 0.371 | 0.270 | 0.312 | 0.279 | 0.362 |
| XGBoost (raw) | 0.145 | 0.795 | 0.337 | 0.257 | 0.292 | 0.267 | 0.375 |
| XGBoost (tuned, all features) | **0.257** | **0.906** | 0.242 | **0.526** | **0.332** | **0.388** | **0.675** |
| LightGBM (tuned, all features) | 0.248 | 0.903 | 0.242 | 0.520 | 0.331 | 0.384 | 0.671 |

Huot et al. report PR AUC 0.115 (persistence), 0.198 (logistic regression), 0.225 (random forest) and 0.284 (CNN); their LR / RF also see a 3x3 neighbourhood of every channel, so these are reference points, not a like-for-like comparison. More tables: `reports/results_table.md`.

## Reproducibility

A fresh clone and a clean venv reproduce `reports/best_params.json` and `reports/ablation.csv` exactly; only the row order of `reports/results.csv` depends on which script runs first.

## Team

| Member | Role | Delivered |
|---|---|---|
| Harshil | Feature engineering, main models, repo | Repo structure, TFRecord reader (`src/data.py`), feature pipeline (`src/features.py`), shared evaluation (`src/evaluate.py`), XGBoost / LightGBM tuning (`src/train.py`), feature-group ablation (`src/ablation.py`), patch orientation fix, methodology section |
| Sushant | EDA, data quality, baselines, ablation | Dataset notes, EDA notebook and figures, data quality audit, raw-channel baselines (`src/baselines.py`), Dataset & EDA section with the ablation write-up, EDA slides |
| Priyanshu | Literature review, theory, LaTeX | Report LaTeX setup and bibliography, related work and theory sections, confidence intervals, calibration and learning curves (`reports/checks/`), related-work slides |
| Sachin | Evaluation, visualisation, interpretability | Evaluation notebook: PR / ROC curves, probability maps, gain importance and SHAP, downwind check, error analysis, event-level evaluation; results table, results section and slides |
| Piyush | Presentation, README, data dictionary | Data dictionary, README, presentation deck, introduction and conclusion, final proofread |
