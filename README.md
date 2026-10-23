# Next Day Wildfire Spread Prediction

Tabular AML (Phase 1) and spatial deep learning (Phase 2) for predicting next-day fire masks on the [Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread/data) dataset (Huot et al., 2022).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle dataset and put the `*.tfrecord` files in `data/`.

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
notebooks/         EDA
reports/
  figures/         plots
  preds/           saved test predictions (not in git)
```

## How to run

```bash
python -m src.data
python -m src.features
python -m src.train
python -m src.baselines
python -m src.ablation
```

## Results

TBD

## Team

| Member | Role |
|---|---|
| Harshil | Feature engineering, main models, repo |
| Sushant | EDA, data quality, baselines, ablation |
| Priyanshu | Literature review, theory, LaTeX |
| Sachin | Evaluation, visualisation, interpretability |
| Piyush | Presentation, README, data dictionary |
