# House Price Prediction: Regression, Step by Step

An end-to-end machine learning project that predicts the **median house value of California neighbourhoods** from census data. It takes one dataset from raw CSV to a saved, reusable prediction pipeline, using seven regression approaches and honest evaluation (cross-validation, a test set opened once, and a location-aware check).

**Final model:** XGBoost inside a scikit-learn `Pipeline` with custom feature engineering. **Test R² 0.843, MAE 0.251 (about $25,100), RMSE 0.387.**

> 🇰🇭 The lesson notebooks are written in Khmer (technical terms and code in English). A Khmer version of this README is available here: [README_km.md](README_km.md).

## Dataset

- **California Housing** (1990 US census), 20,640 block groups, loaded with `sklearn.datasets.fetch_california_housing` (OpenML fallback included).
- **Row = one block group** (a neighbourhood of roughly 600–3,000 people), not a single house.
- **Target:** `MedHouseVal`, the median house value in units of $100,000 (2.5 = $250,000).
- **Features (8):** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
- **Cleaning (notebook 01):** removed 992 rows whose target is capped at $500,001 (a known label error) and 215 rows that are not normal homes (`AveRooms > 15` or `AveOccup > 6`), using fixed rules that learn nothing from the data. 19,433 rows remain, so the model is valid only for neighbourhoods below $500,000.

## Project structure

```
House Price Prediction/
├── data/
│   ├── raw/          # original CSV, never modified
│   ├── processed/    # clean CSV written by notebook 01
│   └── examples/     # example input for src/predict.py
├── notebooks/        # 9 numbered notebooks (run in order)
├── models/           # saved pipeline (git-ignored, created by notebook 09)
├── src/
│   ├── download_data.py   # downloads the raw dataset
│   ├── features.py        # HousingFeatures transformer + cleaning-rule helpers
│   └── predict.py         # command-line predictions with the saved pipeline
├── requirements.txt
├── README.md
└── README_km.md      # Khmer README
```

Workflow rules: cleaning happens once in notebook 01; every other notebook reads only `data/processed/`, uses the same split (`test_size=0.2`, `random_state=42`), and fits scalers on the training set only.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_california_housing_data_cleaning` | Leakage check, capped target, duplicates, outliers by fixed rules, sanity `assert`s, save and verify |
| 02 | `02_linear_regression` | Train/test split, baseline, MAE/RMSE/R², residual plots, coefficients, multicollinearity, Ridge and Lasso |
| 03 | `03_knn` | KNN regression, why scaling is mandatory, choosing `k`, bias/variance |
| 04 | `04_decision_tree` | Regression tree, overfitting with `max_depth`, readable tree, feature importance, decision paths |
| 05 | `05_random_forest` | Bagging, instability of one tree, OOB score, tuning by OOB, impurity vs permutation importance |
| 06 | `06_svm` | Support Vector Regression, `C`/`gamma`, scaling, `GridSearchCV`, training-time limits |
| 07 | `07_xgboost` | Gradient boosting, early stopping, `RandomizedSearchCV`, permutation importance |
| 08 | `08_cross_validation_tuning` | Repeated CV, `GroupKFold` by region, learning/validation curves, grid vs random search, final model choice |
| 09 | `09_pipelines` | `Pipeline` + `ColumnTransformer`, custom `HousingFeatures`, tuning features and model together, saving with `joblib`, CLI predictions |

## Results

All scores are on the same held-out test set (3,887 rows). MAE and RMSE are in units of $100,000. Lower MAE/RMSE and higher R² are better.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (always predict the mean) | 0.790 | 0.979 | 0.000 |
| Linear Regression (Ridge: identical) | 0.439 | 0.590 | 0.636 |
| Decision Tree (`max_depth=9`) | 0.385 | 0.561 | 0.671 |
| KNN (`k=15`, scaled) | 0.380 | 0.535 | 0.701 |
| SVR (`C=3`, `gamma=0.3`) | 0.328 | 0.477 | 0.762 |
| Random Forest (300 trees, tuned by OOB) | 0.295 | 0.441 | 0.797 |
| XGBoost (default) | 0.283 | 0.421 | 0.815 |
| XGBoost (tuned) | 0.264 | 0.399 | 0.834 |
| **XGBoost Pipeline + `HousingFeatures`** | **0.251** | **0.387** | **0.843** |

Cross-validation agrees with the test set: the tuned XGBoost scored a CV RMSE of 0.402 ± 0.009 (repeated 5-fold on the training set) against a test RMSE of 0.399, and the final pipeline scored 0.387 on both. Note that `k` (KNN) and `max_depth` (tree) were first picked using the test set for demonstration, which the notebooks flag; notebook 08 redoes the depth choice by cross-validation and gets the same answer.

## Key findings

- **Non-linear models win clearly.** Gradient boosting cuts MAE by about 43% versus Linear Regression (0.251 vs 0.439). On 19,000+ rows the gaps between model families are far larger than the noise of a split (R² varies by about ±0.015 across random splits).
- **Location effect (GroupKFold).** With random cross-validation, XGBoost reaches R² 0.81. When whole 1°×1° regions are held out (`GroupKFold`), it falls to **0.63**, while Linear Regression barely moves (0.63 → 0.60). A large part of the boosted trees' advantage comes from memorising nearby neighbourhoods, so the scores describe neighbourhoods in regions the model has seen, not brand-new regions.
- **Feature engineering helps only in known regions.** Distance-to-city features reduce CV RMSE by 3.8% (XGBoost) and 5.3% (Linear Regression) under random CV, but show no gain under `GroupKFold`.
- **Weak spot above $300k.** For expensive neighbourhoods (588 test rows) the MAE is 0.517 (about $51,700), roughly **3× the error of cheap ones**, and the model under-predicts them by about $38,000 on average. The capped rows were removed, and true values in this range are widely spread.
- **Scaling matters for distance-based models.** KNN test R² goes from 0.104 (unscaled) to 0.700 (scaled); SVR from −0.03 to 0.72.
- **Ridge and Lasso do not help here**, because with 15,546 training rows and 8 features the problem is bias, not variance.
- **Leakage discipline.** Test set opened once for the final model; scaling and features live inside the `Pipeline`; early stopping uses a validation split carved from training data.

## How to run

Requires Python 3.10+ (tested with 3.13). Commands are for Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py        # downloads data/raw/california_housing_raw.csv if missing
jupyter notebook                   # open notebooks/ and run 01 → 09 in order
```

On macOS/Linux activate with `source .venv/bin/activate`. Run the notebooks in order: 01 creates the cleaned data, and 09 creates `models/house_price_pipeline.joblib`. Total runtime is roughly 11 minutes (notebooks 06 and 08 are the slowest).

Predict from the command line with the saved pipeline:

```powershell
python src/predict.py data/examples/new_neighbourhoods.csv
```

The input needs the columns `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude` (raw values). Rows outside the training range (`AveRooms > 15` or `AveOccup > 6`) get a warning.

## Tech stack

Python · pandas · NumPy · scikit-learn (pipelines, model selection, inspection) · XGBoost · matplotlib · seaborn · joblib · Jupyter

## Limitations

- Valid only for neighbourhoods with a median value below $500,000 and for normal homes.
- Data is from the 1990 census, so it reflects 1990 prices; each row is a neighbourhood, not a single house.
- Performance on entirely unseen regions is much lower than the headline scores (see the GroupKFold finding).
