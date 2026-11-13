# House Price Prediction: One Regression Algorithm at a Time

មេរៀនជាជំហានៗដែលយក **regression algorithm** សំខាន់ៗម្តងមួយៗ **ពី raw data ដល់ការទាយ** ដោយប្រើ dataset ពិតតែមួយ (California Housing, ជំរឿនឆ្នាំ 1990), pandas, scikit-learn, XGBoost និង Jupyter notebooks។

**Model ចុងក្រោយ:** [`XGBRegressor` ក្នុង `Pipeline`](notebooks/09_pipelines.ipynb) ដែលមាន feature engineering (`HousingFeatures`)។ Test R² **0.843**, MAE **0.251** (កំហុសជាធម្មតា ≈ **$25,100**), RMSE **0.387** (test set បើកម្តងគត់)។
គម្រោងនេះមានរចនាសម្ព័ន្ធដូច Titanic/Machine Learning course (`D:\Machine_Learning`) ដែរ: cleaning ដាច់ដោយឡែក, algorithm notebooks ដែលអានតែ `data/processed/`, `src/` សម្រាប់ code ដែលប្រើឡើងវិញបាន។ ប្រើ **regression** ជំនួស classification (MAE/RMSE/R² និង residual plots ជំនួស accuracy និង confusion matrix)។

## Course summary

Notebook ទាំង 9 ត្រូវបានបញ្ចប់ និងដំណើរការពីលើចុះក្រោមដោយគ្មាន error។ Algorithm នីមួយៗត្រូវបានយកពី raw data ដល់ការទាយ ដោយ cleaning ស្ថិតក្នុង notebook ដាច់ដោយឡែក ហើយគ្រប់ជំហានដែលរៀនពី data ត្រូវបាន fit លើ training set តែប៉ុណ្ណោះ។

| Stage | Lessons | What was covered |
|---|---|---|
| Data cleaning | 1 | Leakage check, capped target (`MedHouseVal >= 5.0`), duplicates, outliers by fixed rules, sanity checks `assert` |
| Supervised models | 2, 3, 4, 5, 6, 7 | Linear/Ridge/Lasso Regression, KNN, Decision Tree, Random Forest, SVR, XGBoost |
| Evaluation and tuning | 8 | Cross-validation, region-aware `GroupKFold`, bias/variance, learning and validation curves, grid and random search |
| Engineering | 9 | Pipeline, ColumnTransformer, custom transformer (`HousingFeatures`), saving and serving a model, `src/predict.py` |

**មិនមាននៅក្នុងគម្រោងនេះ (បើធៀបនឹង Titanic course):** Naive Bayes (classification ប៉ុណ្ណោះ), K-Means និង PCA (unsupervised, Titanic ប្រើ penguins/breast cancer) និង final project (Adult Income)។ មើល "Differences from the Titanic course" ខាងក្រោម។

ការរកឃើញចំនួនបួនដែលកើតឡើងម្តងហើយម្តងទៀត:

1. លើ data ធំ (19,433 rows) ភាពខុសគ្នារវាង models **ពិតប្រាកដ** (R² 0.636 → 0.834)។ ត្រង់ Titanic (891 rows) ភាពខុសគ្នាតូចជាង noise នៃ split តែមួយ។ ប៉ុន្តែ split តែមួយនៅតែមាន noise (±0.015 R²)។
2. Scaling, feature engineering និង model ជា `Pipeline` តែមួយ ដែល fit លើ training folds តែប៉ុណ្ណោះ។
3. Test set ផ្តល់លេខស្មោះត្រង់ លុះត្រាតែប្រើម្តង បន្ទាប់ពីការសម្រេចចិត្តទាំងអស់ចប់។ CV RMSE (0.387) ស្របនឹង test RMSE (0.387)។
4. **ទីតាំងគឺជាអ្វីៗ:** `GroupKFold` តាមតំបន់ បង្ហាញថា R² របស់ XGBoost ធ្លាក់ពី 0.81 ទៅ 0.63 ពេលទាយតំបន់ដែលមិនធ្លាប់ឃើញ។ ពិន្ទុក្នុងគម្រោងនេះអនុវត្តចំពោះសង្កាត់ក្នុងតំបន់ដែល model ស្គាល់ មិនមែនតំបន់ថ្មីទាំងស្រុងទេ។

## Setup

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by the execution policy:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, ipykernel, xgboost)
pip install -r requirements.txt

# 3. (Optional) Register the Jupyter kernel for VS Code
python -m ipykernel install --user --name house-price-venv --display-name "Python (house-price-venv)"

# 4. Download the raw dataset into data/raw/ (existing files are skipped)
python src/download_data.py
```

ក្នុង VS Code បើក notebook ហើយជ្រើស kernel **`.venv`** (ឬ **Python (house-price-venv)**)។ Notebooks ត្រូវតែដំណើរការតាមលំដាប់ **01 → 09** ព្រោះ notebook 01 បង្កើតទិន្នន័យ clean ហើយ notebook 09 បង្កើត model។ ចាប់ផ្តើម Jupyter ពី folder `notebooks/` ឬបើកដោយ VS Code ព្រោះ notebooks ប្រើ path ដូចជា `../data/...`។

### Predict with the saved pipeline

ដំណើរការ notebook 09 ម្តងដើម្បីបង្កើត `models/house_price_pipeline.joblib` រួចដំណើរការ:

```powershell
python src/predict.py data/examples/new_neighbourhoods.csv
```

ឧទាហរណ៍ CSV ត្រូវការ columns: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude` (តម្លៃដើម, មិនទាន់ scale)។ Script ព្រមានពី rows ក្រៅជួរ training (`AveRooms > 15` ឬ `AveOccup > 6`)។

ដើម្បីដំណើរការ notebooks ទាំងអស់ឡើងវិញពី command line:

```powershell
foreach ($nb in Get-ChildItem notebooks\0*.ipynb) {
    jupyter nbconvert --to notebook --execute --inplace $nb.FullName
}
```

## Project structure

```
House Price Prediction/
├── data/
│   ├── raw/          # original downloaded CSV, never modified
│   ├── processed/    # clean CSV saved by the cleaning notebook
│   └── examples/     # small example input for src/predict.py
├── notebooks/        # one notebook per topic, numbered in order
├── models/           # saved pipeline (git-ignored, created by notebook 09)
├── src/
│   ├── download_data.py   # downloads the raw dataset (scikit-learn, with an OpenML fallback)
│   ├── features.py        # custom transformer HousingFeatures, column selector and cleaning-rule functions
│   └── predict.py         # command-line prediction with the saved pipeline
├── _old_version/     # the previous version of this project (notebooks, README, data, model), kept as a backup
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow rules

- **Cleaning notebook** (01) អាន `data/raw/california_housing_raw.csv` ហើយ save `data/processed/california_housing_clean.csv`។
  វាធ្វើតែជំហានដែល **មិនរៀនពី data**: ពិនិត្យ leakage, ដក rows ដែល label ខុសដោយ **ច្បាប់ថេរ** (cap `MedHouseVal >= 5.0`, `AveRooms > 15`, `AveOccup > 6`), ពិនិត្យ duplicates, missing values, dtypes និង outliers។
- **Algorithm notebooks** (02–09) អានតែពី `data/processed/`។ ជំហានដែល **រៀន** (scaling) ធ្វើ **បន្ទាប់ពី** train/test split ហើយ fit លើ training set តែប៉ុណ្ណោះ។ ជាមួយ trees មិនត្រូវការ scaling ទេ។
- **Feature engineering** មិនធ្វើក្នុង notebook 01 ទេ: វាស្ថិតក្នុង `src/features.py` (`HousingFeatures`, stateless) ហើយត្រូវបានវាស់ដោយ CV ក្នុង notebook 09។
- Dataset ត្រូវបាន clean ម្តង ហើយប្រើឡើងវិញដោយគ្រប់ algorithm។ Split ដូចគ្នា (`test_size=0.2`, `random_state=42`) ក្នុងគ្រប់ notebook ដើម្បីឲ្យការប្រៀបធៀបយុត្តិធម៌។

## Datasets

| Raw file | Clean file | Source | Used in |
|----------|------------|--------|---------|
| `data/raw/california_housing_raw.csv` (20,640 x 9) | `data/processed/california_housing_clean.csv` (19,433 x 9) | `sklearn.datasets.fetch_california_housing` (fallback: OpenML dataset 537) | 01–09 |

Row នីមួយៗគឺជា **block group** (សង្កាត់ប្រហែល 600–3,000 នាក់) មិនមែនផ្ទះមួយខ្នងទេ។ Target `MedHouseVal` = តម្លៃផ្ទះមេដ្យាន គិតជាឯកតា **$100,000**។ Cleaning ដក 992 rows ដែល target ជាប់ cap ($500,001) និង 215 rows ដែលមិនមែនជាផ្ទះធម្មតា។

## Progress

លទ្ធផលទាំងអស់គឺលើ **test set** (3,887 rows, `random_state=42`) ហើយ MAE/RMSE ជាឯកតា $100,000 (0.25 = $25,000)។ Baseline (ទាយ mean ជានិច្ច): MAE 0.790, RMSE 0.979, R² 0.000។

| # | Notebook | Topic / Algorithm | Test score | Status |
|---|----------|-------------------|------------|--------|
| 1 | `01_california_housing_data_cleaning.ipynb` | Data cleaning | – (20,640 → 19,433 rows) | ✅ Done |
| 2 | `02_linear_regression.ipynb` | Linear, Ridge, Lasso Regression | R² 0.636, MAE 0.439, RMSE 0.590 (Ridge ដូចគ្នា; Lasso α=1 ទាយ mean ជានិច្ច) | ✅ Done |
| 3 | `03_knn.ipynb` | K-Nearest Neighbors (k=15, scaled) | R² 0.701, MAE 0.380, RMSE 0.535 (k ជ្រើសដោយ test set; មិន scale: R² 0.104) | ✅ Done |
| 4 | `04_decision_tree.ipynb` | Decision Tree (max_depth=9) | R² 0.671, MAE 0.385, RMSE 0.561 (depth ជ្រើសដោយ test set; គ្មានដែនកំណត់: R² 0.546) | ✅ Done |
| 5 | `05_random_forest.ipynb` | Random Forest (300 trees, max_features=0.5, chosen by OOB) | R² 0.797, MAE 0.295, RMSE 0.441, OOB R² 0.792 | ✅ Done |
| 6 | `06_svm.ipynb` | SVR (rbf, C=3, gamma=0.3, tuned by 3-fold GridSearchCV) | R² 0.762, MAE 0.328, RMSE 0.477 (train ≈ 14 s; support vectors 74%) | ✅ Done |
| 7 | `07_xgboost.ipynb` | Gradient Boosting (XGBoost, tuned by RandomizedSearchCV) | R² 0.834, MAE 0.264, RMSE 0.399, CV RMSE 0.406 (default: R² 0.815) | ✅ Done |
| 8 | `08_cross_validation_tuning.ipynb` | Cross-Validation, GroupKFold, bias/variance, grid vs random (final pick: XGBoost tuned) | CV RMSE 0.402 ± 0.009, test RMSE 0.399 (touched once) | ✅ Done |
| 9 | `09_pipelines.ipynb` | Pipeline + ColumnTransformer + `HousingFeatures` + GridSearchCV (best: XGBoost, max_depth 7) | R² 0.843, MAE 0.251, RMSE 0.387, CV RMSE 0.387 (touched once) | ✅ Done |

Status legend: ⬜ Not started · 🔄 In progress · ✅ Done

### Differences from the Titanic course

| ចំណុច | Titanic course | គម្រោងនេះ | ហេតុផល |
|---|---|---|---|
| Datasets | 5 datasets (Titanic, MPG, Breast cancer, Penguins, Adult) | 1 dataset | គម្រោងមួយសម្រាប់ dataset តែមួយ |
| Notebooks | 17 (`01`–`16b`) | 9 (`01`–`09`) | មិនមាន Naive Bayes, K-Means, PCA, final project |
| Naive Bayes | Gaussian Naive Bayes | មិនមាន | មិនមាន regression equivalent |
| K-Means, PCA | Penguins, Breast cancer | មិនមាន | unsupervised; មិនជា "classification → regression" |
| Metrics | accuracy, F1, confusion matrix | MAE, RMSE, R², residual plots | regression |
| Preprocessing | imputer + one-hot + scaler | scaler ប៉ុណ្ណោះ | data គ្មាន NaN ឬ categorical |
| "Duplicates across train and test" | មាននៅ notebooks 2, 5, 7, 8, 10 | មិនមាន | គ្មាន duplicates (notebook 01) |
| "How this project was built" | មាន | មិនមាន | ជាការរៀបរាប់របស់អ្នកនិពន្ធអំពីដំណើរការ |
| Kernel name | `ml-venv` | `python3` (ឬ `.venv`) | កុំជាន់ kernel របស់ Titanic course |

## Key concepts learned

- **Raw vs. processed data**: raw ជាច្បាប់ដើមដែលមិនកែ; processed ជាច្បាប់ចម្លងដែល clean។ ដូច្នេះកំហុស cleaning អាចធ្វើឡើងវិញពីដើម។
- **Target ជាប់ cap = label ខុស**: 992 rows (4.8%) ត្រូវបានកត់ជា $500,001 ទោះតម្លៃពិតខ្ពស់ជាង។ ដក rows ទាំងនោះ ដោយច្បាប់ថេរ (មិនរៀនពី data)។ តម្លៃដែលត្រូវបង់: model ត្រឹមត្រូវតែក្រោម $500k។
- **ច្បាប់ថេរ ទល់ percentile**: "AveRooms > 15 ឬ AveOccup > 6" ពន្យល់បាន ហើយអនុវត្តលើ row តែមួយបាន; percentile រៀនពី data។
- **គ្មាន missing/duplicate ≠ គ្មានការងារ**: coordinates ស្ទួន (11,747 rows) ជាសង្កាត់ខុសគ្នា (បង្គត់ 0.01°) មិនមែន duplicates ពិតទេ។
- **Linear Regression**: បន្ទាត់ `y = w·x + b` ដែលបន្ថយ squared errors។ Slope = ការប្រែប្រួលនៃ y ពេល x កើន 1 ឯកតា (ចំណូល +$10,000 → តម្លៃ +$40,239)។
- **Regression metrics**: MAE និង RMSE ជាឯកតា target ($100,000); RMSE ធំជាង MAE ពេលមាន error ធំៗ; R² = ភាគរយនៃ variance ដែល model ពន្យល់។ ប្រៀបធៀបជាមួយ `DummyRegressor` ជានិច្ច។
- **Residual plot**: ល្អ = ពពកចៃដន្យជុំវិញ 0។ រាងចីវលោ (funnel): កំហុសកើនជាមួយតម្លៃ។ គែមទ្រេត: ដោយសារ target ត្រូវបានកាត់ត្រឹម [0.15, 5.0]។
- **Coefficient ≠ correlation**: `AveRooms` មាន correlation +0.15 ប៉ុន្តែ coefficient −0.15 (ពេល features ផ្សេងថេរ)។ Multicollinearity (`Latitude`/`Longitude` ≈ −0.93) ធ្វើឲ្យ coefficients មិនស្ថិតស្ថេរ។
- **Ridge vs. Lasso**: Ridge បង្រួម coefficients ទាំងអស់; Lasso អាចកំណត់ជាសូន្យ។ លើ data នេះគ្មានមួយណាឈ្នះ Linear ធម្មតាទេ ព្រោះ bias ទើបជាបញ្ហា មិនមែន variance (15,546 rows, 8 features)។
- **KNN**: ទាយមធ្យមនៃ k អ្នកជិតខាង; ទន្ទេញ training data។ **Scaling ចាំបាច់**: R² 0.104 (មិន scale) → 0.700 (scale)។ k=1 ទន្ទេញ (train R² 1.000, test 0.493)។ `weights="distance"` ធ្វើឲ្យ train score គ្មានន័យ។
- **Decision Tree (regression)**: split បន្ថយ MSE; leaf ទាយមធ្យម; ទាយបានតែតម្លៃមិនបន្ត (depth 3 → 8 តម្លៃ)។ Tree គ្មានដែនកំណត់ overfit (train 1.000, test 0.546); `max_depth` គ្រប់គ្រង bias/variance។ Tree ដែលរាក់ (greedy) មិនប្រើទីតាំងទេ។
- **Random Forest**: bootstrap + random features; ទាយមធ្យម។ Tree តែមួយមិនស្ថិតស្ថេរ (R² 0.60–0.65 លើ 5 bootstrap samples; មធ្យម 0.72)។ **OOB score** (0.792) ព្យាករ test R² (0.797) ដោយមិនប៉ះ test set។ Trees ច្រើនជាងមិន overfit។ `max_features` default សម្រាប់ regression គឺ 1.0 (tune ទៅ 0.5)។
- **Impurity vs. permutation importance**: impurity (training) លម្អៀងទៅ features មានតម្លៃច្រើនបែប; permutation (test) ជឿជាក់ជាង ប៉ុន្តែ features ទាក់ទងគ្នា (`Latitude`/`Longitude`) អាចបំប៉ោង (R² ធ្លាក់ >1 ក្នុង XGBoost)។ កុំជឿ importance វិធីតែមួយ។
- **SVR**: tube `±epsilon` ជុំវិញអនុគមន៍; `C` និង `gamma` tune ជាមួយគ្នា; scaling ចាំបាច់ (R² −0.029 → 0.718)។ ដែនកំណត់: ពេល train កើន ≈ 4× ពេល rows ទ្វេដង; 74% នៃ rows ក្លាយជា support vectors។
- **Gradient Boosting (XGBoost)**: trees តូចៗមួយបន្ទាប់ពីមួយ ដែលនីមួយៗទាយ residual។ `learning_rate` តូច ត្រូវការ trees ច្រើន។ Early stopping ប្រើ validation ដែលកាត់ពី training (មិនមែន test)។ `RandomizedSearchCV`: CV RMSE 0.432 → 0.406 (−6%)។
- **K-fold Cross-Validation**: ពិន្ទុជា mean ± std; `RepeatedKFold` កាត់បន្ថយ noise។ Regression ប្រើ `KFold(shuffle=True)` (គ្មាន `StratifiedKFold`)។
- **Single splits are noisy**: 20 splits ចៃដន្យ: Linear R² 0.621–0.652, XGBoost 0.795–0.824។ ជាមួយ 3,887 test rows noise តូចជាង Titanic ប្រហែល 3 ដង ប៉ុន្តែនៅមាន។
- **Spatial leakage**: ជាមួយ random `KFold` សង្កាត់ជិតខាងស្ថិតក្នុង train និង validation ទាំងពីរ។ `GroupKFold` តាមតំបន់ 1°×1°: XGBoost R² 0.807 → 0.626 ខណៈ Linear 0.634 → 0.603។ ពិន្ទុរបស់ម៉ូដែលក្នុងគម្រោងនេះមិនធានាសម្រាប់តំបន់ដែលមិនធ្លាប់ឃើញ។
- **Pipelines inside CV**: preprocessing ក្នុង `Pipeline` ត្រូវបាន fit ឡើងវិញក្នុងរាល់ fold។ leakage ពី scaling ខាងក្រៅ CV តូច (0.0006) ប៉ុន្តែកើនឡើងពេល preprocessing ប្រើ target។
- **Bias vs. variance**: Linear: train ≈ validation ≈ 0.635 រាបស្មើ → bias (data បន្ថែមមិនជួយ)។ XGBoost: គម្លាត 0.14 ហើយ validation នៅកើន → variance (data បន្ថែមជួយ)។
- **Grid vs. random search**: Random (10 បន្សំ) ស្ទើរតែល្អស្មើ Grid (48 បន្សំ) ក្នុងពេលតិចជាង ≈ 4 ដង (CV RMSE 0.4576 ទល់ 0.4568)។
- **Choose by CV, touch test once**: XGBoost tuned ឈ្នះ (CV RMSE 0.402 ± 0.009; គ្មាន model ណាស្ថិតក្នុង 1 std); test RMSE 0.399 ស្ថិតក្នុង 1 std។
- **Pipeline as one object**: feature engineering + scaler + model fit, tune, save និងប្រើជាមួយគ្នា។ **Custom transformer** ក្នុង `src/features.py` (មិនមែន lambda) អាច import, pickle និង test។
- **Measure feature engineering**: `HousingFeatures` (ចម្ងាយទៅទីក្រុងធំ ជាដើម) បន្ថយ CV RMSE 3.8% (XGBoost) និង 5.3% (Linear) ប៉ុន្តែ **មិនជួយ** ក្នុង `GroupKFold`។ អត្ថប្រយោជន៍មានតែក្នុងតំបន់ដែលស្គាល់។
- **Where the model is weakest**: សង្កាត់ថ្លៃ (> $300k): MAE 0.517 ធំជាងសង្កាត់ថោក 3 ដង ហើយទាយទាបជាង ≈ $38,000។
- **Saving models**: `joblib.dump` pipeline ទាំងមូល; `outside_training_range` ព្រមានពី input ក្រៅជួរ។

### Leakage checklist

1. ដក features ដែលជា target ក្នុងទម្រង់ផ្សេង ឬមិនអាចដឹងពេលទាយ។ (California: ពិនិត្យហើយ គ្មាន។)
2. ដក rows ដែល label ខុសដោយ **ច្បាប់ថេរ** មិនមែនដោយស្ថិតិដែលរៀនពី data។
3. Split មុន; fit scalers (និង imputers/encoders ពេលមាន) លើ training set តែប៉ុណ្ណោះ។
4. ដាក់ preprocessing ក្នុង `Pipeline` ដើម្បីឲ្យវា fit ឡើងវិញក្នុងរាល់ CV fold។
5. កុំជ្រើស model ឬ hyperparameters ដោយ test set; ប្រើ CV, OOB ឬ validation set។ (Notebooks 03 និង 04 បានជ្រើស `k` និង `max_depth` ដោយ test set សម្រាប់ការបង្ហាញ ហើយ notebook 08 ធ្វើវាឡើងវិញដោយ CV: `max_depth=9` ដូចគ្នា។)
6. Early stopping ប្រើ validation set ដែលកាត់ពី training ហើយមិនមែន test set។
7. បើ data មានទីតាំង ឬពេលវេលា ពិចារណា `GroupKFold`/`TimeSeriesSplit` ជំនួស `KFold` ចៃដន្យ។
8. វាយតម្លៃ model ចុងក្រោយលើ test set **ម្តងគត់**។
9. ពិន្ទុ CV របស់ model ដែលបាន tune ស្ថិតក្នុងភាពសុទិដ្ឋិនិយមតិចតួច។
