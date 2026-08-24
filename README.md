# California Property Close Price Prediction

Predicting the final close price of single-family residential properties in
California using historical CRMLS (California Regional Multiple Listing Service)
transaction data.

**Program:** IDX Exchange — Data Science Internship

---

## Project Overview

The goal is to build a machine learning model that predicts the `ClosePrice`
(final sales price) of a single residential property in California based on its
characteristics (living area, bedrooms, bathrooms, lot size, location, etc.).

`ClosePrice` is the target variable. The dataset is restricted to:
- `PropertyType = "Residential"`
- `PropertySubType = "SingleFamilyResidence"`

---

## Data

This project uses proprietary MLS transaction data licensed for internship use,
so the dataset itself is not included in this repository. The code is structured
to run against the `data/` directory once the data is placed there locally.

- **Source:** CRMLS via IDX Exchange (`CRMLSSold` files)
- **Coverage:** 21 monthly files, ~243K transactions after filtering
- **Metadata reference:** `Trestle Property MetaData.pdf`
- **Split:** temporal — most recent month (2026-06) held out as test, everything prior used for training


---


## Project Approach

The project follows a staged pipeline, each phase building on the last:

1. **Data Exploration** — understand structure, distributions, and key price drivers
2. **Preprocessing** — missing value handling, geocoding, outlier removal, one-hot encoding, temporal split
3. **Baseline Model** — Linear Regression to establish a benchmark
4. **Model Comparison** — Decision Tree and Random Forest vs. baseline
5. **Feature Engineering** — derived features (property age, bed/bath ratio)
6. **Geo Clustering** — KMeans on coordinates as a location feature
7. **Advanced Models** — XGBoost and LightGBM with hyperparameter tuning

_Current status: pipeline complete through notebook 08. Best model is a tuned LightGBM with school district features at R² 0.9431._


---

## Repository Structure


```
ds64-ca-price-prediction/
├── data/             # gitignored — data lives here locally, never committed
├── notebooks/        # weekly EDA and modeling notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_evaluation.ipynb
├── src/               # reusable Python modules
├── models/            # saved trained models (joblib/pickle)
├── app.py             # optional Streamlit prediction app
├── .gitignore
├── requirements.txt
└── README.md
```

---

---

## Setup

```bash
# clone the repo
git clone <repo-url>
cd IDX_Data_Science

# create environment
conda create -n dsc80 python=3.10
conda activate dsc80

# install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn geopy xgboost lightgbm jupyter
```

> Place the CRMLSSold data in the `data/` folder before running notebooks.

---

## Models

| Model              | Status   | Notes                                    |
|--------------------|----------|------------------------------------------|
| Linear Regression  | Done     | Baseline                                 |
| Decision Tree      | Done     | `max_depth=10`                           |
| Random Forest      | Done     | `n_estimators=100, max_depth=15`         |
| XGBoost            | Done     | `n_estimators=500, lr=0.1, max_depth=6`  |
| LightGBM           | Done     | Tuned via grid search — **best model**   |


---

## Results

All models trained on log-transformed `ClosePrice`, evaluated on the held-out
test month (2026-06, 12,851 properties).

| Model              | R²     | MAPE   | MdAPE  |
|--------------------|--------|--------|--------|
| Linear Regression  | 0.7888 | 0.2397 | 0.1692 |
| Decision Tree      | 0.8016 | 0.2216 | 0.1510 |
| Random Forest      | 0.8979 | 0.1525 | 0.1023 |
| XGBoost            | 0.9244 | 0.1347 | 0.0914 |
| LightGBM           | 0.9294 | 0.1301 | 0.0894 |
| LightGBM (tuned)   | 0.9400 | 0.1176 | 0.0776 |
| **+ School Districts** | **0.9431** | **0.1143** | **0.0757** |

**Best model:** LightGBM with `num_leaves=127, learning_rate=0.05, n_estimators=1000, min_child_samples=50`

Median absolute percentage error of 7.8% — down from 16.9% at the Linear Regression baseline.

### Error by price band (tuned LightGBM)

| Band      | n    | MAPE   | MdAPE  |
|-----------|------|--------|--------|
| <300K     | 301  | 0.3917 | 0.2056 |
| 300–600K  | 2593 | 0.1030 | 0.0643 |
| 600K–1M   | 4248 | 0.0925 | 0.0624 |
| 1–2M      | 3917 | 0.1155 | 0.0874 |
| 2–5M      | 1540 | 0.1411 | 0.1113 |
| 5M+       | 252  | 0.2519 | 0.2100 |

The `<300K` and `5M+` bands remain 3–4x worse than mid-range. These are data
problems rather than model problems — likely residual below-market transfers at
the bottom and sparse training data at the top.

---

## Key Preprocessing Decisions

- **Nominal-value sales removed** (`ClosePrice < $50,000`). These are non-arms-length
  transfers — family transfers, quitclaim deeds, trustee sales — recorded at token
  prices. Removing 70 such rows cut Random Forest MAPE by 35% with no change to
  MdAPE, the signature of removing extreme outliers rather than shifting the
  distribution.
- **Upper outlier ceiling** at $100M to drop data entry errors.
- **Missing coordinates geocoded** via Nominatim rather than dropped. 16 of 29 rows
  recovered; the remaining 13 had addresses too malformed to resolve and were dropped.
- **Log-transformed target** to compress the right-skewed price distribution without
  excluding legitimate high-value sales.
- **High-cardinality location columns dropped** (City, SubdivisionName, PostalCode,
  school names) to keep one-hot dimensionality manageable at ~4,000 features.

---

## What Didn't Work

Documented because negative results are still results:

- **Engineered features** (`BedBathRatio`, `PropertyAge`) — no measurable effect.
  Both are derivable from columns the model already has, so tree splits can
  already capture these relationships.
- **Geo clustering** (KMeans, k=20 on lat/long) — no measurable effect. Cluster
  sizes were highly uneven (largest ~39K rows, smallest 2), reflecting California's
  population density distribution.

Gradient boosting delivered far larger gains than any feature engineering attempt.

---

## What Did Work: School District Features

Spatially joining each property to its CA school district (2024-25 boundaries)
and attaching three district-level attributes — total enrollment, % socioeconomically
disadvantaged, % English learners — improved every metric.

Feature importance ranked these three columns 7th, 9th, and 14th out of 4,000 features.

This contrasts sharply with the failed KMeans geo clustering: arbitrary geometric
clusters carry no information beyond position, while school district boundaries
follow real community lines and come with demographics that correlate directly
with price.

District identity was added as numeric attributes rather than one-hot encoded
names — 937 districts would have added ~900 columns for no additional gain.

---

## How to Run

Notebooks are sequential — each reads the CSVs written by the previous stage.

```bash
conda activate dsc80


jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_baseline_model.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_geo_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_advanced_models.ipynb
```

Any change to `02_preprocessing.ipynb` requires re-running everything downstream,
since 03–07 all read from `data/train_final.csv` and `data/test_final.csv`.

---

## Next Steps

- Investigate remaining `<300K` error — likely more below-market transfers above the $50K floor
- Extend the LightGBM grid; `num_leaves` was still improving at the ceiling of 127
- School district spatial join (scoped but not yet implemented)
- Revisit dropped high-cardinality location columns with target or frequency encoding

---

## License

The code in this repository is released under the MIT License.
The MLS dataset is proprietary and is not included.