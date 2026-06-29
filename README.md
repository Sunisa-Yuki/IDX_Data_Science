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
- **Coverage:** Minimum 6 months of CRMLSSold transactions
- **Metadata reference:** `Trestle Property MetaData.pdf`

---

## Project Approach

The project follows a staged pipeline, each phase building on the last:

1. **Data Exploration** — understand structure, distributions, and key price drivers
2. **Preprocessing** — handle missing values, encode categoricals, build the train/test split
3. **Baseline Model** — Linear Regression to establish a benchmark
4. **Model Comparison** — Decision Tree and Random Forest vs. baseline
5. **Feature Engineering** — derived features (e.g. property age, bed/bath ratio) and a
   geographic layer from California school district boundaries
6. **Advanced Models** — Gradient Boosting (XGBoost / LightGBM) with hyperparameter tuning
7. **Evaluation** — R², MAPE, and MdAPE across all models, with error analysis by price band

_Current status: [update as you progress]_

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

## Setup

```bash
# clone the repo
git clone <repo-url>
cd ds64-ca-price-prediction

# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# install dependencies
pip install -r requirements.txt
```

> Place the CRMLSSold data in the `data/` folder before running notebooks.

---

## Models

The project builds models in order of increasing complexity and compares them:

| Model              | Status   | Notes                          |
|--------------------|----------|--------------------------------|
| Linear Regression  | TODO     | Baseline                       |
| Decision Tree      | TODO     |                                |
| Random Forest      | TODO     |                                |
| Gradient Boosting  | TODO     | XGBoost / LightGBM             |

---

## Results

_TODO — fill in after evaluation._

| Model              | R²   | MAPE | MdAPE |
|--------------------|------|------|-------|
| Linear Regression  | —    | —    | —     |
| Decision Tree      | —    | —    | —     |
| Random Forest      | —    | —    | —     |
| Gradient Boosting  | —    | —    | —     |

Best model: _TODO_

---

## How to Run

_TODO — fill in once the pipeline is stable._

---

## License

The code in this repository is released under the MIT License.
The MLS dataset is proprietary and is not included.
