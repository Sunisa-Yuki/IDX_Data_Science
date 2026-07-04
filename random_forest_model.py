"""
Random Forest model for predicting CRMLS ClosePrice
Scope: PropertyType = "Residential", PropertySubType = "SingleFamilyResidence"
Owner: Yuki (DS64)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# TODO: point this at your verified CRMLSSold CSV(s) once combined
DATA_PATH = "data/CRMLSSold_combined.csv"

df = pd.read_csv(DATA_PATH)

df = df[
    (df["PropertyType"] == "Residential")
    & (df["PropertySubType"] == "SingleFamilyResidence")
].copy()

print(f"Rows after filter check: {len(df)}")

FEATURES = [
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LivingArea",
    "LotSizeSquareFeet",
    "YearBuilt",
    "GarageSpaces",
]
TARGET = "ClosePrice"

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)
print("\nFeature importances:")
print(importances)

joblib.dump(rf, "models/random_forest_v1.joblib")
print("\nModel saved to models/random_forest_v1.joblib")