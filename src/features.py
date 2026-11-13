"""Reusable feature code for the California Housing pipeline.

Everything here is importable (no lambdas), so a fitted Pipeline that uses it
can be saved with joblib and loaded again from another script.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns the model expects (same as data/processed/california_housing_clean.csv, minus the target)
CLEAN_COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# Fixed domain rules from 01_california_housing_data_cleaning: rows beyond them are not normal homes
# (resorts, prisons, dormitories) and were removed before training.
MAX_ROOMS, MAX_OCCUP = 15, 6

# Big cities as (latitude, longitude). These are constants, not learned from the data.
CITIES = {"la": (34.05, -118.24), "sf": (37.77, -122.42), "san_diego": (32.72, -117.16), "sacramento": (38.58, -121.49)}

NUMERIC = CLEAN_COLUMNS + [
    "bedrooms_ratio", "rooms_per_person", "log_population",
    "dist_la", "dist_sf", "dist_san_diego", "dist_sacramento", "dist_nearest_city",
]


class HousingFeatures(BaseEstimator, TransformerMixin):
    """Add engineered features. Stateless: it learns nothing from the data.

    bedrooms_ratio    = AveBedrms / AveRooms (share of rooms that are bedrooms)
    rooms_per_person  = AveRooms / AveOccup (space per person)
    log_population    = log(1 + Population) (tames the long right tail)
    dist_<city>       = straight-line distance in degrees from (Latitude, Longitude) to a big city
    dist_nearest_city = the smallest of the four distances
    """

    NEW_COLUMNS = ["bedrooms_ratio", "rooms_per_person", "log_population"] + \
                  [f"dist_{name}" for name in CITIES] + ["dist_nearest_city"]

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        X["bedrooms_ratio"] = X["AveBedrms"] / X["AveRooms"]
        X["rooms_per_person"] = X["AveRooms"] / X["AveOccup"]
        X["log_population"] = np.log1p(X["Population"])
        for name, (lat, lon) in CITIES.items():
            X[f"dist_{name}"] = np.sqrt((X["Latitude"] - lat) ** 2 + (X["Longitude"] - lon) ** 2)
        X["dist_nearest_city"] = X[[f"dist_{name}" for name in CITIES]].min(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else self.feature_names_in_
        return base + self.NEW_COLUMNS


def numeric_columns(X):
    """Column selector for ColumnTransformer. It picks only the columns that exist,
    so the same preprocessor works with or without the HousingFeatures step."""
    return [c for c in NUMERIC if c in X.columns]


def apply_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the non-learning rules from 01_california_housing_data_cleaning to new data.

    - keep exactly the model's input columns, in the training order
    - convert them to float64
    Scaling and feature engineering are NOT done here: the fitted Pipeline does them.
    """
    missing = [c for c in CLEAN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    return df[CLEAN_COLUMNS].astype("float64")


def outside_training_range(df: pd.DataFrame) -> pd.Series:
    """True for rows the model was never trained on (AveRooms > 15 or AveOccup > 6).

    The model is also only valid for block groups with a median house value below $500,000,
    because the capped rows were removed in notebook 01.
    """
    return (df["AveRooms"] > MAX_ROOMS) | (df["AveOccup"] > MAX_OCCUP)
