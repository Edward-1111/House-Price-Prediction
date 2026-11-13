"""Predict the median house value of California neighbourhoods (block groups) from a CSV file.

Usage (from the project root):
    python src/predict.py data/examples/new_neighbourhoods.csv
    python src/predict.py data/examples/new_neighbourhoods.csv --model models/house_price_pipeline.joblib

The CSV needs the columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
The saved Pipeline engineers features and scales by itself, so pass the raw values.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import apply_cleaning_rules, outside_training_range  # also makes HousingFeatures importable for joblib

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "house_price_pipeline.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict median house value from a CSV of neighbourhoods.")
    parser.add_argument("csv_path", help="CSV file with one neighbourhood (block group) per row")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the saved pipeline (.joblib)")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    neighbourhoods = pd.read_csv(args.csv_path)
    X = apply_cleaning_rules(neighbourhoods)

    result = neighbourhoods.copy()
    result["predicted_value_usd"] = (pipeline.predict(X) * 100_000).round(-2).astype(int)   # target is in units of $100,000
    result["warning"] = outside_training_range(X).map({True: "outside training range", False: ""})
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
