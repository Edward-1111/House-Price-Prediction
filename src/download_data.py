"""Download the public raw dataset used in this project into data/raw/.

The raw file is never modified. If it already exists it is skipped,
so re-running this script never overwrites the original data.

Run from the project root:
    python src/download_data.py
"""
import warnings
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

# Resolve paths relative to this file so the script works from any working directory
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_california_housing() -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, source). Tries scikit-learn first, then OpenML if the host blocks the download (HTTP 403)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hide the "retry downloading" messages
            frame = fetch_california_housing(as_frame=True).frame
        return frame, "sklearn.datasets.fetch_california_housing"
    except Exception:
        raw = fetch_openml(data_id=537, as_frame=True, parser="auto").frame
        frame = pd.DataFrame({
            "MedInc": raw["median_income"],
            "HouseAge": raw["housing_median_age"],
            "AveRooms": raw["total_rooms"] / raw["households"],
            "AveBedrms": raw["total_bedrooms"] / raw["households"],
            "Population": raw["population"],
            "AveOccup": raw["population"] / raw["households"],
            "Latitude": raw["latitude"],
            "Longitude": raw["longitude"],
            "MedHouseVal": raw["median_house_value"] / 100_000,  # same units as scikit-learn
        })
        return frame, "OpenML dataset 537 (same data, columns rebuilt)"


def download_california_housing(file_name: str = "california_housing_raw.csv") -> None:
    """Save California Housing (1990 census, 20,640 block groups) as CSV, unless it already exists."""
    out = RAW_DIR / file_name
    if out.exists():
        print(f"skip    {file_name} (already exists)")
        return
    df, source = load_california_housing()
    df.to_csv(out, index=False)
    print(f"saved   {file_name}: {df.shape[0]} rows x {df.shape[1]} cols (source: {source})")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_california_housing()
