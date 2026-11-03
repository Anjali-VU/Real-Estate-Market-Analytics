"""
Data cleaning and quality checks for the Real Estate Market Analytics project.
Run from the project root:
    python data_cleaning.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "india_housing_prices_raw.csv"
OUTPUT = DATA_DIR / "cleaned_data.csv"

def clean_data(input_path=INPUT, output_path=OUTPUT):
    data = pd.read_csv(input_path)
    original_rows = len(data)

    # Standardize whitespace and text values.
    for col in data.select_dtypes(include="object").columns:
        data[col] = (data[col].astype(str).str.strip()
                     .str.replace(r"\s+", " ", regex=True))

    for col in ["State", "City", "Locality", "Property_Type", "Furnished_Status",
                "Public_Transport_Accessibility", "Facing", "Owner_Type",
                "Availability_Status"]:
        if col in data.columns:
            data[col] = data[col].str.replace("_", " ", regex=False).str.title()

    for col in ["Parking_Space", "Security"]:
        if col in data.columns:
            data[col] = data[col].str.title()

    if "Availability_Status" in data.columns:
        data["Availability_Status"] = data["Availability_Status"].replace({
            "Ready To Move": "Ready to Move",
            "Under Construction": "Under Construction"
        })

    numeric_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                    "Year_Built", "Floor_No", "Total_Floors", "Age_of_Property",
                    "Nearby_Schools", "Nearby_Hospitals"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.drop_duplicates().copy()

    # Recalculate derived metrics with clear units.
    if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(data.columns):
        data["Price_per_SqFt_Original"] = data["Price_per_SqFt"]
        data["Price_per_SqFt"] = np.where(
            data["Size_in_SqFt"] > 0,
            data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"],
            np.nan
        )

    if {"Year_Built", "Age_of_Property"}.issubset(data.columns):
        data["Age_of_Property_Original"] = data["Age_of_Property"]
        data["Age_of_Property"] = 2026 - data["Year_Built"]

    if {"Floor_No", "Total_Floors"}.issubset(data.columns):
        data["Floor_Data_Valid"] = (
            data["Floor_No"].notna() & data["Total_Floors"].notna()
            & (data["Floor_No"] >= 0) & (data["Total_Floors"] >= 1)
            & (data["Floor_No"] <= data["Total_Floors"])
        )

    # Flag, do not automatically delete, statistically unusual records.
    for col, flag in [
        ("Price_in_Lakhs", "Price_Outlier_Flag"),
        ("Size_in_SqFt", "Size_Outlier_Flag"),
        ("Price_per_SqFt", "Price_per_SqFt_Outlier_Flag")
    ]:
        if col in data.columns:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            data[flag] = (data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)

    if "Amenities" in data.columns:
        data["Amenities_Count"] = data["Amenities"].fillna("").apply(
            lambda x: len([item for item in str(x).split(",") if item.strip()])
        )
    if {"Price_in_Lakhs", "BHK"}.issubset(data.columns):
        data["Price_per_BHK_Lakh"] = np.where(
            data["BHK"] > 0, data["Price_in_Lakhs"] / data["BHK"], np.nan
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Original rows: {original_rows:,}")
    print(f"Cleaned rows:  {len(data):,}")
    print(f"Saved cleaned dataset: {output_path}")
    print(f"Remaining missing values: {int(data.isna().sum().sum()):,}")
    if "Floor_Data_Valid" in data:
        print(f"Floor inconsistencies flagged: {(~data['Floor_Data_Valid']).sum():,}")
    return data

if __name__ == "__main__":
    clean_data()
