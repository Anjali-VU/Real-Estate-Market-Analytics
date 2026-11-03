"""
Run:
    python eda.py
Outputs summary tables and a set of EDA charts under outputs/.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cleaned_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA)

# Use a manageable sample for high-density scatter plots.
sample = df.sample(min(12000, len(df)), random_state=42)

# Summary tables supporting the 20 EDA questions.
tables = {
    "01_price_summary": df["Price_in_Lakhs"].describe().to_frame(),
    "02_size_summary": df["Size_in_SqFt"].describe().to_frame(),
    "03_price_sqft_by_type": df.groupby("Property_Type")["Price_per_SqFt"].mean().sort_values(ascending=False).to_frame(),
    "06_price_sqft_by_state": df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).to_frame(),
    "07_avg_price_by_city": df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).to_frame(),
    "08_median_age_by_locality": df.groupby("Locality")["Age_of_Property"].median().to_frame(),
    "10_top_localities": df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).to_frame(),
    "11_numeric_correlation": df.select_dtypes("number").corr(),
    "12_schools_vs_price_sqft": df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().to_frame(),
    "13_hospitals_vs_price_sqft": df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().to_frame(),
    "14_furnishing_vs_price": df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().to_frame(),
    "15_facing_vs_price_sqft": df.groupby("Facing")["Price_per_SqFt"].mean().to_frame(),
    "16_owner_counts": df["Owner_Type"].value_counts().to_frame("Property_Count"),
    "17_availability_counts": df["Availability_Status"].value_counts().to_frame("Property_Count"),
    "18_parking_vs_price": df.groupby("Parking_Space")["Price_in_Lakhs"].mean().to_frame(),
    "19_amenities_vs_price_sqft": df.groupby("Amenities_Count")["Price_per_SqFt"].mean().to_frame(),
    "20_transport_vs_price_sqft": df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().to_frame(),
}
for name, table in tables.items():
    table.to_csv(OUT / f"{name}.csv")

sns.set_theme(style="whitegrid")
charts = [
    ("01_price_distribution", "Price_in_Lakhs", "Property price distribution (₹ Lakhs)", "Price (₹ Lakhs)"),
    ("02_size_distribution", "Size_in_SqFt", "Property size distribution", "Size (sq. ft.)"),
    ("05_price_sqft_boxplot", "Price_per_SqFt", "Price per sq. ft. outlier check", "Price (₹ / sq. ft.)"),
]
for filename, col, title, xlabel in charts:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax)
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.35, s=18, ax=ax)
ax.set(title="Property size vs price", xlabel="Size (sq. ft.)", ylabel="Price (₹ Lakhs)")
fig.tight_layout(); fig.savefig(OUT/"04_size_vs_price.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 9))
corr = df.select_dtypes("number").drop(columns=["ID"], errors="ignore").corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Numeric feature correlation")
fig.tight_layout(); fig.savefig(OUT/"11_correlation_heatmap.png", dpi=150); plt.close(fig)

print(f"EDA tables and charts saved in: {OUT}")
