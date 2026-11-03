# Real Estate Market Analytics Dashboard

An interactive data analytics project that explores Indian housing listings through data cleaning, exploratory data analysis (EDA), and a Streamlit dashboard.

> **Scope:** This version intentionally focuses on data analytics. Machine learning classification, regression forecasting, and MLflow tracking are excluded.

## Project objectives

- Clean and validate a large real-estate dataset.
- Explore property prices, sizes, locations, and property characteristics.
- Build an interactive dashboard with filters and visual insights.
- Provide transparent, descriptive market analysis rather than unsupported investment predictions.

## Dataset

- Source file: `india_housing_prices.csv`
- Records: 250,000
- Original columns: 23
- Geographic coverage: 20 states and 42 cities in the supplied data.
- The locality values are coded (for example, `Locality_84`), not descriptive neighborhood names.

## Tools and technologies

Python, Pandas, NumPy, Plotly, Streamlit, Matplotlib, Seaborn, VS Code, Git, GitHub.

## Repository structure

```text
Real-Estate-Market-Analytics/
├── data/
│   ├── india_housing_prices_raw.csv
│   ├── cleaned_data.csv
│   └── data_quality_report.csv
├── notebooks/
│   └── eda.ipynb
├── outputs/
├── app.py
├── data_cleaning.py
├── eda.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Data cleaning methodology

1. Standardized whitespace and category formatting.
2. Converted numerical fields to numeric types.
3. Removed exact duplicate rows (none were present in the supplied file).
4. Recalculated `Price_per_SqFt` as `Price_in_Lakhs * 100000 / Size_in_SqFt`, expressed in INR per square foot.
5. Recalculated `Age_of_Property` using 2026 minus `Year_Built`.
6. Flagged records where `Floor_No` exceeds `Total_Floors`; these records are retained because the source data may contain generated/inconsistent floor attributes.
7. Flagged IQR-based outliers in price, size, and price per square foot without automatically deleting them.
8. Created `Amenities_Count` and `Price_per_BHK_Lakh` derived features.

### Important data quality observations

- The supplied source has no missing values and no exact duplicate rows.
- 116,304 records have floor-number inconsistencies (`Floor_No > Total_Floors` or invalid floor fields). They are flagged, not silently discarded.
- The original `Price_per_SqFt` is rounded and does not provide sufficient precision. The project recalculates it from price and area.
- The supplied locality labels are anonymized/coded. Interpret locality comparisons as dataset-level groupings, not verified real-world neighborhood rankings.
- The data is treated as a supplied dataset for analysis; results should not be interpreted as verified live market quotes.

## EDA questions covered

### Price and size
1. Distribution of property prices.
2. Distribution of property sizes.
3. Price per square foot by property type.
4. Relationship between property size and price.
5. Outliers in price per square foot and property size.

### Location
6. Average price per square foot by state.
7. Average property price by city.
8. Median property age by locality.
9. BHK distribution across cities.
10. Five localities with the highest average listed price.

### Feature relationships
11. Correlation among numerical features.
12. Nearby schools versus price per square foot.
13. Nearby hospitals versus price per square foot.
14. Price by furnishing status.
15. Price per square foot by facing direction.

### Market and investment-related characteristics
16. Properties by owner type.
17. Properties by availability status.
18. Parking availability versus property price.
19. Amenities count versus price per square foot.
20. Public transport accessibility versus price per square foot.

## Run locally in VS Code

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run data cleaning

```bash
python data_cleaning.py
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 5. Generate static EDA outputs (optional)

```bash
python eda.py
```

## GitHub upload

```bash
git init
git add .
git commit -m "Build real estate market analytics dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Real-Estate-Market-Analytics.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and create the repository on GitHub before running the remote commands.

## Limitations

- The dashboard is descriptive and does not predict future property prices.
- No classification model or automated investment recommendation is included.
- Correlation and group comparisons do not establish causation.
- Source locality labels are coded and may not represent actual locality names.
- Property data should be independently verified before real financial decisions.

## Project status

Analytics implementation: data cleaning, EDA and Streamlit dashboard. Run the app locally and validate all charts before submitting.
