from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Market Analytics", page_icon="🏡",
                   layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_data.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    # Backward-compatible calculation if a cleaned file is regenerated differently.
    if "Price_per_SqFt" not in data.columns:
        data["Price_per_SqFt"] = data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"]
    if "Age_of_Property" not in data.columns:
        data["Age_of_Property"] = 2026 - data["Year_Built"]
    if "Amenities_Count" not in data.columns:
        data["Amenities_Count"] = data["Amenities"].fillna("").apply(
            lambda x: len([v for v in str(x).split(",") if v.strip()])
        )
    return data

df = load_data()

st.title("🏡 Real Estate Market Analytics Dashboard")
st.caption("Explore property prices, locations, features and market characteristics across India.")
st.info("This dashboard provides descriptive analytics only. It does not generate machine-learning predictions or guarantee investment returns.")

# Sidebar filters
st.sidebar.header("🔎 Dashboard Filters")
states = sorted(df["State"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("State", states, default=[])
filtered = df[df["State"].isin(selected_states)] if selected_states else df.copy()

cities = sorted(filtered["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("City", cities, default=[])
if selected_cities:
    filtered = filtered[filtered["City"].isin(selected_cities)]

types = sorted(filtered["Property_Type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("Property type", types, default=[])
if selected_types:
    filtered = filtered[filtered["Property_Type"].isin(selected_types)]

bhks = sorted(filtered["BHK"].dropna().unique().tolist())
selected_bhks = st.sidebar.multiselect("BHK", bhks, default=[])
if selected_bhks:
    filtered = filtered[filtered["BHK"].isin(selected_bhks)]

min_price, max_price = float(filtered["Price_in_Lakhs"].min()), float(filtered["Price_in_Lakhs"].max())
price_range = st.sidebar.slider("Price range (₹ Lakhs)", min_value=float(np.floor(min_price)),
                                max_value=float(np.ceil(max_price)),
                                value=(float(np.floor(min_price)), float(np.ceil(max_price))))
filtered = filtered[filtered["Price_in_Lakhs"].between(price_range[0], price_range[1])]

availability_options = sorted(filtered["Availability_Status"].dropna().unique().tolist())
selected_availability = st.sidebar.multiselect("Availability", availability_options, default=[])
if selected_availability:
    filtered = filtered[filtered["Availability_Status"].isin(selected_availability)]

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} records")
if filtered.empty:
    st.warning("No properties match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Properties", f"{len(filtered):,}")
k2.metric("Average price", f"₹{filtered['Price_in_Lakhs'].mean():,.2f} L")
k3.metric("Average size", f"{filtered['Size_in_SqFt'].mean():,.0f} sq. ft.")
k4.metric("Avg. price / sq. ft.", f"₹{filtered['Price_per_SqFt'].mean():,.0f}")

tab_overview, tab_price, tab_location, tab_features, tab_invest = st.tabs([
    "Overview", "Price & Size", "Location Insights", "Property Features", "Market & Investment Insights"
])

with tab_overview:
    st.subheader("Market overview")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="Price_in_Lakhs", nbins=40,
                           title="1. Distribution of property prices",
                           labels={"Price_in_Lakhs":"Price (₹ Lakhs)"}, color_discrete_sequence=["#2563eb"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(filtered, x="Size_in_SqFt", nbins=40,
                           title="2. Distribution of property sizes",
                           labels={"Size_in_SqFt":"Size (sq. ft.)"}, color_discrete_sequence=["#0f766e"])
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        city_counts = filtered["City"].value_counts().head(15).reset_index()
        city_counts.columns = ["City", "Property_Count"]
        fig = px.bar(city_counts, x="City", y="Property_Count", title="Property listings by city (top 15)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        status = filtered["Availability_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig = px.pie(status, names="Status", values="Count", title="17. Properties by availability status", hole=.4)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

with tab_price:
    st.subheader("Price and size analysis")
    c1, c2 = st.columns(2)
    with c1:
        by_type = filtered.groupby("Property_Type", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
        fig = px.bar(by_type, x="Property_Type", y="Price_per_SqFt",
                     title="3. Average price per sq. ft. by property type",
                     labels={"Price_per_SqFt":"₹ per sq. ft."}, color="Property_Type")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sample = filtered.sample(min(7000, len(filtered)), random_state=42)
        fig = px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type",
                         hover_data=["City","BHK"], title="4. Property size vs property price",
                         labels={"Size_in_SqFt":"Size (sq. ft.)","Price_in_Lakhs":"Price (₹ Lakhs)"})
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered, y="Price_per_SqFt", points=False, title="5a. Price per sq. ft. – outlier review",
                     labels={"Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(filtered, y="Size_in_SqFt", points=False, title="5b. Property size – outlier review")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Outliers are visualized and flagged in the cleaned dataset; extreme values are retained unless verified as data errors.")

with tab_location:
    st.subheader("Location-based analysis")
    c1, c2 = st.columns(2)
    with c1:
        state_price = filtered.groupby("State", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
        fig = px.bar(state_price, x="State", y="Price_per_SqFt", title="6. Average price per sq. ft. by state",
                     labels={"Price_per_SqFt":"₹ per sq. ft."})
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        city_price = filtered.groupby("City", as_index=False)["Price_in_Lakhs"].mean().sort_values("Price_in_Lakhs", ascending=False).head(20)
        fig = px.bar(city_price, x="City", y="Price_in_Lakhs", title="7. Average property price by city (top 20)",
                     labels={"Price_in_Lakhs":"Average price (₹ Lakhs)"})
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        locality_age = filtered.groupby("Locality", as_index=False)["Age_of_Property"].median().sort_values("Age_of_Property", ascending=False).head(20)
        fig = px.bar(locality_age, x="Locality", y="Age_of_Property", title="8. Median property age by locality (20 highest)",
                     labels={"Age_of_Property":"Median age (years)"})
        fig.update_xaxes(tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bhk_city = filtered.groupby(["City","BHK"]).size().reset_index(name="Count")
        top_cities = filtered["City"].value_counts().head(12).index
        bhk_city = bhk_city[bhk_city["City"].isin(top_cities)]
        fig = px.bar(bhk_city, x="City", y="Count", color="BHK", barmode="stack",
                     title="9. BHK distribution across top 12 cities")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    top_localities = filtered.groupby("Locality", as_index=False)["Price_in_Lakhs"].mean().nlargest(5, "Price_in_Lakhs")
    fig = px.bar(top_localities, x="Locality", y="Price_in_Lakhs",
                 title="10. Five most expensive localities by average listed price",
                 labels={"Price_in_Lakhs":"Average price (₹ Lakhs)"})
    st.plotly_chart(fig, use_container_width=True)

with tab_features:
    st.subheader("Feature relationships and correlation")
    numeric = filtered.select_dtypes(include=np.number).drop(
        columns=["ID","Price_per_SqFt_Original","Age_of_Property_Original"], errors="ignore"
    )
    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="11. Correlation between numerical features")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        schools = filtered.groupby("Nearby_Schools", as_index=False)["Price_per_SqFt"].mean()
        fig = px.line(schools, x="Nearby_Schools", y="Price_per_SqFt", markers=True,
                      title="12. Nearby schools vs average price per sq. ft.",
                      labels={"Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hospitals = filtered.groupby("Nearby_Hospitals", as_index=False)["Price_per_SqFt"].mean()
        fig = px.line(hospitals, x="Nearby_Hospitals", y="Price_per_SqFt", markers=True,
                      title="13. Nearby hospitals vs average price per sq. ft.",
                      labels={"Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        furnishing = filtered.groupby("Furnished_Status", as_index=False)["Price_in_Lakhs"].mean()
        fig = px.bar(furnishing, x="Furnished_Status", y="Price_in_Lakhs",
                     title="14. Average price by furnished status",
                     labels={"Price_in_Lakhs":"Average price (₹ Lakhs)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        facing = filtered.groupby("Facing", as_index=False)["Price_per_SqFt"].mean()
        fig = px.bar(facing, x="Facing", y="Price_per_SqFt",
                     title="15. Average price per sq. ft. by facing direction",
                     labels={"Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)

with tab_invest:
    st.subheader("Market and investment-related insights")
    c1, c2 = st.columns(2)
    with c1:
        owners = filtered["Owner_Type"].value_counts().rename_axis("Owner Type").reset_index(name="Count")
        fig = px.pie(owners, names="Owner Type", values="Count", title="16. Properties by owner type", hole=.35)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        parking = filtered.groupby("Parking_Space", as_index=False)["Price_in_Lakhs"].mean()
        fig = px.bar(parking, x="Parking_Space", y="Price_in_Lakhs",
                     title="18. Average property price by parking availability",
                     labels={"Price_in_Lakhs":"Average price (₹ Lakhs)","Parking_Space":"Parking"})
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        amenity = filtered.groupby("Amenities_Count", as_index=False)["Price_per_SqFt"].mean()
        fig = px.bar(amenity, x="Amenities_Count", y="Price_per_SqFt",
                     title="19. Number of amenities vs average price per sq. ft.",
                     labels={"Amenities_Count":"Number of listed amenities","Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        transport = filtered.groupby("Public_Transport_Accessibility", as_index=False)["Price_per_SqFt"].mean()
        order = ["Low","Medium","High"]
        transport["Public_Transport_Accessibility"] = pd.Categorical(
            transport["Public_Transport_Accessibility"], categories=order, ordered=True
        )
        transport = transport.sort_values("Public_Transport_Accessibility")
        fig = px.bar(transport, x="Public_Transport_Accessibility", y="Price_per_SqFt",
                     title="20. Public transport accessibility vs price per sq. ft.",
                     labels={"Public_Transport_Accessibility":"Accessibility","Price_per_SqFt":"₹ per sq. ft."})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data quality notes")
    floor_invalid = int((~filtered["Floor_Data_Valid"].fillna(False)).sum()) if "Floor_Data_Valid" in filtered else 0
    outliers = int(filtered["Price_Outlier_Flag"].sum()) if "Price_Outlier_Flag" in filtered else 0
    q1, q2, q3 = st.columns(3)
    q1.metric("Floor inconsistencies", f"{floor_invalid:,}")
    q2.metric("Price outlier flags", f"{outliers:,}")
    q3.metric("Rows retained", f"{len(filtered):,}")
    st.caption("These are descriptive relationships, not proof of causation or a personalized investment recommendation.")

# Download current filtered dataset
st.sidebar.download_button(
    "⬇️ Download filtered data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_real_estate_data.csv",
    mime="text/csv"
)
