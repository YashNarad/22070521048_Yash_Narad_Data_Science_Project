# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
    
# ----------------------------
# Page config & simple styling
# ----------------------------
st.set_page_config(page_title="Energy Data Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
    font-size: 18px;
    padding: 18px;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 {
    color: white;
    font-weight: 600;
}
div.stButton > button:first-child {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data loader (accept CSV or Excel)
# ----------------------------
@st.cache_data
def load_data(path="cleaned_dataset.csv"):
    # Try CSV then Excel
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not read {path} as CSV or Excel. Put the cleaned dataset (cleaned_dataset.csv or .xls/.xlsx) next to app.py."
            ) from e

    # Ensure expected columns exist
    expected = {"region", "state", "is_union_territory", "month", "quarter",
                "energy_requirement_mu", "energy_availability_mu", "energy_deficit"}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")

    # Create derived columns
    df["gap"] = df["energy_requirement_mu"] - df["energy_availability_mu"]
    df["deficit_flag"] = (df["energy_deficit"] > 0).astype(int)

    # Ensure numeric types
    for col in ["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# load dataset (file name used by you earlier)
try:
    df = load_data("cleaned_dataset.csv")
except Exception:
    # fallback to xls
    df = load_data("cleaned_dataset.xls")

# ----------------------------
# Helper: robust geojson featureidkey picker
# ----------------------------
@st.cache_data
def load_geojson(url):
    raw = requests.get(url, timeout=10)
    geo = raw.json()
    # inspect first feature for property keys
    props = geo.get("features", [{}])[0].get("properties", {})
    # common possible name keys in various India geojson files
    candidates = ["ST_NM", "st_nm", "NAME_1", "state_name", "NAME", "STATE", "st_name"]
    for c in candidates:
        if c in props:
            return geo, f"properties.{c}"
    # fallback: try any property that looks like a name (string)
    for k, v in props.items():
        if isinstance(v, str) and len(v) > 1:
            return geo, f"properties.{k}"
    # final fallback (may fail later)
    return geo, "properties.ST_NM"

geo_url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
try:
    india_geo, featureidkey = load_geojson(geo_url)
except Exception:
    india_geo, featureidkey = None, None

# ----------------------------
# Main title / dashboard header
# ----------------------------
st.markdown("""
<h1 style='text-align: center; color: #2563eb;'>⚡ Energy Data Dashboard</h1>
<p style='text-align: center; font-size: 18px;'>Explore energy deficits, seasonality trends, and predictive models across India</p>
""", unsafe_allow_html=True)


# ----------------------------
# Sidebar navigation (Styled & Larger Font)
# ----------------------------
st.sidebar.markdown("""
<style>
/* Sidebar background and text */
[data-testid="stSidebar"] {
    background-color: #1e293b;
    color: white;
}

/* Sidebar header */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, label {
    color: white;
}

/* Radio buttons container */
[data-testid="stSidebar"] .stRadio {
    background-color: #334155;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 5px;
}

/* Individual radio buttons - larger font */
div.stRadio > div > label {
    font-size: 22px;  /* increased font size */
    font-weight: bold; /* make it bold */
    padding: 12px 16px;
    display: block;
    border-radius: 10px;
    margin-bottom: 5px;
    background-color: #334155;
    color: white;
    cursor: pointer;
    transition: 0.3s;
}

/* Hover effect for buttons */
div.stRadio > div > label:hover {
    background-color: #3b82f6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar title with emoji
st.sidebar.markdown("## ⚡ Navigation Panel")

# Styled radio buttons
page = st.sidebar.radio(
    "Go to:", 
    ["📘 Dataset Description", "📊 EDA", "🤖 ML Models", "🔮 Prediction"]
)

# ----------------------------
# Page 1: Dataset Description
# ----------------------------
if page == "📘 Dataset Description":
    st.title("📘 Dataset Description")
    st.markdown(
        "Data source: [India Data Portal](https://indiadataportal.com/p/power/r/mop-power_supply_position-st-mn-aaa)"
    )

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Dataset info & column descriptions")
    st.write(f"Rows: {df.shape[0]}   |   Columns: {df.shape[1]}")
    st.markdown("""
    **Columns**
    - **region** — geographical region (e.g., North, South). Useful for regional analysis.
    - **state** — state / union territory name. Used in maps and drill-downs.
    - **is_union_territory** — True/False flag.
    - **month** — month name/abbrev (seasonality).
    - **quarter** — financial quarter (Q1..Q4).
    - **energy_requirement_mu** — energy requirement in Million Units (MU) (numeric).
    - **energy_availability_mu** — available energy in MU (numeric).
    - **energy_deficit** — deficit in MU (numeric) — can be target for regression/analysis.
    - **gap** — derived: requirement - availability.
    - **deficit_flag** — derived: (energy_deficit > 0) as 0/1 — for classification tasks.
    """)

    st.subheader("Basic distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Energy Requirement (MU)**")
        st.write(df["energy_requirement_mu"].describe())
    with col2:
        st.markdown("**Energy Availability (MU)**")
        st.write(df["energy_availability_mu"].describe())

# ----------------------------
# Page 2: EDA (Seasonality & Multi-select)
# ----------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # ----------------------------
    # Multi-select filters
    # ----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_region = st.multiselect(
            "Region", 
            options=sorted(df["region"].dropna().unique().tolist()), 
            default=sorted(df["region"].dropna().unique().tolist())
        )
    with col2:
        sel_quarter = st.multiselect(
            "Quarter", 
            options=sorted(df["quarter"].dropna().unique().tolist()), 
            default=sorted(df["quarter"].dropna().unique().tolist())
        )
    with col3:
        sel_month = st.multiselect(
            "Month", 
            options=sorted(df["month"].dropna().unique().tolist()), 
            default=sorted(df["month"].dropna().unique().tolist())
        )

    # ----------------------------
    # Apply filters
    # ----------------------------
    filtered = df.copy()
    if sel_region:
        filtered = filtered[filtered["region"].isin(sel_region)]
    if sel_quarter:
        filtered = filtered[filtered["quarter"].isin(sel_quarter)]
    if sel_month:
        filtered = filtered[filtered["month"].isin(sel_month)]

    st.subheader("Filtered dataset (top rows)")
    st.dataframe(filtered.head())

    st.subheader("Key numeric statistics")
    st.write("Shape:", filtered.shape)
    st.dataframe(filtered[["energy_requirement_mu", "energy_availability_mu", "energy_deficit", "gap"]].describe())

    # ----------------------------
    # Time-series / month plot
    # ----------------------------
    st.subheader("Energy Requirement vs Availability (by month)")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    if filtered["month"].dtype == object and set(filtered["month"].unique()).issubset(set(month_order)):
        filtered["month"] = pd.Categorical(filtered["month"], categories=month_order, ordered=True)

    fig = px.bar(
        filtered, 
        x="month", 
        y=["energy_requirement_mu", "energy_availability_mu"],
        barmode="group", 
        labels={"value":"Energy (MU)", "month":"Month"},
        title="Monthly Requirement vs Availability (MU)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Energy deficit distribution
    # ----------------------------
    st.subheader("Energy Deficit distribution by region")
    fig2 = px.box(
        filtered, 
        x="region", 
        y="energy_deficit", 
        color="region",
        labels={"energy_deficit":"Deficit (MU)"},
        title="Deficit distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # Choropleth map
    # ----------------------------
    if india_geo is not None:
        st.subheader("State-wise Energy Deficit (choropleth)")
        state_sum = filtered.groupby("state", dropna=False)["energy_deficit"].sum().reset_index()
        try:
            fig_map = px.choropleth_mapbox(
                state_sum,
                geojson=india_geo,
                locations="state",
                featureidkey=featureidkey,
                color="energy_deficit",
                color_continuous_scale="OrRd",
                mapbox_style="carto-positron",
                zoom=3.5,
                center={"lat": 23.0, "lon": 82.0},
                opacity=0.7,
                labels={"energy_deficit": "Deficit (MU)"},
                title="Total Energy Deficit by State (MU)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning("Could not plot map: state names in dataset may not match GeoJSON properties.")
            st.write("GeoJSON property key used:", featureidkey)
            st.write("Example state names in dataset:", state_sum["state"].unique()[:10])
    else:
        st.info("India geojson not available; map disabled.")

    # ----------------------------
    # Seasonality Insights
    # ----------------------------
    st.subheader("💡 Seasonal Insights (Quarter-wise)")

    quarter_avg = filtered.groupby("quarter")[["energy_requirement_mu","energy_availability_mu","energy_deficit"]].mean().reset_index()
    st.dataframe(quarter_avg)

    # Alert unusual energy requirements
    for _, row in quarter_avg.iterrows():
        avg_req = row["energy_requirement_mu"]
        quarter = row["quarter"]
        # You can set a heuristic threshold (e.g., ±20% from median)
        median_req = df["energy_requirement_mu"].median()
        if avg_req > 1.2 * median_req:
            st.warning(f"⚠️ Unusually high energy requirement in {quarter}: {avg_req:.1f} MU")
        elif avg_req < 0.8 * median_req:
            st.info(f"ℹ️ Lower than usual energy requirement in {quarter}: {avg_req:.1f} MU")


# ----------------------------
# Page 3: ML Models (independent of EDA filters)
# ----------------------------
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models and Results (Seasonality-Aware)")
    st.markdown("""
    Models are trained on the **full cleaned dataset** (no filters).  
    Regression models are tested **with and without quarter features** to see the impact of seasonality.
    """)

    model_opts = st.multiselect("Select regression models:", 
                                ["Linear Regression", 
                                 "Decision Tree",
                                 "Random Forest",
                                 "Gradient Boosting",
                                 "XGBoost"],
                                default=["Linear Regression", "Random Forest"])

    # --- Prepare dataset ---
    df_ml = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
    df_ml["quarter"] = df_ml["quarter"].astype("category")

    # Base regression features
    base_features = ["energy_requirement_mu","energy_availability_mu","gap"]
    X_base = df_ml[base_features]
    y_reg = df_ml["energy_deficit"].astype(float)

    # Seasonality-aware features (one-hot encode quarter)
    df_season = pd.get_dummies(df_ml, columns=["quarter"], drop_first=True)
    seasonal_features = base_features + [col for col in df_season.columns if col.startswith("quarter_")]
    X_season = df_season[seasonal_features]

    # Train-test split
    Xtr_base, Xte_base, ytr_base, yte_base = train_test_split(X_base, y_reg, test_size=0.3, random_state=42)
    Xtr_season, Xte_season, ytr_season, yte_season = train_test_split(X_season, y_reg, test_size=0.3, random_state=42)

    # --- Helper RMSE function ---
    def safe_rmse(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true, dtype=float).ravel() - np.array(y_pred, dtype=float).ravel()) ** 2))

    # Import regression models
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt
    import plotly.express as px

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }

    results_list = []

    for name, model in regression_models.items():
        if name in model_opts:
            st.markdown(f"---\n### {name}")

            # Base model
            model.fit(Xtr_base, ytr_base)
            preds_base = model.predict(Xte_base)
            rmse_base = safe_rmse(yte_base, preds_base)
            r2_base = r2_score(yte_base, preds_base)

            # Seasonality-aware model
            model.fit(Xtr_season, ytr_season)
            preds_season = model.predict(Xte_season)
            rmse_season = safe_rmse(yte_season, preds_season)
            r2_season = r2_score(yte_season, preds_season)

            # Display metrics
            st.write(f"**Base Features:** RMSE = {rmse_base:.3f} | R² = {r2_base:.3f}")
            st.write(f"**Seasonality-Aware:** RMSE = {rmse_season:.3f} | R² = {r2_season:.3f}")

            # Store results
            results_list.append([name, round(rmse_base,3), round(r2_base,3), round(rmse_season,3), round(r2_season,3)])

            # Plot predicted vs actual (side-by-side using Plotly)
            df_plot = pd.DataFrame({
                "Actual_Base": yte_base,
                "Predicted_Base": preds_base,
                "Actual_Season": yte_season,
                "Predicted_Season": preds_season
            })

            fig = px.scatter(df_plot, x="Actual_Base", y="Predicted_Base", opacity=0.7,
                             labels={"Actual_Base":"Actual Energy Deficit","Predicted_Base":"Predicted (Base)"},
                             title=f"{name}: Actual vs Predicted (Base)")
            fig.add_scatter(x=df_plot["Actual_Season"], y=df_plot["Predicted_Season"], mode='markers',
                            name="Predicted (Seasonality)", marker_color='orange')
            fig.add_scatter(x=[df_plot["Actual_Base"].min(), df_plot["Actual_Base"].max()],
                            y=[df_plot["Actual_Base"].min(), df_plot["Actual_Base"].max()],
                            mode='lines', line=dict(color='red', dash='dash'), name='Ideal')
            st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    if results_list:
        st.write("### 📊 Regression Model Comparison (Base vs Seasonality-Aware)")
        st.dataframe(pd.DataFrame(results_list, columns=["Model","RMSE_Base","R2_Base","RMSE_Season","R2_Season"]))

    # ----------------------------
    # Classification Models
    # ----------------------------
    st.markdown("---\n### Classification Models (deficit_flag)")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    X_clf = df_ml[base_features]
    y_clf = df_ml["deficit_flag"].astype(int)
    Xtr_clf, Xte_clf, ytr_clf, yte_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr_clf, ytr_clf)
    pred_knn = knn.predict(Xte_clf)
    st.subheader("K-Nearest Neighbors")
    st.write("Accuracy:", round(accuracy_score(yte_clf, pred_knn), 3))
    st.text(classification_report(yte_clf, pred_knn))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(Xtr_clf, ytr_clf)
    pred_nb = nb.predict(Xte_clf)
    st.subheader("Gaussian Naive Bayes")
    st.write("Accuracy:", round(accuracy_score(yte_clf, pred_nb), 3))
    st.text(classification_report(yte_clf, pred_nb))

    # Logistic Regression
    logr = LogisticRegression(max_iter=1000)
    logr.fit(Xtr_clf, ytr_clf)
    pred_logr = logr.predict(Xte_clf)
    st.subheader("Logistic Regression")
    st.write("Accuracy:", round(accuracy_score(yte_clf, pred_logr), 3))
    st.text(classification_report(yte_clf, pred_logr))

    # ----------------------------
    # Clustering
    # ----------------------------
    st.markdown("---\n### K-Means Clustering")
    from sklearn.cluster import KMeans
    X_cluster = df_ml[base_features]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    df_cluster = X_cluster.copy()
    df_cluster["Cluster"] = clusters
    st.dataframe(df_cluster.head())



# ----------------------------
# Page 4: Prediction (user inputs with seasonal check)
# ----------------------------
elif page == "🔮 Prediction":
    st.title("🔮 Predict Energy Deficit / Surplus (Seasonality Aware)")
    st.markdown("""
    This prediction uses models trained on the **full dataset**.  
    You can provide inputs, select a model, and check **seasonal energy patterns**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        state_in = st.selectbox("State", sorted(df["state"].dropna().unique()))
    with col2:
        quarter_in = st.selectbox("Quarter", sorted(df["quarter"].dropna().unique()))
    with col3:
        model_choice = st.selectbox("Model for prediction", [
            "Linear Regression", 
            "Random Forest Regressor", 
            "Decision Tree Regressor", 
            "Gradient Boosting Regressor"
        ])

    # Numeric inputs
    nr1, nr2 = st.columns(2)
    with nr1:
        req_in = st.number_input(
            "Energy Requirement (MU)", 
            min_value=0.0, 
            value=float(df["energy_requirement_mu"].median()), 
            step=50.0
        )
    with nr2:
        avail_in = st.number_input(
            "Energy Availability (MU)", 
            min_value=0.0, 
            value=float(df["energy_availability_mu"].median()), 
            step=50.0
        )

    if st.button("Predict"):
        # Train model on full dataset
        df_train = df.dropna(subset=["energy_requirement_mu","energy_availability_mu","energy_deficit","gap"])
        X_full = df_train[["energy_requirement_mu","energy_availability_mu","gap"]]
        y_full = df_train["energy_deficit"].astype(float)
        gap_in = req_in - avail_in
        x_input = np.array([[req_in, avail_in, gap_in]])

        # Train chosen model
        if model_choice == "Linear Regression":
            model = LinearRegression().fit(X_full, y_full)
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42).fit(X_full, y_full)
        elif model_choice == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(random_state=42).fit(X_full, y_full)

        # Predict deficit/surplus
        pred_val = model.predict(x_input)[0]

        # ---- Seasonal check ----
        seasonal_median = df[(df["state"]==state_in) & (df["quarter"]==quarter_in)]["energy_requirement_mu"].median()
        if req_in > seasonal_median * 1.2:  # 20% above typical
            st.warning(f"⚠️ Energy requirement is unusually high for {state_in} in {quarter_in} (Expected ~{seasonal_median:.1f} MU).")
        elif req_in < seasonal_median * 0.8:  # 20% below typical
            st.info(f"ℹ️ Energy requirement is unusually low for {state_in} in {quarter_in} (Expected ~{seasonal_median:.1f} MU).")

        # ---- Display predicted deficit/surplus ----
        if pred_val > 0:
            st.error(f"Predicted Energy Deficit for {state_in} ({quarter_in}): {pred_val:.2f} MU")
        else:
            st.success(f"Predicted Energy Surplus for {state_in} ({quarter_in}): {abs(pred_val):.2f} MU")

        









