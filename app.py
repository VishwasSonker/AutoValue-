import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AutoValue | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    .stApp {
        background: #c0d6e4;
        color: #000000;
    }

    .block-container {
        max-width: 1120px;
        padding: 42px 28px 60px;
    }

    /* ---------- Header ---------- */
    .app-header {
        text-align: center;
        margin-bottom: 42px;
    }

    .app-title {
        font-size: 55px;
        font-weight: 750;
        letter-spacing: -1.2px;
        margin: 0;
        color: #17191c;
    }

    .app-subtitle {
        margin: 9px auto 0;
        max-width: 620px;
        color: #70757d;
        font-size: 20px;
        line-height: 1.6;
    }

    /* ---------- Section ---------- */
    .section {
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .section-number {
        color: #6b7280;
        font-size: 40px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 36px;
        font-weight: 700;
        margin-top: 3px;
        color: #17191c;
    }

    .section-description {
        color: #777d85;
        font-size: 30px;
        margin-top: 4px;
    }

    /* ---------- Sidebar heading ---------- */

    section[data-testid="stSidebar"] h2 {
        font-size: 24px !important;
        color: #000000 !important;
    }

    /* Sidebar input labels */
    section[data-testid="stSidebar"] label {
        font-size: 25px !important;
        font-weight: 600 !important;
        color: #000000 !important;
    }

    /* Input text */
    section[data-testid="stSidebar"] input {
        font-size: 25px !important;
    }

    /* ---------- Prediction ---------- */
    .prediction {
        background: #17191c;
        color: white;
        border-radius: 14px;
        padding: 30px;
        text-align: center;
    }

    .prediction-label {
        color: #b9bec5;
        font-size: 16px;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-top: 7px;
    }

    .prediction-note {
        color: #9ca3af;
        font-size: 20px;
        margin-top: 8px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        height: 48px;
        border-radius: 9px;
        font-weight: 650;
        font-size: 20px;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        min-height: 48px;
        font-size: 18px;
        border-radius: 10px;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 20px;
        font-weight: 700;
        color: #17191c;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #000000;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 10px;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #9aa0a8;
        font-size: 15px;
        margin-top: 45px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL / DATA
# ============================================================
@st.cache_resource
def load_model():
    with open("linear_model.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_feature_importance():
    return pd.read_excel("feature_importance.xlsx")


model = load_model()
feature_importance = load_feature_importance()

# ============================================================
# FEATURE ORDER
# Must match the model training order exactly.
# ============================================================
FEATURES = [
    "Horsepower_No",
    "Torque_No",
    "Make_Aston Martin",
    "Make_Audi",
    "Make_BMW",
    "Make_Bentley",
    "Make_Ford",
    "Make_Mercedes-Benz",
    "Make_Nissan",
    "Body Size_Compact",
    "Body Size_Large",
    "Body Size_Midsize",
    "Body Style_Cargo Minivan",
    "Body Style_Cargo Van",
    "Body Style_Convertible",
    "Body Style_Convertible SUV",
    "Body Style_Coupe",
    "Body Style_Hatchback",
    "Body Style_Passenger Minivan",
    "Body Style_Passenger Van",
    "Body Style_Pickup Truck",
    "Body Style_SUV",
    "Body Style_Sedan",
    "Body Style_Wagon",
    "Engine Aspiration_Electric Motor",
    "Engine Aspiration_Naturally Aspirated",
    "Engine Aspiration_Supercharged",
    "Engine Aspiration_Turbocharged",
    "Engine Aspiration_Twin-Turbo",
    "Engine Aspiration_Twincharged",
    "Drivetrain_4WD",
    "Drivetrain_AWD",
    "Drivetrain_FWD",
    "Drivetrain_RWD",
    "Transmission_automatic",
    "Transmission_manual",
]

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <div class="app-title">AutoValue</div>
    <div class="app-subtitle">
        A simple vehicle pricing tool powered by a Linear Regression model.
        Enter the vehicle specifications below to estimate its price.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 01 — VEHICLE SPECIFICATIONS
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-number">01 — Vehicle specifications</div>
    <div class="section-title">Tell us about the vehicle</div>
    <div class="section-description">
        Provide the specifications used by the trained model.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Row 1: performance
st.markdown("**<h3>Performance</h3>**", unsafe_allow_html=True)
perf1, perf2 = st.columns(2)

with perf1:
    horsepower = st.number_input(
        "Horsepower",
        min_value=0,
        max_value=1000,
        value=300,
        step=1,
    )

with perf2:
    torque = st.number_input(
        "Torque",
        min_value=0,
        max_value=1500,
        value=400,
        step=1,
    )

st.markdown("---")

# Row 2: basic vehicle information
st.markdown("**<h3>Vehicle</h3>**", unsafe_allow_html=True)
vehicle1, vehicle2, vehicle3 = st.columns(3)

with vehicle1:
    make = st.selectbox(
        "Make",
        ["Aston Martin", "Audi", "BMW", "Bentley", "Ford",
         "Mercedes-Benz", "Nissan"],
    )

with vehicle2:
    body_size = st.selectbox(
        "Body Size",
        ["Compact", "Large", "Midsize"],
    )

with vehicle3:
    body_style = st.selectbox(
        "Body Style",
        ["Cargo Minivan", "Cargo Van", "Convertible", "Convertible SUV",
         "Coupe", "Hatchback", "Passenger Minivan", "Passenger Van",
         "Pickup Truck", "SUV", "Sedan", "Wagon"],
    )

st.markdown("---")

# Row 3: drivetrain
st.markdown("**<h3>Powertrain & Transmission</h3>**", unsafe_allow_html=True)
power1, power2, power3 = st.columns(3)

with power1:
    engine_aspiration = st.selectbox(
        "Engine Aspiration",
        ["Electric Motor", "Naturally Aspirated", "Supercharged",
         "Turbocharged", "Twin-Turbo", "Twincharged"],
    )

with power2:
    drivetrain = st.selectbox(
        "Drivetrain",
        ["4WD", "AWD", "FWD", "RWD"],
    )

with power3:
    transmission = st.selectbox(
        "Transmission",
        ["automatic", "manual"],
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 02 — PREDICTION
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-number">02 — Estimate</div>
    <div class="section-title">Get the estimated price</div>
    <div class="section-description">
        Review your selections and run the model.
    </div>
</div>
""", unsafe_allow_html=True)

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.metric("Make", make)

with summary2:
    st.metric("Performance", f"{horsepower} HP")

with summary3:
    st.metric("Drivetrain", drivetrain)

with summary4:
    st.metric("Transmission", transmission)

st.write("")

button_col1, button_col2, button_col3 = st.columns([1, 1.2, 1])

with button_col2:
    predict = st.button(
        "Estimate Vehicle Price",
        type="primary",
        use_container_width=True,
    )

if predict:
    data = {feature: 0 for feature in FEATURES}

    data["Horsepower_No"] = horsepower
    data["Torque_No"] = torque
    data[f"Make_{make}"] = 1
    data[f"Body Size_{body_size}"] = 1
    data[f"Body Style_{body_style}"] = 1
    data[f"Engine Aspiration_{engine_aspiration}"] = 1
    data[f"Drivetrain_{drivetrain}"] = 1
    data[f"Transmission_{transmission}"] = 1

    input_array = np.array([[data[feature] for feature in FEATURES]])
    prediction = model.predict(input_array)[0]

    st.write("")

    st.markdown(f"""
    <div class="prediction">
        <div class="prediction-label">Estimated vehicle price</div>
        <div class="prediction-value">${prediction:,.2f}</div>
        <div class="prediction-note">
            Based on the specifications provided above
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 03 — MODEL INSIGHT
# ============================================================
st.markdown("""
<div class="section">
    <div class="section-number">03 — Model insight</div>
    <div class="section-title">What influences the prediction?</div>
    <div class="section-description">
        Feature importance from the model-training pipeline.
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("View feature importance", expanded=True):
    plot_data = feature_importance.sort_values(
        "Feature Importance Score",
        ascending=True,
    )

    fig = px.bar(
        plot_data,
        x="Feature Importance Score",
        y="Variable",
        orientation="h",
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    )

    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=5, r=30, t=10, b=10),
        xaxis_title="Importance score",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    AutoValue · Car Price Prediction · Linear Regression
</div>
""", unsafe_allow_html=True)
