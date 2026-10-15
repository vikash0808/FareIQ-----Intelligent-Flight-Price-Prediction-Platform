import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Flight Intelligence Platform", page_icon="✈️", layout="wide")

CITIES        = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
TIME_WINDOWS  = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS         = ["zero", "one", "two_or_more"]
CLASSES       = ["Economy", "Business"]
AIRLINES      = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/Clean_Dataset.csv")
    except FileNotFoundError:
        return None

df = load_data()

st.sidebar.title("✈️ Airfare Control Center")
app_mode = st.sidebar.radio("Navigate:", ["🔮 Live ML Price Predictor", "📊 Historical Market Trends"])

# ── MODULE 1: PREDICTOR ──────────────────────────────────────────────
if app_mode == "🔮 Live ML Price Predictor":
    st.title("🔮 Airfare Forecasting Engine")
    st.caption("Dispatches to FastAPI + XGBoost microservice for real-time inference.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        source_city    = st.selectbox("🛫 Source City",       CITIES)
        departure_time = st.selectbox("⏰ Departure Window",  TIME_WINDOWS)
        stops          = st.selectbox("🛑 Stops",              STOPS)
        class_type     = st.selectbox("💺 Cabin Class",       CLASSES)
    with col2:
        destination_city = st.selectbox("🛬 Destination City", CITIES)
        arrival_time     = st.selectbox("🛬 Arrival Window",   TIME_WINDOWS)
        airline          = st.selectbox("✈️ Airline",           AIRLINES)
        days_left        = st.number_input("📅 Days Until Departure", min_value=1, max_value=50, value=10, step=1)

    duration = st.slider("⏱️ Flight Duration (Hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)
    st.markdown("---")

    if st.button("🔮 Forecast Ticket Price", use_container_width=True):
        if source_city == destination_city:
            st.error("❌ Source and destination cannot be the same.")
        else:
            payload = {
                "airline":          airline,
                "source_city":      source_city,
                "departure_time":   departure_time,
                "stops":            stops,
                "arrival_time":     arrival_time,
                "destination_city": destination_city,
                "class_type":       class_type,
                "duration":         float(duration),
                "days_left":        int(days_left),
            }
            try:
                with st.spinner("Running inference..."):
                    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

                if response.status_code == 200:
                    price = float(response.json()["predicted_price"])
                    st.balloons()
                    st.success(f"### 💸 Estimated Fare: ₹{price:,.2f}")
                    st.caption("Prediction based on historical pricing patterns via XGBoost.")
                else:
                    st.error(f"❌ API Error: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach API. Make sure your FastAPI server is running in the other terminal!")

# ── MODULE 2: ANALYTICS ──────────────────────────────────────────────
elif app_mode == "📊 Historical Market Trends":
    st.title("📊 Dataset Analysis & Visualizations")
    st.markdown("---")

    if df is None:
        st.warning("⚠️ `data/Clean_Dataset.csv` not found.")
    else:
        tab1, tab2 = st.tabs(["✈️ Airline Pricing", "📉 Booking Window"])

        with tab1:
            st.subheader("Average Fare by Airline")
            airline_counts = df.groupby("airline")["price"].mean().sort_values()
            st.bar_chart(airline_counts)

        with tab2:
            st.subheader("Price vs Days Left to Departure")
            trend = df.groupby("days_left")["price"].mean()
            st.line_chart(trend)

        st.markdown("---")
        st.subheader("📋 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Avg Fare (INR)", f"₹{df['price'].mean():,.0f}")
        c3.metric("Routes",         df.groupby(["source_city", "destination_city"]).ngroups)