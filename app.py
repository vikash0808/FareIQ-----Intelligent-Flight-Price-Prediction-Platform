import gradio as gr
import pickle
import pandas as pd
import numpy as np

with open("flight_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
TIME_WINDOWS = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
AIRLINES = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]

def predict(
    airline,
    source_city,
    departure_time,
    stops,
    arrival_time,
    destination_city,
    class_type,
    duration,
    days_left,
):
    if source_city == destination_city:
        return "Source and destination cannot be the same."

    raw = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": class_type,
        "duration": duration,
        "days_left": days_left,
    }

    input_df = pd.DataFrame([raw])

    categorical_cols = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
    ]

    encoded_df = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        drop_first=False,
    )

    final = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns,
    )

    for col in encoded_df.columns:
        if col in final.columns:
            final[col] = encoded_df[col].values

    prediction = model.predict(final)[0]

    return f"₹{prediction:,.2f}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(AIRLINES, label="✈️ Airline"),
        gr.Dropdown(CITIES, label="🛫 Source City"),
        gr.Dropdown(TIME_WINDOWS, label="⏰ Departure Time"),
        gr.Dropdown(["zero", "one", "two_or_more"], label="🛑 Stops"),
        gr.Dropdown(TIME_WINDOWS, label="🛬 Arrival Time"),
        gr.Dropdown(CITIES, label="🛬 Destination City"),
        gr.Dropdown(["Economy", "Business"], label="💺 Class"),
        gr.Slider(0.5, 50.0, value=2.5, step=0.5, label="⏱️ Duration (Hours)"),
        gr.Slider(1, 50, value=10, step=1, label="📅 Days Until Departure"),
    ],
    outputs=gr.Textbox(label="💸 Predicted Fare"),
    title="✈️ Flight Price Predictor",
    description="XGBoost model trained on Indian domestic flight data",
)

if __name__ == "__main__":
    demo.launch()
