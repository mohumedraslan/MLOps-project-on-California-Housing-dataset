import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("california_house_price_model.pkl")

st.set_page_config(page_title="California House Price Predictor", layout="wide")
st.title("🏡 California House Price Predictor")
st.markdown("""
Predict the **median house price** for a California block group using a linear regression model.
All inputs are real-world values — just enter them and get the predicted price.
""")

# -----------------------------
# Step 1: User inputs with guidance
# -----------------------------
st.subheader("House Information")

st.markdown("""
**Instructions:**  
Enter realistic values for the house and neighborhood. Typical ranges are provided.
""")

medinc = st.number_input(
    "Median Income (in $10k)", min_value=0.0, max_value=20.0, value=5.0,
    help="Median income in the block group in $10,000s. Typical range: 0–15"
)
house_age = st.number_input(
    "House Age (years)", min_value=0, max_value=60, value=20,
    help="Median house age in the block group. Typical range: 0–60 years"
)
ave_rooms = st.number_input(
    "Average Rooms per House", min_value=1.0, max_value=15.0, value=5.0,
    help="Average number of rooms per house. Typical range: 3–10"
)
ave_bedrooms = st.number_input(
    "Average Bedrooms per House", min_value=1.0, max_value=5.0, value=2.0,
    help="Average number of bedrooms per house. Typical range: 1–5"
)
population = st.number_input(
    "Population in Block Group", min_value=50, max_value=5000, value=1000,
    help="Population in the block group. Typical range: 100–3000"
)
ave_occup = st.number_input(
    "Average Occupants per Household", min_value=1.0, max_value=10.0, value=3.0,
    help="Average number of people per house. Typical range: 1–5"
)
latitude = st.number_input(
    "Latitude", min_value=32.0, max_value=42.0, value=37.0,
    help="Latitude of the block group. California: ~32–42°N"
)
longitude = st.number_input(
    "Longitude", min_value=-125.0, max_value=-114.0, value=-120.0,
    help="Longitude of the block group. California: ~-125 to -114°W"
)

# -----------------------------
# Step 2: Prepare input
# -----------------------------
input_df = pd.DataFrame([[
    medinc, house_age, ave_rooms, ave_bedrooms,
    population, ave_occup, latitude, longitude
]], columns=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"])

# -----------------------------
# Step 3: Predict
# -----------------------------
prediction = None
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0] * 100000  # scale to dollars
    st.success(f"🏠 Predicted Median House Price: ${prediction:,.2f}")
    st.info("This is the predicted median house price for the block group based on the provided features.")

# -----------------------------
# Step 4: Optional logging
# -----------------------------
if st.button("Save Prediction") and prediction is not None:
    log = input_df.copy()
    log["PredictedPrice"] = prediction
    log_file = "house_price_predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.info(f"Prediction saved locally to {log_file}")
