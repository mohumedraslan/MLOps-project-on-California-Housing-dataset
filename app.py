import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('california_house_price_model.pkl')

st.title("California House Price Predictor")

# Sliders for user input
med_inc = st.slider("Median Income", 0.0, 15.0, 3.87)  # Default to mean
house_age = st.slider("House Age", 0, 60, 29)  # Default to mean (rounded)

# Use mean values for other features
input_data = pd.DataFrame([[med_inc, house_age, 5.43, 1.10, 1425.48, 3.07, 35.63, -119.57]], 
                          columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                   'Population', 'AveOccup', 'Latitude', 'Longitude'])

# Predict and ensure non-negative
prediction = model.predict(input_data)[0]
prediction = max(0, prediction)  # Clip to ensure non-negative
st.write(f"Predicted Price: ${prediction * 100000:.2f}")
