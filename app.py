import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load('car_model.pkl')

# 2. App Title and Description
st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to get an estimated price.")

# 3. User Inputs
year = st.number_input("Car Year", min_value=1990, max_value=2024, value=2018)
mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
hp = st.number_input("Horsepower", min_value=50, max_value=1000, value=150)

# 4. Prediction Logic
if st.button("Predict Price"):
    # Prepare input data as a 2D array
    features = np.array([[year, mileage, hp]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display result
    st.success(f"Estimated Price: ${round(prediction[0], 2)}")