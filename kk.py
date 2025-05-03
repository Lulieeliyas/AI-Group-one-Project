import streamlit as st
import pandas as pd
import joblib

# Try loading the model
try:
    model = joblib.load("house_price_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'house_price_model.pkl' not found. Please train and save the model first.")
    st.stop()

# Title and intro
st.title("🏠 House Price Predictor")
st.markdown("Enter property details to estimate the price.")

# User inputs
area = st.slider("Area (sq ft)", 500, 15000, 2000)
bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("Number of Stories", [1, 2, 3, 4])
mainroad = st.radio("Main Road Access", ["yes", "no"])
guestroom = st.radio("Guest Room", ["yes", "no"])
basement = st.radio("Basement", ["yes", "no"])
hotwaterheating = st.radio("Hot Water Heating", ["yes", "no"])
airconditioning = st.radio("Air Conditioning", ["yes", "no"])
parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])
prefarea = st.radio("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Encode input like training
input_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": 1 if mainroad == "yes" else 0,
    "guestroom": 1 if guestroom == "yes" else 0,
    "basement": 1 if basement == "yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "yes" else 0,
    "airconditioning": 1 if airconditioning == "yes" else 0,
    "parking": parking,
    "prefarea": 1 if prefarea == "yes" else 0,
    "furnishingstatus": {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}[furnishingstatus]
}])

# Show prediction only when button is clicked
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹ {int(prediction):,}")
