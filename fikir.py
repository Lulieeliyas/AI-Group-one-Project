
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
df = pd.read_csv("C:\\Users\\gh\\Downloads\\House Price.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# Save models
with open("models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

# Streamlit UI
st.title("House Price Prediction App")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
stories = st.slider("Number of Stories", 1, 4, 2)
parking = st.slider("Number of Parking Spaces", 0, 5, 1)

# Categorical Inputs
mainroad = st.selectbox("Main Road", ['yes', 'no'])
guestroom = st.selectbox("Guest Room", ['yes', 'no'])
basement = st.selectbox("Basement", ['yes', 'no'])
hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Convert inputs
def encode_input(value, column):
    return label_encoders[column].transform([value])[0]

input_data = np.array([
    area, bedrooms, bathrooms, stories, parking,
    encode_input(mainroad, 'mainroad'),
    encode_input(guestroom, 'guestroom'),
    encode_input(basement, 'basement'),
    encode_input(hotwaterheating, 'hotwaterheating'),
    encode_input(airconditioning, 'airconditioning'),
    encode_input(prefarea, 'prefarea'),
    encode_input(furnishingstatus, 'furnishingstatus')
]).reshape(1, -1)

# Model Selection
model_choice = st.selectbox("Select a Model", list(trained_models.keys()))

if st.button("Predict Price"):
    model = trained_models[model_choice]
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")