

import streamlit as st
import pandas as pd
import numpy as np
# Load data
data_path = 'House price.csv'
df = pd.read_csv(data_path)

# Streamlit app layout
st.title("House Price Prediction")
st.write("This app predicts house prices based on various features.")

# Display the dataset
if st.checkbox("Show Dataset"):
    st.write(df)

# Data Overview
st.subheader("Data Overview")
st.write(df.describe())

# Feature selection
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
target = 'price'

# Input for prediction
st.sidebar.header("User Input Features")
def user_input_features():
    area = st.sidebar.number_input('Area (sq ft)', min_value=0, value=1500)
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=1, max_value=6, value=3)
    bathrooms = st.sidebar.number_input('Bathrooms', min_value=1, max_value=4, value=2)
    stories = st.sidebar.number_input('Stories', min_value=1, max_value=4, value=2)
    parking = st.sidebar.number_input('Parking Spaces', min_value=0, max_value=3, value=1)
    
    return pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]], columns=features)

input_data = user_input_features()

# Train the model
X = df[features]
y = df[target]


st.subheader("Predicted House Price")


# Visualizations
st.subheader("Price Distribution")
st.pyplot()

# Feature importance (optional)
if st.checkbox("Show Feature Importances"):
    importance = model.coef_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    st.pyplot()