import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("House Price.csv")

data = load_data()

# Define features and target
features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
    'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
    'parkig', 'prefarea', 'furnishingstatus'
]
target = 'price'

X = data[features]
y = data[target]

# Categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Save model
joblib.dump(model, "house_price_model.pkl")

# Streamlit UI
st.title("🏠 House Price Prediction")
st.markdown("Predict house prices based on property features")

# Sidebar for user inputs
st.sidebar.header("Property Details")

area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=15000, value=2000)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4])
stories = st.sidebar.selectbox("Stories", [1, 2, 3, 4])
mainroad = st.sidebar.radio("Main Road Access", ['yes', 'no'])
guestroom = st.sidebar.radio("Guest Room", ['yes', 'no'])
basement = st.sidebar.radio("Basement", ['yes', 'no'])
hotwaterheating = st.sidebar.radio("Hot Water Heating", ['yes', 'no'])
airconditioning = st.sidebar.radio("Air Conditioning", ['yes', 'no'])
parking = st.sidebar.selectbox("Parking Spaces", [0, 1, 2, 3])
prefarea = st.sidebar.radio("Preferred Area", ['yes', 'no'])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Create input DataFrame
input_data = pd.DataFrame([{
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus
}])

# Show model metrics
with st.expander("Model Performance"):
    st.write(f"Training R² Score: {train_score:.3f}")
    st.write(f"Test R² Score: {test_score:.3f}")
    st.write(f"Mean Absolute Error: ₹{mae:,.0f}")

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"## Predicted Price: ₹{int(prediction):,}")
    
    # Show input summary
    st.subheader("Input Summary")
    st.json(input_data.iloc[0].to_dict())

# Data preview
with st.expander("View Raw Data"):
    st.dataframe(data)