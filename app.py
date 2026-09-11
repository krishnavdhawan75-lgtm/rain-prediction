import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("🌧️ Rain Prediction System")

# Load dataset
data = pd.read_csv("rain_prediction_dataset.csv")
data.columns = data.columns.str.strip()
st.subheader("Enter Weather Conditions")

temperature = st.number_input("Temperature", value=28.0)
humidity = st.number_input("Humidity", value=82.0)
windspeed = st.number_input("Windspeed", value=14.0)
pressure = st.number_input("Pressure", value=1005.0)
cloudcover = st.number_input("CloudCover", value=65.0)

# Prepare data
X = data[["Temperature", "Humidity", "Windspeed", "Pressure", "CloudCover"]]
y = data["Rain"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

if st.button("Predict Rain"):
    prediction = model.predict([[
        temperature,
        humidity,
        windspeed,
        pressure,
        cloudcover
    ]])[0]

    st.success(f"Prediction: {prediction}")
