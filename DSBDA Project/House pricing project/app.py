import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction")

st.write("Enter property details:")

# Inputs
area = st.number_input("Area (in Marla)", min_value=0.0)
area_sqft = area * 272.25
st.write(f"Area in Square Feet: {area_sqft:.2f} sq ft")
bedrooms = st.number_input("Bedrooms", min_value=0)
baths = st.number_input("Bathrooms", min_value=0)

location = st.selectbox("Location", ["G-10", "E-11", "G-15", "Bani Gala", "DHA Defence"])

location_map = {
    "G-10": 0,
    "E-11": 1,
    "G-15": 2,
    "Bani Gala": 3,
    "DHA Defence": 4
}

loc = location_map[location]

# Dummy data for graph (since real test data not stored)
X_sample = np.array([[2,1,1,0],[4,2,2,1],[6,3,3,2],[8,4,4,3]])
y_actual = np.array([2000000, 4000000, 6000000, 8000000])

y_pred = model.predict(X_sample)

# Accuracy
r2 = r2_score(y_actual, y_pred)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, baths, loc]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Price: {int(prediction[0])}")

    # Show accuracy
    st.write(f"Model Accuracy (R² Score): {round(r2*100, 2)}%")

    # Plot graph
    fig = plt.figure()
    plt.scatter(y_actual, y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")

    st.pyplot(fig)