import streamlit as st
import pickle  # or joblib
import numpy as np
import pandas as pd


# Load the trained model
def load_model():
    with open("./decision_tree_model.sav", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Iris Flower Prediction")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)

    st.write(f"Predicted Iris Species: {prediction[0]}")
