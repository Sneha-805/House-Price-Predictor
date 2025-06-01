import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), "models/house_price_model.pkl")
model = joblib.load(model_path)

st.title("🏠 House Price Predictor")

# User inputs
square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    # Create DataFrame with proper column names
    features = pd.DataFrame([[square_footage, bedrooms, bathrooms]],
                            columns=["square_footage", "bedrooms", "bathrooms"])

    # Make prediction
    prediction = model.predict(features)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

    # Feature importance (absolute coefficients)
    feature_names = ["Square Footage", "Bedrooms", "Bathrooms"]
    coefs = model.coef_
    importance = np.abs(coefs)

    # Prepare DataFrame for plotting
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True).set_index("Feature")

    st.subheader("Feature Importance")
    st.bar_chart(df_importance)
