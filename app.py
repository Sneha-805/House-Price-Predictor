import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px

# Set page config must be the first Streamlit command
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models/house_price_model.pkl")
model = joblib.load(model_path)

# Custom CSS for button styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 House Price Predictor")

square_footage = st.number_input(
    "Square Footage", min_value=100, max_value=10000, value=1500, step=100,
    help="Total area of the house in square feet"
)

bedrooms = st.number_input(
    "Number of Bedrooms", min_value=1, max_value=10, value=3, step=1,
    help="Total number of bedrooms"
)

bathrooms = st.number_input(
    "Number of Bathrooms", min_value=1, max_value=10, value=2, step=1,
    help="Total number of bathrooms"
)

if st.button("Predict Price"):
    features = pd.DataFrame(
        [[square_footage, bedrooms, bathrooms]],
        columns=["square_footage", "bedrooms", "bathrooms"]
    )

    with st.spinner('Predicting price...'):
        prediction = model.predict(features)
    
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

    st.markdown("""
    <details>
    <summary style='font-size: 20px; font-weight: bold;'>🔍 What is Feature Importance?</summary>
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #d4d4d4; margin-top: 10px;">
        <p style="font-size: 16px;">
            When predicting house prices, our model looks at key features like:
            <ul>
                <li>📐 <b>Square Footage</b></li>
                <li>🛏️ <b>Bedrooms</b></li>
                <li>🛁 <b>Bathrooms</b></li>
            </ul>
            <b>Feature importance</b> tells us which of these have the <span style="color: green;"><b>most influence</b></span> on the predicted price.<br><br>
            In this chart, we show the importance based on the <b>absolute value of model coefficients</b>. A higher value means the feature plays a <span style="color: red;"><b>greater role</b></span> in determining the house price.
        </p>
    </div>
    </details>
    """, unsafe_allow_html=True)

    # Feature importance visualization
    st.subheader("📊 Feature Importance (Model Coefficients)")

    feature_names = ["Square Footage", "Bedrooms", "Bathrooms"]
    importance = np.abs(model.coef_)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    fig = px.bar(
        df_importance, x='Importance', y='Feature', orientation='h',
        title="Feature Importance",
        text='Importance',
        labels={"Importance": "Coefficient Magnitude"}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
