import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="House Price Predictor", layout="wide")


model_path = os.path.join(os.path.dirname(__file__), "models/house_price_model.pkl")
model = joblib.load(model_path)


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
st.sidebar.header("Enter House Features")
square_footage = st.sidebar.slider(
    "Square Footage", min_value=100, max_value=10000, value=1500, step=100,
    help="Total area of the house in square feet"
)

bedrooms = st.sidebar.slider(
    "Number of Bedrooms", min_value=1, max_value=10, value=3, step=1,
    help="Total number of bedrooms"
)

bathrooms = st.sidebar.slider(
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

data_path = os.path.join(os.path.dirname(__file__),"data/house_price_data.csv")
df = pd.read_csv(data_path)

with st.expander("📊 View Sample Data"):
    st.write(df.head())

st.subheader("📈 Price vs Square Footage")
fig = px.scatter(df, x="square_footage", y="price", color="bedrooms",
                 labels={"square_footage": "Square Footage", "price": "Price ($)"},
                 title="House Prices Based on Square Footage and Bedrooms")
st.plotly_chart(fig, use_container_width=True)