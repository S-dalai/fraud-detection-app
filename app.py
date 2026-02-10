import streamlit as st
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")

distance_from_home = st.number_input("Distance from Home", value=0.0)
distance_from_last_transaction = st.number_input("Distance from Last Transaction", value=0.0)
ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", value=1.0)
repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
used_chip = st.selectbox("Used Chip", [0, 1])
used_pin_number = st.selectbox("Used PIN Number", [0, 1])
online_order = st.selectbox("Online Order", [0, 1])

if st.button("Check Transaction"):
    data = np.array([[distance_from_home,
                      distance_from_last_transaction,
                      ratio_to_median_purchase_price,
                      repeat_retailer,
                      used_chip,
                      used_pin_number,
                      online_order]])

    pred = model.predict(data)
    if pred[0] == 1:
        st.error("🚨 FRAUD ALERT")
    else:
        st.success("✅ NORMAL TRANSACTION")
