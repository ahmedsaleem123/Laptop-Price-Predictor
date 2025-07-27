import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #4caf50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5em;
        color: #E0E0E0;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #1E1E1E;
        color: #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4caf50;
        margin-top: 20px;
        font-size: 1.2em;
        text-align: center;
    }
    .stSelectbox, .stSlider {
        background-color: #2C2C2C;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: Could not find {str(e)}. Please ensure model.pkl, scaler.pkl, and columns.pkl are in the same directory as app.py.")
    st.stop()

# Feature engineering functions
def extract_resolution(res):
    if isinstance(res, str):
        match = re.search(r'(\d+)x(\d+)', res)
        if match:
            return int(match.group(1)) * int(match.group(2))
    return 0

def extract_cpu_speed(cpu):
    if isinstance(cpu, str):
        match = re.search(r'(\d+\.\d+)(?:GHz)', cpu)
        if match:
            return float(match.group(1))
    return 2.0

def extract_storage(memory):
    if isinstance(memory, str) and 'SSD' in memory:
        match = re.search(r'(\d+)(GB|TB)', memory)
        if match:
            size = int(match.group(1))
            unit = match.group(2)
            return size * 1000 if unit == 'TB' else size
    return 0

# Currency conversion function
def convert_price(price_inr, currency):
    exchange_rates = {
        'INR': 1.0,
        'PKR': 3.33,
        'USD': 0.0119,
        'EUR': 0.0110,
        'GBP': 0.0093
    }
    return price_inr * exchange_rates.get(currency, 1.0)

# Streamlit app
st.markdown("<div class='main-title'>Laptop Price Predictor</div>", unsafe_allow_html=True)
st.write("Configure your laptop specifications and get a price estimate in your preferred currency.")

# Input sections
with st.container():
    st.markdown("<div class='section-header'>Laptop Brand and Type</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple'], help="Select the laptop manufacturer.")
    with col2:
        type_name = st.selectbox("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'], help="Choose the laptop category (e.g., Gaming for high-performance).")

with st.container():
    st.markdown("<div class='section-header'>Hardware Specifications</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6, step=0.1, help="Select screen size in inches.")
        resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '2560x1600'], help="Higher resolution improves display quality.")
    with col2:
        cpu = st.selectbox("Processor (CPU)", ['Intel Core i5 2.3GHz', 'Intel Core i7 2.7GHz', 'AMD Ryzen 5 2.1GHz'], help="Select CPU for performance.")
    with col3:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32], help="More RAM improves multitasking.")
        memory = st.selectbox("Storage", ['128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD', '1TB HDD'], help="SSD is faster; HDD offers more capacity.")

with st.container():
    st.markdown("<div class='section-header'>Operating System and Currency</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        op_sys = st.selectbox("Operating System", ['Windows', 'macOS', 'Linux'], help="Choose the operating system.")
    with col2:
        currency = st.selectbox("Currency", ['INR', 'PKR', 'USD', 'EUR', 'GBP'], help="Select currency for price display.")

# Predict button
if st.button("Predict Price"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Inches': [inches],
        'Ram': [int(ram)],
        'Resolution': [extract_resolution(resolution)],
        'CpuSpeed': [extract_cpu_speed(cpu)],
        'Storage': [extract_storage(memory)]
    })

    # Add dummy variables for categorical features
    for col in columns:
        if col.startswith('Company_') or col.startswith('TypeName_') or col.startswith('OpSys_'):
            input_data[col] = 0
    input_data[f'Company_{company}'] = 1
    input_data[f'TypeName_{type_name}'] = 1
    input_data[f'OpSys_{op_sys}'] = 1

    # Ensure all columns match training data
    try:
        input_data = input_data.reindex(columns=columns, fill_value=0)
    except ValueError as e:
        st.error(f"Error: Column mismatch. Ensure the model was trained with the same features. Details: {str(e)}")
        st.stop()

    # Scale and predict
    try:
        input_scaled = scaler.transform(input_data)
        pred_log = model.predict(input_scaled)
        pred_price_inr = np.exp(pred_log)[0]
        pred_price_converted = convert_price(pred_price_inr, currency)
        st.markdown(f"<div class='prediction-box'>Predicted Laptop Price: {currency} {pred_price_converted:.2f}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.stop()
else:
    st.info("Click 'Predict Price' to see the estimated price based on your selections.")
