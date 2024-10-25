import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# File to store user data
USER_DATA_FILE = 'users.json'

# Load user data from JSON file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save user data to JSON file
def save_user_data(users_db):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_db, f)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('ADANI ENTERPRISES LIMITED.csv')
    data.columns = data.columns.str.strip()  # Clean column names
    return data

# Preprocess the data
def preprocess_data(data):
    required_columns = ['HIGH', 'LOW', 'close', 'Day', 'Month']
    for column in required_columns:
        if column not in data.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return None, None, None, None

    data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
    data['Month'] = pd.to_numeric(data['Month'], errors='coerce')

    features = data[['HIGH', 'LOW', 'close', 'Day', 'Month']].values
    target = data['close'].values.reshape(-1, 1)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    return scaled_features, scaled_target, scaler_features, scaler_target

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape, name='lstm_layer'))
    model.add(Dense(32, activation='relu', name='dense_32'))
    model.add(Dense(1, name='output'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# User signup function
def signup(username, password):
    username = username.strip()
    users_db = load_user_data()
    
    if username in users_db:
        st.error("Username already exists.")
        return False
    else:
        users_db[username] = password
        save_user_data(users_db)  # Save the updated user data
        st.success("Signup successful! You can now log in.")
        return True

# User login function
def login(username, password):
    username = username.strip()
    st.write("Attempting to log in with:", username)  # Debugging line

    users_db = load_user_data()  # Load user data from the file
    if username in users_db:
        if users_db[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid password.")
            return False
    else:
        st.error("Username not found.")
        return False

# Main function to run the app
def main():
    st.title("Stock Price Prediction")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        st.subheader("Login / Signup")

        login_or_signup = st.radio("Choose an option", ("Login", "Signup"))

        if login_or_signup == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                if login(username, password):
                    st.session_state['logged_in'] = True
                else:
                    st.session_state['logged_in'] = False

        else:  # Signup
            username = st.text_input("New Username")
            password = st.text_input("New Password", type='password')
            if st.button("Signup"):
                signup(username, password)

    if st.session_state['logged_in']:
        st.write(f"Welcome, {st.session_state['username']}!")

        data = load_data()
        X, y, scaler_features, scaler_target = preprocess_data(data)
        if X is None or y is None:
            return

        # Reshaping the input data for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        if st.button("Train Model"):
            model.fit(X_train, y_train, epochs=50, batch_size=32)
            st.success("Model trained successfully!")

        st.subheader("Enter input features for prediction")
        high = st.number_input("High Price:", min_value=0.0)
        low = st.number_input("Low Price:", min_value=0.0)
        close = st.number_input("Previous Close Price:", min_value=0.0)
        day = st.number_input("Day (1-31):", min_value=1, max_value=31)
        month = st.number_input("Month (1-12):", min_value=1, max_value=12)

        if st.button("Make Prediction"):
            input_data = np.array([[high, low, close, day, month]])
            input_data_scaled = scaler_features.transform(input_data)
            input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

            predicted_price = model.predict(input_data_reshaped)
            predicted_price = scaler_target.inverse_transform(predicted_price)

            st.subheader("Predicted Closing Price")
            st.write("Predicted Closing Price: {:.2f}".format(predicted_price[0][0]))





if __name__ == "__main__":
    main()
