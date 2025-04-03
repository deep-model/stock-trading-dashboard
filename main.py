import requests
import os
from datetime import datetime
from bs4 import BeautifulSoup
from twilio.rest import Client
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Load credentials from Streamlit secrets ---
TWILIO_SID = st.secrets["TWILIO_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_FROM = st.secrets["TWILIO_FROM"]
TWILIO_RECIPIENTS = st.secrets["TWILIO_RECIPIENTS"]

EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
EMAIL_RECIPIENTS = st.secrets["EMAIL_RECIPIENTS"]

# --- Initialize alert log ---
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

if "user_stocks" not in st.session_state:
    st.session_state.user_stocks = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

if "predicted_today" not in st.session_state:
    st.session_state.predicted_today = set()

# --- ML Model Integration ---
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 3])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def predict_stock(stock):
    df = yf.download(stock, start="2021-01-01", interval="1d")
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    scaled_data, scaler = preprocess_data(df)
    X, Y = create_dataset(scaled_data, time_step=60)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, min_delta=0.0001)
    model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test), callbacks=[early_stopping])

    # Save the trained model
    model_path = f"models/{stock}_lstm_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_path, callbacks=[early_stopping])

    predictions = model.predict(X_test)
    Y_test_actual = scaler.inverse_transform(np.concatenate([np.zeros((len(Y_test), 3)), Y_test.reshape(-1, 1), np.zeros((len(Y_test), 1))], axis=1))[:, 3]
    predictions_actual = scaler.inverse_transform(np.concatenate([np.zeros((len(predictions), 3)), predictions.reshape(-1, 1), np.zeros((len(predictions), 1))], axis=1))[:, 3]

    now = datetime.now()
    st.session_state.alert_log.append({
        "Symbol": stock,
        "Price": float(predictions_actual[-1]),
        "Trigger": "MODEL_PREDICT",
        "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
    })
    st.session_state.predicted_today.add(stock)

def run_daily_prediction()

# --- Pre-market price prediction from 8:00 AM to 8:30 AM CST ---
def predict_premarket_prices():
    now = datetime.now()
    if now.hour == 13 and 0 <= now.minute < 30:  # 8:00 to 8:30 AM CST (13:00 to 13:29 UTC)
        for stock in st.session_state.user_stocks:
            model_path = f"models/{stock}_lstm_model.h5"
            if not os.path.exists(model_path):
                continue

            model = load_model(model_path)
            premarket_data = yf.download(stock, period="1d", interval="1m", prepost=True)
            premarket_data = premarket_data.between_time("08:00", "08:30")

            if premarket_data.empty or len(premarket_data) < 60:
                continue

            df = premarket_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df)
            scaled_data = scaler.transform(df)
            X_live = np.array([scaled_data])

            prediction = model.predict(X_live)
            predicted_price = scaler.inverse_transform(
                np.concatenate([np.zeros((1, 3)), prediction.reshape(-1, 1), np.zeros((1, 1))], axis=1)
            )[:, 3][0]

            st.session_state.alert_log.append({
                "Symbol": stock,
                "Price": float(predicted_price),
                "Trigger": "PREMARKET_PREDICT",
                "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
            })

predict_premarket_prices()

# --- Load saved model and predict in real-time ---
def load_and_predict_realtime():
    for stock in st.session_state.user_stocks:
        model_path = f"models/{stock}_lstm_model.h5"
        if not os.path.exists(model_path):
            continue

        model = load_model(model_path)
        live_data = yf.download(stock, period="1d", interval="1m")[-60:]

        if live_data.empty or len(live_data) < 60:
            continue

        df = live_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df)
        scaled_data = scaler.transform(df)
        X_live = np.array([scaled_data])

        prediction = model.predict(X_live)
        predicted_price = scaler.inverse_transform(
            np.concatenate([np.zeros((1, 3)), prediction.reshape(-1, 1), np.zeros((1, 1))], axis=1)
        )[:, 3][0]

        now = datetime.now()
        st.session_state.alert_log.append({
            "Symbol": stock,
            "Price": float(predicted_price),
            "Trigger": "REALTIME_PREDICT",
            "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
        })

load_and_predict_realtime():
    now = datetime.now()
    for stock in st.session_state.user_stocks:
        if stock not in st.session_state.predicted_today:
            predict_stock(stock)

run_daily_prediction()

# Display model prediction results for each monitored stock
st.subheader("📈 Model Prediction Results")
for log in st.session_state.alert_log:
    if log["Trigger"] == "MODEL_PREDICT":
        st.write(f"**{log['Symbol']}** — Predicted Close Price: **${log['Price']:.2f}** on {log['DateTime']}")

# Display metrics per symbol
st.subheader("📊 Model Accuracy Metrics")
metric_df = pd.DataFrame([
    log for log in st.session_state.alert_log
    if log["Trigger"] in ["MODEL_PREDICT", "REALTIME_PREDICT", "PREMARKET_PREDICT"]
])

if not metric_df.empty:
    for symbol in metric_df['Symbol'].unique():
        symbol_df = metric_df[metric_df['Symbol'] == symbol]
        actuals = [get_stock_price(symbol)["Close"].iloc[-1]] * len(symbol_df)
        predictions = symbol_df['Price'].tolist()
        if len(predictions) >= 2:
            mape = mean_absolute_percentage_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            st.markdown(f"**{symbol}** — MAPE: `{mape:.2f}`, RMSE: `{rmse:.2f}`, R²: `{r2:.2f}`")
st.subheader("📈 Model Prediction Results")
for log in st.session_state.alert_log:
    if log["Trigger"] == "MODEL_PREDICT":
        st.write(f"**{log['Symbol']}** — Predicted Close Price: **${log['Price']:.2f}** on {log['DateTime']}")

# --- Dashboard UI ---

st.image("uhv_logo.jpg", width=200)
st.title("University of Houston - Victoria")
st.title("COSC 6380 Capstone Project")
st.title("📈 Stock Trading Dashboard")
st.title("by Matthew Harper")

# --- User input to add stocks ---
new_stock = st.text_input("Add a stock symbol to monitor:")
if new_stock and new_stock.upper() not in st.session_state.user_stocks:
    st.session_state.user_stocks.append(new_stock.upper())

stocks = st.multiselect("Select stocks to monitor:", st.session_state.user_stocks, default=st.session_state.user_stocks)

# --- Threshold inputs for each stock ---
thresholds = {}
for stock in stocks:
    with st.expander(f"Set thresholds for {stock}"):
        buy = st.number_input(f"{stock} - Buy threshold ($)", value=600.00, step=1.0, key=f"buy_{stock}")
        sell = st.number_input(f"{stock} - Sell threshold ($)", value=50.00, step=1.0, key=f"sell_{stock}")
        thresholds[stock] = {"buy": buy, "sell": sell}

# --- Time and Y-axis adjustment ---
st.sidebar.markdown("### Chart Controls")
x_hours = st.sidebar.slider("Select time window (hours):", min_value=1, max_value=24, value=6)
y_min = st.sidebar.number_input("Y-axis min price:", value=0.0)
y_max = st.sidebar.number_input("Y-axis max price:", value=0.0)

# --- Send SMS ---
def send_sms(message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    for recipient in TWILIO_RECIPIENTS:
        client.messages.create(body=message, from_=TWILIO_FROM, to=recipient)
        st.success(f"SMS sent to {recipient}: {message}")

# --- Send Email with CSV Attachment ---
def send_email_with_attachment(file_path):
    msg = EmailMessage()
    msg['Subject'] = 'Daily Trading Alert Summary'
    msg['From'] = EMAIL_SENDER
    msg.set_content('This is your daily stock action report. Please find attached the daily trading alert summary.')

    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    for recipient in EMAIL_RECIPIENTS:
        msg['To'] = recipient
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

# --- Send CSV via SMS (link to download) ---
def send_summary_via_sms(csv_path):
    message = f"Market closed. Download daily trading alerts: {csv_path}"
    send_sms(message)

# --- Check if market is open today ---
def is_market_open_today():
    url = "https://www.nyse.com/market-status"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    status_element = soup.find("div", {"class": "markets-status-indicator__label"})
    if status_element:
        status = status_element.text.strip().lower()
        return "open" in status
    return False

# --- Get current hour in Eastern Time (for market hours check) ---
def is_market_hours():
    now = datetime.utcnow()
    hour_est = (now.hour - 4) % 24  # Convert UTC to EST
    return 9 <= hour_est < 16

# --- Get stock price using Yahoo Finance API ---
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

# --- Display Chart and Check Alerts ---
for stock in stocks:
    data = get_stock_price(stock)
    if not data.empty:
        current_price = data["Close"].iloc[-1]
        st.metric(label=f"Current {stock} Price", value=f"${current_price:.2f}")

        buy_threshold = thresholds[stock]["buy"]
        sell_threshold = thresholds[stock]["sell"]
        now = datetime.now()

        if current_price > buy_threshold:
            advice = f"BUY signal: {stock} is at ${current_price:.2f}, above ${buy_threshold}"
            st.warning(advice)
            send_sms(advice)
            st.session_state.alert_log.append({
                "Symbol": stock,
                "Price": current_price,
                "Trigger": "BUY",
                "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
            })
        elif current_price < sell_threshold:
            advice = f"SELL signal: {stock} is at ${current_price:.2f}, below ${sell_threshold}"
            st.warning(advice)
            send_sms(advice)
            st.session_state.alert_log.append({
                "Symbol": stock,
                "Price": current_price,
                "Trigger": "SELL",
                "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            st.info(f"No trade action for {stock}. Price is within thresholds.")

        # Always display chart for all monitored stocks
        st.subheader(f"📊 {stock} Price Chart")
        recent_data = data.last(f"{x_hours}h") if x_hours < 24 else data
        fig, ax = plt.subplots()
        ax.plot(recent_data.index, recent_data["Close"], label=f"{stock} Actual Price")

        # Overlay prediction if available
        predicted_prices = [log["Price"] for log in st.session_state.alert_log if log["Symbol"] == stock and log["Trigger"] == "MODEL_PREDICT"]
        if predicted_prices:
            predicted_price = predicted_prices[-1]  # Show most recent model prediction
            ax.axhline(predicted_price, color='orange', linestyle='--', label=f"Predicted Close: ${predicted_price:.2f}")

        # Overlay premarket prediction if available
        premarket_predictions = [log["Price"] for log in st.session_state.alert_log if log["Symbol"] == stock and log["Trigger"] == "PREMARKET_PREDICT"]
        if premarket_predictions:
            premarket_price = premarket_predictions[-1]
            ax.axhline(premarket_price, color='red', linestyle='--', label=f"Pre-Market Predict: ${premarket_price:.2f}")

        ax.set_title(f"{stock} - Price Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        if y_min < y_max:
            ax.set_ylim([y_min, y_max])
        ax.legend()
        st.pyplot(fig)
    else:
        st.error(f"Failed to retrieve stock data for {stock}.")

# --- Log predictions every minute and export to CSV ---
def log_and_export_predictions():
    now = datetime.now()
    log_data = [
        log for log in st.session_state.alert_log
        if log["Trigger"] in ["MODEL_PREDICT", "REALTIME_PREDICT", "PREMARKET_PREDICT"]
    ]
    if log_data:
        df = pd.DataFrame(log_data)
        df['Error'] = df.apply(
        lambda row: abs(
            row['Price'] - get_stock_price(row['Symbol'])["Close"].iloc[-1]
        ) if get_stock_price(row['Symbol']).shape[0] > 0 else None,
        axis=1
    )

        # Add performance metrics per stock
        df['MAPE'] = None
        df['RMSE'] = None
        df['R2'] = None
        for symbol in df['Symbol'].unique():
            subset = df[df['Symbol'] == symbol]
            actuals = [get_stock_price(symbol)["Close"].iloc[-1]] * len(subset)
            predictions = subset['Price'].tolist()
            if len(predictions) >= 2:
                df.loc[df['Symbol'] == symbol, 'MAPE'] = mean_absolute_percentage_error(actuals, predictions)
                df.loc[df['Symbol'] == symbol, 'RMSE'] = np.sqrt(mean_squared_error(actuals, predictions))
                df.loc[df['Symbol'] == symbol, 'R2'] = r2_score(actuals, predictions)["Close"].iloc[-1] 
            if get_stock_price(row['Symbol']).shape[0] > 0 else None,axis=1
            
        timestamp = now.strftime("%Y%m%d_%H%M")
        csv_path = f"/tmp/prediction_log_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        send_summary_via_sms(csv_path)
        send_email_with_attachment(csv_path)
        st.success("Prediction log CSV exported and sent via SMS and email.")

log_and_export_predictions()

# --- Export summary CSV when market closes ---




# Refresh every 5 minutes (300000 ms)
st_autorefresh(interval=60000, key="datarefresh")
