import requests
import os
from datetime import datetime
from pytz import timezone
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
from tensorflow.keras.models import load_model
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

# --- Display model prediction results ---
st.subheader("📈 Model Prediction Results")
for log in st.session_state.alert_log:
    if log["Trigger"] == "MODEL_PREDICT":
        st.write(f"**{log['Symbol']}** — Predicted Close Price: **${log['Price']:.2f}** on {log['DateTime']}")

# --- Display metrics per symbol ---
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

# --- Model training mode toggle ---
model_mode = st.radio("Select LSTM model mode:", ["General Model (shared)", "Per-stock Model"])

# --- User selects stock for model training (default = TSLA) ---
selected_stock = st.selectbox("Select stock for LSTM model training:", st.session_state.user_stocks, index=st.session_state.user_stocks.index("TSLA"))
def train_selected_stock_model():
    now = datetime.now(timezone('US/Central'))
    if now.hour == 13 and now.minute == 30:  # 8:30 AM CST = 13:30 UTC
        if is_market_open_today():
            df = yf.download(selected_stock, start="2021-01-01", interval="1d")
            if df.empty:
                st.warning(f"No data found for {selected_stock}.")
                return
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

            def create_dataset(data, step=60):
                X, Y = [], []
                for i in range(len(data) - step - 1):
                    X.append(data[i:(i + step), :])
                    Y.append(data[i + step, 3])
                return np.array(X), np.array(Y)

            X, Y = create_dataset(scaled_data)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            Y_train, Y_test = Y[:split], Y[split:]

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=['mae'])
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20,
                      batch_size=32, callbacks=[EarlyStopping(monitor='val_mae', patience=10)])

            os.makedirs("models", exist_ok=True)
            model_path = "models/pre_trained_lstm_model.h5" if model_mode == "General Model (shared)" else f"models/{selected_stock}_lstm_model.h5"
            model.save(model_path)
            st.success(f"Model for {selected_stock} trained and saved as {'shared model' if model_mode == 'General Model (shared)' else 'individual stock model'} at 8:30 AM CST.")

train_selected_stock_model()

# --- User input to add stocks ---
new_stock = st.text_input("Add a stock symbol to monitor:")
if new_stock and new_stock.upper() not in st.session_state.user_stocks:
    st.session_state.user_stocks.append(new_stock.upper())

stocks = st.multiselect("Select stocks to monitor:", st.session_state.user_stocks, default=st.session_state.user_stocks)

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

# --- Display charts for each stock ---
for stock in stocks:
    data = yf.download(stock, period="1d", interval="1m")
    if not data.empty:
        st.subheader(f"📊 {stock} Price Chart")
        recent_data = data.last(f"{x_hours}h") if x_hours < 24 else data
        fig, ax = plt.subplots()
        ax.plot(recent_data.index, recent_data["Close"], label=f"{stock} Actual Price")

        # Get predicted price
        predictions = [log for log in st.session_state.alert_log if log["Symbol"] == stock and log["Trigger"] == "MODEL_PREDICT"]
        if predictions:
            predicted_price = predictions[-1]["Price"]
            ax.axhline(predicted_price, color='orange', linestyle='--', label=f"Predicted Close: ${predicted_price:.2f}")

        # Get pre-market prediction
        premarket = [log for log in st.session_state.alert_log if log["Symbol"] == stock and log["Trigger"] == "PREMARKET_PREDICT"]
        if premarket:
            premarket_price = premarket[-1]["Price"]
            ax.axhline(premarket_price, color='red', linestyle='--', label=f"Pre-Market Predict: ${premarket_price:.2f}")

        # Display prediction error if available
        if predictions:
            actual_price = data["Close"].iloc[-1]
            prediction_error = abs(actual_price - predicted_price)
            ax.text(recent_data.index[-1], predicted_price, f"Error: ${prediction_error:.2f}", fontsize=9, color='gray')

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title(f"{stock} Price vs Prediction")
        if y_min < y_max:
            ax.set_ylim([y_min, y_max])
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {stock}.")

# --- Predict stock prices every minute during trading hours ---
def predict_realtime_prices():
    now = datetime.now()
    if now.weekday() < 5 and 13 <= now.hour < 21:  # Trading hours 8:00 AM–3:00 PM CST (13:00–21:00 UTC)
        model_path = "models/pre_trained_lstm_model.h5" if model_mode == "General Model (shared)" else f"models/{selected_stock}_lstm_model.h5"
        model = load_model(model_path)
        scaler = MinMaxScaler(feature_range=(0, 1))

        for stock in st.session_state.user_stocks:
            live_data = yf.download(stock, period="1d", interval="1m")[-60:]
            if live_data.empty or len(live_data) < 60:
                continue

            df = live_data[['Open', 'High', 'Low', 'Close', 'Volume']]
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
                "Trigger": "REALTIME_PREDICT",
                "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
            })

predict_realtime_prices()

# --- Export prediction log at market close ---
def export_and_notify_predictions()

# --- Manual send report button ---
if st.button("📤 Send Prediction Report Now"):
    df = pd.DataFrame([
        log for log in st.session_state.alert_log
        if log["Trigger"] in ["MODEL_PREDICT", "REALTIME_PREDICT", "PREMARKET_PREDICT"]
    ])
    if not df.empty:
        now = datetime.now()
        filename = f"prediction_log_manual_{now.strftime('%Y%m%d_%H%M')}.csv"
        csv_path = f"/tmp/{filename}"
        df.to_csv(csv_path, index=False)
        send_summary_via_sms(csv_path)
        send_email_with_attachment(csv_path)
        st.download_button(
            label="⬇️ Download Prediction CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )
        st.success("Manual prediction report sent and available for download.")
    else:
        st.info("No prediction data available to export."):
    now = datetime.now()
    if now.hour == 20 and now.minute == 0:  # 4:00 PM EST = 3:00 PM CST
        df = pd.DataFrame([
            log for log in st.session_state.alert_log
            if log["Trigger"] in ["MODEL_PREDICT", "REALTIME_PREDICT", "PREMARKET_PREDICT"]
        ])
        if not df.empty:
            filename = f"prediction_log_{now.strftime('%Y%m%d_%H%M')}.csv"
            csv_path = f"/tmp/{filename}"
            df.to_csv(csv_path, index=False)
            send_summary_via_sms(csv_path)
            send_email_with_attachment(csv_path)
            st.success(f"📤 Prediction report sent via SMS and Email: {filename}")

export_and_notify_predictions()

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
