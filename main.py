import requests
import os
from datetime import datetime
from bs4 import BeautifulSoup
from twilio.rest import Client
import streamlit as st
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

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

# --- Dashboard UI ---
st.image("uhv_logo.jpg", width=200)
st.title("University of Houston - Victoria")
st.title("COSC 6380 Capstone Project")
st.title("📈 Stock Trading Dashboard")
st.title("by Matthew Harper")

stocks = st.multiselect("Select stocks to monitor:", ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "PLTR", "NVDA"], default=["TSLA"])

# --- User input to add stocks ---
new_stock = st.text_input("Add a stock symbol to monitor:")
if new_stock and new_stock.upper() not in st.session_state.user_stocks:
    st.session_state.user_stocks.append(new_stock.upper())

stocks = st.multiselect("Select stocks to monitor:", st.session_state.user_stocks, default=st.session_state.user_stocks)

selected_stock = st.selectbox("Select stock to display chart:", stocks)

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
    hour_est = (now.hour - 4) % 24
    return 9 <= hour_est < 16

# --- Get stock price using Yahoo Finance API ---
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

# --- Display Chart and Price with Predicted Overlay ---
import time

# --- Create persistent chart placeholders for each stock ---
plot_placeholders = {ticker: st.empty() for ticker in st.session_state.user_stocks}

while is_market_hours():
    for stock in st.session_state.user_stocks:
        data = get_stock_price(stock)
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            st.metric(label=f"Current {stock} Price", value=f"${current_price:.2f}")

            recent_data = data.last(f"{x_hours}h") if x_hours < 24 else data
            fig, ax = plt.subplots()
            ax.plot(recent_data.index, recent_data["Close"], label=f"{stock} Price")

            if 'trained_model' in st.session_state and 'trained_scaler' in st.session_state:
                recent_scaled = st.session_state['trained_scaler'].transform(
                    recent_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                )
                input_data = np.array([recent_scaled])
                predicted = st.session_state['trained_model'].predict(input_data)
                predicted_price = st.session_state['trained_scaler'].inverse_transform(
                    np.concatenate([np.zeros((1, 3)), predicted.reshape(-1, 1), np.zeros((1, 1))], axis=1)
                )[:, 3][0]
                ax.axhline(predicted_price, color='red', linestyle='--', label='Predicted Price')

            
            try:
                recent_scaled = st.session_state['trained_scaler'].transform(
                    recent_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                )
                input_data = np.array([recent_scaled])
                predicted = st.session_state['trained_model'].predict(input_data)
                predicted_price = st.session_state['trained_scaler'].inverse_transform(
                    np.concatenate([np.zeros((1, 3)), predicted.reshape(-1, 1), np.zeros((1, 1))], axis=1)
                )[:, 3][0]
            except Exception:
                predicted_price = current_price  # fallback
            ax.axhline(predicted_price, color='red', linestyle='--', label='Predicted Price')(f"{stock} - Price Chart")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            if y_min < y_max:
                ax.set_ylim([y_min, y_max])
            ax.legend()
            plot_placeholders[stock].pyplot(fig)
        else:
            st.error(f"Failed to retrieve stock data for {stock}.")
    time.sleep(1)

# --- Export summary CSV when market closes ---
current_time = datetime.utcnow()
if current_time.hour == 20 and current_time.minute == 0:
    if st.session_state.alert_log:
        df = pd.DataFrame(st.session_state.alert_log)
        filename = f"daily_alerts_{datetime.now().strftime('%Y%m%d')}.csv"
        csv_path = f"/tmp/{filename}"
        df.to_csv(csv_path, index=False)
        send_summary_via_sms(csv_path)
        send_email_with_attachment(csv_path)
        st.success("Daily alert summary exported, emailed, and link sent via SMS.")

# --- LSTM Model Execution at 8:30 AM if Market is Open ---
trained_model = None
trained_scaler = None

if datetime.now().hour == 8 and datetime.now().minute == 30:
    if is_market_open_today():
        ticker = 'tsla'
        start_date = '2021-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        interval = '1d'

        data = yf.download(ticker, start=start_date, interval=interval)
        df = pd.DataFrame(data)
        df.to_csv(f'{ticker}.csv', index=True)

        def preprocess_data(df):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
            return scaled_data, scaler

        def create_dataset(data, time_step=1000):
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

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        scaled_data, scaler = preprocess_data(df)
        time_step = 120
        X, Y = create_dataset(scaled_data, time_step)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        num_features = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], num_features)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], num_features)

        model = build_lstm_model((X_train.shape[1], num_features))
        early_stopping = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, min_delta=0.0001)
        model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test), callbacks=[early_stopping])

        trained_model = model
        trained_scaler = scaler
        st.session_state['trained_model'] = trained_model
        st.session_state['trained_scaler'] = trained_scaler

# --- Use Trained Model for Real-Time Predictions Every Minute ---
if is_market_hours() and 'trained_model' in st.session_state and 'trained_scaler' in st.session_state:
    trained_model = st.session_state['trained_model']
    trained_scaler = st.session_state['trained_scaler']
    for ticker in st.session_state.user_stocks:
        live_data = yf.download(ticker, period="1d", interval="1m")[-30:]
        if not live_data.empty:
            recent_data = live_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            scaled = trained_scaler.transform(recent_data)
            input_data = np.array([scaled])
            prediction = trained_model.predict(input_data)
            predicted_price = trained_scaler.inverse_transform(
                np.concatenate([np.zeros((1, 3)), prediction.reshape(-1, 1), np.zeros((1, 1))], axis=1)
            )[:, 3][0]
            actual_price = recent_data['Close'].iloc[-1]
            movement = "UP" if predicted_price > actual_price else "DOWN"
            st.write(f"🔁 Real-Time {ticker} Prediction: {movement} | Predicted: ${predicted_price:.2f} | Actual: ${actual_price:.2f}")

            # --- Trigger alerts only on BUY or SELL based on prediction ---
            now = datetime.now()
            if movement == "UP":
                message = f"BUY ALERT: {ticker} predicted to go UP. Current: ${actual_price:.2f}, Predicted: ${predicted_price:.2f}"
                send_sms(message)
                st.session_state.alert_log.append({
                    "Symbol": ticker,
                    "Price": actual_price,
                    "Prediction": predicted_price,
                    "Trigger": "BUY",
                    "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
                })
            elif movement == "DOWN":
                message = f"SELL ALERT: {ticker} predicted to go DOWN. Current: ${actual_price:.2f}, Predicted: ${predicted_price:.2f}"
                send_sms(message)
                st.session_state.alert_log.append({
                    "Symbol": ticker,
                    "Price": actual_price,
                    "Prediction": predicted_price,
                    "Trigger": "SELL",
                    "DateTime": now.strftime("%Y-%m-%d %H:%M:%S")
                })
if datetime.now().hour == 8 and datetime.now().minute == 30:
    if is_market_open_today():
        ticker = 'tsla'
        start_date = '2021-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        interval = '1d'

        data = yf.download(ticker, start=start_date, interval=interval)
        df = pd.DataFrame(data)
        df.to_csv(f'{ticker}.csv', index=True)

        def preprocess_data(df):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
            return scaled_data, scaler

        def create_dataset(data, time_step=1000):
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

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        scaled_data, scaler = preprocess_data(df)
        time_step = 120
        X, Y = create_dataset(scaled_data, time_step)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        num_features = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], num_features)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], num_features)

        model = build_lstm_model((X_train.shape[1], num_features))
        early_stopping = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, min_delta=0.0001)
        model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test), callbacks=[early_stopping])

        scaler.fit(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        tickers = st.session_state.user_stocks

        def get_live_data(ticker, lookback=30):
            data = yf.download(ticker, period="1d", interval="1m")[-lookback:]
            df = pd.DataFrame(data)[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df

        def preprocess_live_data(live_df, scaler):
            if live_df.empty:
                return None
            scaled_data = scaler.transform(live_df)
            return np.array([scaled_data])

        for ticker in tickers:
            live_df = get_live_data(ticker)
            live_input = preprocess_live_data(live_df, scaler)
            if live_input is None:
                continue
            prediction = model.predict(live_input)
            predicted_price = scaler.inverse_transform(
                np.concatenate([np.zeros((1, 3)), prediction.reshape(-1, 1), np.zeros((1, 1))], axis=1)
            )[:, 3][0]
            actual_price = live_df.iloc[-1]['Close'].item()
            movement = "UP" if predicted_price > actual_price else "DOWN"
            st.info(f"{ticker} prediction at 8:30 AM: {movement} (${predicted_price:.2f}) vs actual ${actual_price:.2f}")
