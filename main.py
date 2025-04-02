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

# --- Dashboard UI ---

st.image("uhv_logo.jpg", width=200)
st.title("University of Houston - Victoria")
st.title("COSC 6380 Capstone Project")
st.title("📈 Stock Trading Dashboard")
st.title("by Matthew Harper")


stocks = st.multiselect("Select stocks to monitor:", ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "PLTR", "NVDA"], default=["TSLA"])

# --- Threshold inputs for each stock ---
thresholds = {}
for stock in stocks:
    with st.expander(f"Set thresholds for {stock}"):
        buy = st.number_input(f"{stock} - Buy threshold ($)", value=305.00, step=1.0, key=f"buy_{stock}")
        sell = st.number_input(f"{stock} - Sell threshold ($)", value=250.00, step=1.0, key=f"sell_{stock}")
        thresholds[stock] = {"buy": buy, "sell": sell}

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
    msg.set_content('This is your daily stock action report.\n\nPlease find attached the daily trading alert summary.')

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

        if stock == selected_stock:
            # Filter chart data for x-axis window
            recent_data = data.last(f"{x_hours}h") if x_hours < 24 else data
            fig, ax = plt.subplots()
            ax.plot(recent_data.index, recent_data["Close"], label=f"{stock} Price")
            ax.set_title(f"{stock} - Price Chart")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            if y_min < y_max:
                ax.set_ylim([y_min, y_max])
            ax.legend()
            st.pyplot(fig)
    else:
        st.error(f"Failed to retrieve stock data for {stock}.")

# --- Export summary CSV when market closes ---
current_time = datetime.utcnow()
if current_time.hour == 20 and current_time.minute == 0:  # 4:00 PM EST
    if st.session_state.alert_log:
        df = pd.DataFrame(st.session_state.alert_log)
        filename = f"daily_alerts_{datetime.now().strftime('%Y%m%d')}.csv"
        csv_path = f"/tmp/{filename}"
        df.to_csv(csv_path, index=False)
        send_summary_via_sms(csv_path)
        send_email_with_attachment(csv_path)
        st.success("Daily alert summary exported, emailed, and link sent via SMS.")
