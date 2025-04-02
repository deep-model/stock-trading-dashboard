import requests
import os
from datetime import datetime
from bs4 import BeautifulSoup
from twilio.rest import Client
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Twilio credentials (use environment variables or secret manager in production) ---
TWILIO_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_FROM = "YOUR_TWILIO_PHONE"
TWILIO_RECIPIENTS = ["+19365204521", "+18328393093"]

# --- Dashboard UI ---
st.title("📈 UHV 6380 Capstone Project Stock Trading Dashboard\
Matthew Harper")
stocks = st.multiselect("Select stocks to monitor:", ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "PLTR", "NVDA"], default=["TSLA"])
selected_stock = st.selectbox("Select stock to display chart:", stocks)

# --- Set threshold inputs ---
buy_threshold = st.number_input("Buy threshold ($)", value=305.00, step=1.0)
sell_threshold = st.number_input("Sell threshold ($)", value=250.00, step=1.0)

# --- Send SMS ---
def send_sms(message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    for recipient in TWILIO_RECIPIENTS:
        client.messages.create(body=message, from_=TWILIO_FROM, to=recipient)
        st.success(f"SMS sent to {recipient}: {message}")

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
    return 9 <= hour_est < 16  # Market open from 9:30am to 4pm

# --- Get stock price using Yahoo Finance API ---
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    return data

# --- Display Chart ---
data = get_stock_price(selected_stock)
if not data.empty:
    st.line_chart(data["Close"])
    current_price = data["Close"].iloc[-1]
    st.metric(label=f"Current {selected_stock} Price", value=f"${current_price:.2f}")

    # --- Trading Bot Assistant ---
    if current_price > buy_threshold:
        advice = f"BUY signal: {selected_stock} is at ${current_price:.2f}, above ${buy_threshold}"
        st.warning(advice)
        if st.button("Send Buy Alert SMS"):
            send_sms(advice)
    elif current_price < sell_threshold:
        advice = f"SELL signal: {selected_stock} is at ${current_price:.2f}, below ${sell_threshold}"
        st.warning(advice)
        if st.button("Send Sell Alert SMS"):
            send_sms(advice)
    else:
        st.info("No trade action needed. Price is within thresholds.")
else:
    st.error("Failed to retrieve stock data.")
