import streamlit as st
import pandas as pd
from data_collection import fetch_stock_data
from technical_analysis import calculate_indicators
from model_training import train_model
from sentiment_analysis import sentiment_analysis
from trading_signals import detect_trading_signals
from datetime import datetime, timedelta
import plotly.graph_objs as go
import numpy as np


def create_dashboard():
    st.title("AI Stock Analysis Tool")

    # Input fields
    symbol = st.text_input("Enter Stock Symbol", "RELIANCE.NS")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())

    window = st.slider("Trading Signal Window", min_value=3, max_value=20, value=5)

    # If the 'Analyze' button is clicked
    if st.button("Analyze"):
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is not None:
            # Calculate technical indicators
            data = calculate_indicators(data)
            data = detect_trading_signals(data, window=window)

            # Perform Sentiment Analysis
            news = [
                "Stock rises due to market optimism",
                "Uncertainty clouds future prospects",
            ]
            avg_score, sentiment_rating = sentiment_analysis(news)
            data["sentiment_score"] = avg_score  # Add sentiment_score to data
            st.write(f"Sentiment Score: {avg_score:.2f} ({sentiment_rating})")

            # Train the model and get predictions
            model, accuracy, data, future_dates, future_predictions = train_model(data)

            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Plotting the chart
            fig = go.Figure()

            # Add close price trace
            fig.add_trace(
                go.Scatter(
                    x=data["date"], y=data["close"], mode="lines", name="Close Price"
                )
            )

            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["BB_High"],
                    mode="lines",
                    name="Bollinger High",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["date"], y=data["BB_Low"], mode="lines", name="Bollinger Low"
                )
            )

            # Plot Buy Signals
            buy_signals = data[data["buy_signal"]]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["date"],
                    y=buy_signals["close"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", symbol="triangle-up", size=10),
                )
            )

            # Plot Sell Signals
            sell_signals = data[data["sell_signal"]]
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["date"],
                    y=sell_signals["close"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", symbol="triangle-down", size=10),
                )
            )

            # Combine past and future dates
            x = pd.concat([data["date"], pd.Series(future_dates)], ignore_index=True)

            # Combine past and future predictions
            y = pd.concat([data["prediction"], pd.Series(future_predictions)], ignore_index=True)

            # Add Prediction Line
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines", name="Predicted Close Price", line=dict(color="orange")
                )
            )

            st.plotly_chart(fig)