import streamlit as st
import pandas as pd
from scripts.data_collection import fetch_stock_data
from scripts.technical_analysis import calculate_indicators
from scripts.model_training import train_model
from scripts.sentiment_analysis import sentiment_analysis
from datetime import datetime, timedelta
import plotly.graph_objs as go


def create_dashboard():
    st.title("AI Stock Analysis Tool")

    symbol = st.text_input("Enter Stock Symbol", "RELIANCE.NS")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())

    if st.button("Analyze"):
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is not None:
            data = calculate_indicators(data)
            model, accuracy, data = train_model(data)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data["date"], y=data["close"], mode="lines", name="Close Price"
                )
            )
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
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["prediction"] * max(data["close"]),
                    mode="markers",
                    name="Predicted Trend",
                    marker=dict(color="blue", size=5),
                )
            )
            st.plotly_chart(fig)

            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
            news = [
                "Stock rises due to market optimism",
                "Uncertainty clouds future prospects",
            ]
            avg_score, sentiment_rating = sentiment_analysis(news)
            st.write(f"Sentiment Score: {avg_score:.2f} ({sentiment_rating})")
