from textblob import TextBlob
import numpy as np


def sentiment_analysis(news_list):
    sentiment_scores = []
    for news in news_list:
        analysis = TextBlob(news)
        sentiment_scores.append(analysis.sentiment.polarity)
    avg_score = np.mean(sentiment_scores)
    sentiment_rating = (
        "Positive" if avg_score > 0.1 else "Neutral" if avg_score > -0.1 else "Negative"
    )
    return avg_score, sentiment_rating
