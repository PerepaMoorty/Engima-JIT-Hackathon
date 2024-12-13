import numpy as np

# Sample sentiment dictionary (replace with a more comprehensive one)
SENTIMENT_DICT = {
    "good": 0.8,
    "excellent": 1.0,
    "bad": -0.7,
    "terrible": -1.0,
    "neutral": 0.0,
}

def lorentzian(x, x0=0, gamma=1):
    """
    Lorentzian function for weighting sentiment scores.
    
    Args:
        x (float): Sentiment value for a word.
        x0 (float): Center of the Lorentzian peak (neutral sentiment).
        gamma (float): Width parameter controlling sharpness.

    Returns:
        float: Weighted sentiment score.
    """
    return gamma / (np.pi * ((x - x0)**2 + gamma**2))

def compute_lorentzian_sentiment(text, gamma=1):
    """
    Compute sentiment score for a given text using the Lorentzian function.
    
    Args:
        text (str): Input text or sentence.
        gamma (float): Width parameter for Lorentzian weighting.

    Returns:
        float, str: Average sentiment score and sentiment rating.
    """
    words = text.lower().split()  # Tokenize the text
    scores = []
    for word in words:
        if word in SENTIMENT_DICT:  # Look up the word's sentiment
            sentiment_value = SENTIMENT_DICT[word]
            weighted_score = lorentzian(sentiment_value, x0=0, gamma=gamma)
            scores.append(weighted_score)
    # Average the weighted scores for the final sentiment
    if scores:
        avg_score = np.mean(scores)
        sentiment_rating = (
            "Positive" if avg_score > 0.1 else "Neutral" if avg_score > -0.1 else "Negative"
        )
        return avg_score, sentiment_rating
    else:
        return 0.0, "Neutral"

def sentiment_analysis(news_list):
    """
    Analyze sentiment for a list of news articles using the Lorentzian function.
    
    Args:
        news_list (list): List of news articles as strings.

    Returns:
        float, str: Average sentiment score and sentiment rating.
    """
    avg_scores = []
    for news in news_list:
        score, _ = compute_lorentzian_sentiment(news)
        avg_scores.append(score)
    avg_score = np.mean(avg_scores) if avg_scores else 0.0
    sentiment_rating = (
        "Positive" if avg_score > 0.1 else "Neutral" if avg_score > -0.1 else "Negative"
    )
    return avg_score, sentiment_rating