import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
import warnings

# Suppress warnings from the language model
warnings.filterwarnings(action="ignore", category=UserWarning)

# Setup spaCy and spacytextblob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Load the dataset using a relative path
dataframe = pd.read_csv("amazon_product_reviews.csv")
dataframe.dropna(subset=['reviews.text'], inplace=True)  # Clean missing values
reviews_data = dataframe['reviews.text']  # Select the review text column

def preprocess_text(text):
    """
    Preprocess text by removing stopwords and performing basic cleaning."""
    doc = nlp(text)
    clean_tokens = [token.text.lower().strip() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(clean_tokens)

def analyse_sentiment(review_text):
    """
    Analyze the sentiment of a review, classify it, and return its classification and polarity score."""
    doc = nlp(review_text)
    polarity = doc._.blob.polarity
    sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
    return sentiment, polarity

# Example of preprocessing and sentiment analysis on sample reviews
for i in range(5):
    sample_review = reviews_data.iloc[i]  # Select the first review
    clean_review = preprocess_text(sample_review)
    sentiment, polarity = analyse_sentiment(clean_review)
    print(f"Review: {sample_review[:100]}...\nCleaned Review: {clean_review[:100]}...\nSentiment: {sentiment}, Polarity: {polarity:.2f}\n")

# Comparing similarity of two reviews (example)
review1 = nlp(preprocess_text(reviews_data.iloc[0]))
review2 = nlp(preprocess_text(reviews_data.iloc[1]))
similarity = review1.similarity(review2)
print(f"Similarity between two reviews: {similarity:.2f}\n")
