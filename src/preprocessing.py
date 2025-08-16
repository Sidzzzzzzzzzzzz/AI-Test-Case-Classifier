import re
import nltk

# Download stopwords once (Streamlit caches this)
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation, numbers, special characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove stopwords
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)
