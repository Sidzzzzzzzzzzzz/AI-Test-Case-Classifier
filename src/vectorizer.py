from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample topics (these can be extended or loaded from a config/file)
TOPICS = [
    "emergency stop",
    "digital output signal",
    "PLC communication",
    "ethernet connection",
    "overheat sensor",
    "start button function"
]


def get_relevant_topic(clean_text, topics=TOPICS):
    texts = topics + [clean_text]  # Last is the test case
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between test case and each topic
    similarities = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]

    # Get the topic with highest similarity
    max_index = similarities.argmax()
    return topics[max_index], similarities[max_index]
