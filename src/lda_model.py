from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess(texts):
    cleaned = []
    for doc in texts:
        if not isinstance(doc, str):
            doc = str(doc)
        tokens = doc.lower().split()
        words = [w for w in tokens if w.isalpha() and w not in stop_words and w not in string.punctuation]
        cleaned.append(" ".join(words))
    return cleaned

def apply_lda_sklearn(texts, num_topics=5):
    processed = preprocess(texts)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed)

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)

    topic_ids = lda_model.transform(X).argmax(axis=1)

    topics = []
    for idx, topic in enumerate(lda_model.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
        topics.append((idx, top_words))

    return topics, topic_ids


