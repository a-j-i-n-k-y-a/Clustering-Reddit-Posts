from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(corpus, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
