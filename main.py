from config import *
from data.load_data import load_reddit_data
from preprocessing.text_cleaning import clean_text
from vectorization.tfidf import tfidf_vectorize
from clustering.kmeans import run_kmeans
from clustering.hierarchical import run_hierarchical
from clustering.dbscan import run_dbscan
from evaluation.metrics import evaluate_clustering
from visualization.pca_plot import plot_clusters

# Load data
df = load_reddit_data(subreddits=["AskReddit", "technology"])

df["text"] = (
    df["title"].fillna("") + " " + df["selftext"].fillna("")
).apply(clean_text)

df = df[df["text"].str.len() > MIN_POST_LENGTH]

# Vectorize
X, vectorizer = tfidf_vectorize(df["text"], MAX_FEATURES)

# Run clustering algorithms
models = {
    "KMeans": run_kmeans(X, N_CLUSTERS, RANDOM_STATE),
    "Hierarchical": run_hierarchical(X, N_CLUSTERS),
    "DBSCAN": run_dbscan(X)
}

# Compare results
for name, (_, labels) in models.items():
    scores = evaluate_clustering(X, labels)
    print(f"\n{name}")
    print(scores)
    plot_clusters(X, labels, name)
