from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    if len(set(labels)) <= 1 or -1 in set(labels):
        return None

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X.toarray(), labels)
    }
