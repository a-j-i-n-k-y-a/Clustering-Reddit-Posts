from sklearn.cluster import DBSCAN

def run_dbscan(X, eps=0.7, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(X)
    return model, labels
