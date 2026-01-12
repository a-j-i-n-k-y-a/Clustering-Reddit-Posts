from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters, random_state):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels
