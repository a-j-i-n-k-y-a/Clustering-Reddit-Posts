from sklearn.cluster import AgglomerativeClustering

def run_hierarchical(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X.toarray())
    return model, labels
