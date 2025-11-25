from sklearn.cluster import KMeans
import hdbscan


def cluster_kmeans(X, k):
    n_samples = len(X)
    if n_samples == 0:
        return []

    n_clusters = min(k, n_samples)
    model = KMeans(n_clusters=n_clusters)
    return model.fit_predict(X)


def cluster_hdbscan(X, min_size=15):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
    return clusterer.fit_predict(X)
