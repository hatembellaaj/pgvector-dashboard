from sklearn.cluster import KMeans
import hdbscan


def cluster_kmeans(X, k):
    model = KMeans(n_clusters=k)
    return model.fit_predict(X)


def cluster_hdbscan(X, min_size=15):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
    return clusterer.fit_predict(X)
