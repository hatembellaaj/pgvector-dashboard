import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def reduce_pca(X, dims=2):
    pca = PCA(n_components=dims)
    return pca.fit_transform(X)


def reduce_tsne(X, dims=2, perplexity=30):
    return TSNE(
        n_components=dims,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca"
    ).fit_transform(X)


def reduce_umap(X, dims=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_components=dims
    )
    return reducer.fit_transform(X)
