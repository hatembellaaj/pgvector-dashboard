import streamlit as st
import numpy as np

from app.data_loader import load_embeddings
from app.reducers import reduce_pca, reduce_tsne, reduce_umap
from app.clustering import cluster_kmeans, cluster_hdbscan
from app.visualizations import scatter_2d, scatter_3d
from app.ui_components import aggrid_view

st.set_page_config(page_title="PGVector Dashboard", layout="wide")

st.title("📊 PGVector Dashboard — Analyse des embeddings")

# --------------------------
# Charger data
# --------------------------
df = load_embeddings()
X = np.vstack(df["embedding"].values)

# --------------------------
# Barre latérale
# --------------------------
st.sidebar.header("Paramètres")

reducer = st.sidebar.selectbox("Réduction dimensionnelle", ["UMAP", "PCA", "t-SNE"])
dims = st.sidebar.radio("Dimensions", [2, 3], index=0)

cluster_algo = st.sidebar.selectbox("Clustering", ["KMeans", "HDBSCAN"])

if cluster_algo == "KMeans":
    k = st.sidebar.slider("k clusters", 2, 50, 10)

# --------------------------
# Réduction dim
# --------------------------
if reducer == "UMAP":
    coords = reduce_umap(X, dims=dims)
elif reducer == "PCA":
    coords = reduce_pca(X, dims=dims)
else:
    coords = reduce_tsne(X, dims=dims)

# --------------------------
# Affecter coords
# --------------------------
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]
if dims == 3:
    df["z"] = coords[:, 2]

# --------------------------
# Clustering
# --------------------------
if cluster_algo == "KMeans":
    df["cluster"] = cluster_kmeans(X, k)
else:
    df["cluster"] = cluster_hdbscan(X)

# --------------------------
# Display
# --------------------------
st.subheader("Visualisation")

if dims == 2:
    fig = scatter_2d(df)
else:
    fig = scatter_3d(df)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Tableau interactif")
aggrid_view(df)
