import streamlit as st
import numpy as np

from app.data_loader import get_db_config, load_embeddings
from app.reducers import reduce_pca, reduce_tsne, reduce_umap
from app.clustering import cluster_kmeans, cluster_hdbscan
from app.visualizations import scatter_2d, scatter_3d
from app.ui_components import aggrid_view

st.set_page_config(page_title="PGVector Dashboard", layout="wide")

st.title("📊 PGVector Dashboard — Analyse des embeddings")

# --------------------------
# Barre latérale
# --------------------------
st.sidebar.header("Source de données")
db_config = get_db_config()
default_table = db_config.get("table_name", "my_embeddings_table")

table_name = st.sidebar.text_input("Table embeddings", value=default_table)
sample_seed = st.sidebar.number_input("Seed échantillonnage", min_value=0, max_value=10_000, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Réduction dimensionnelle")
reducer = st.sidebar.selectbox("Algorithme", ["UMAP", "PCA", "t-SNE"])
dims = st.sidebar.radio("Dimensions", [2, 3], index=0, horizontal=True)

umap_neighbors = st.sidebar.slider("Voisins UMAP", 5, 100, 15) if reducer == "UMAP" else None
umap_min_dist = st.sidebar.slider("Min distance UMAP", 0.0, 0.99, 0.1) if reducer == "UMAP" else None
tsne_perplexity = st.sidebar.slider("Perplexité t-SNE", 5, 100, 30) if reducer == "t-SNE" else None

st.sidebar.markdown("---")
st.sidebar.header("Clustering")
cluster_algo = st.sidebar.selectbox("Algorithme", ["KMeans", "HDBSCAN"])
k = st.sidebar.slider("k clusters", 2, 50, 10) if cluster_algo == "KMeans" else None
hdbscan_min_size = (
    st.sidebar.slider("Taille minimale HDBSCAN", 5, 200, 15)
    if cluster_algo == "HDBSCAN"
    else None
)

# --------------------------
# Charger data
# --------------------------
try:
    df = load_embeddings(table_name=table_name.strip() or None)
except Exception as exc:  # pragma: no cover - handled at runtime
    st.error(f"Impossible de charger les embeddings : {exc}")
    st.stop()

if df.empty:
    st.warning("Aucun embedding trouvé dans la table fournie.")
    st.stop()

max_sample = min(len(df), 5000)

if max_sample <= 1:
    st.sidebar.info("Un seul embedding disponible pour la visualisation.")
    sample_size = 1
else:
    min_sample = 1 if max_sample < 100 else 100
    default_sample = min(max_sample, 1000) if max_sample >= min_sample else max_sample
    step = 50 if max_sample > 50 else 1

    sample_size = st.sidebar.slider(
        "Taille échantillon (visualisation)",
        min_value=min_sample,
        max_value=max_sample,
        value=default_sample,
        step=step,
    )

if sample_size < len(df):
    df_viz = df.sample(sample_size, random_state=sample_seed).reset_index(drop=True)
else:
    df_viz = df.copy().reset_index(drop=True)

X = np.vstack(df_viz["embedding"].values)

# --------------------------
# Réduction dim
# --------------------------
if reducer == "UMAP":
    coords = reduce_umap(X, dims=dims, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
elif reducer == "PCA":
    coords = reduce_pca(X, dims=dims)
else:
    coords = reduce_tsne(X, dims=dims, perplexity=tsne_perplexity)

# --------------------------
# Affecter coords
# --------------------------
df_viz["x"] = coords[:, 0]
df_viz["y"] = coords[:, 1]
if dims == 3:
    df_viz["z"] = coords[:, 2]

# --------------------------
# Clustering
# --------------------------
if cluster_algo == "KMeans":
    df_viz["cluster"] = cluster_kmeans(X, k)
else:
    df_viz["cluster"] = cluster_hdbscan(X, min_size=hdbscan_min_size)

# --------------------------
# Display
# --------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Embeddings chargés", len(df))
col2.metric("Échantillon visualisé", len(df_viz))
col3.metric("Clusters détectés", int(df_viz["cluster"].nunique()))

tab_viz, tab_table, tab_clusters = st.tabs(["Vue 2D/3D", "Tableau interactif", "Résumé clusters"])

with tab_viz:
    st.subheader("Visualisation interactive")
    fig = scatter_3d(df_viz) if dims == 3 else scatter_2d(df_viz)
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    st.subheader("Exploration type AGGrid")
    st.caption("Filtrez, triez et exportez les embeddings pour affiner votre analyse.")
    aggrid_view(df_viz)

with tab_clusters:
    st.subheader("Distribution des clusters")
    cluster_counts = df_viz["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    st.dataframe(cluster_counts.rename_axis("cluster").reset_index(name="compte"))
