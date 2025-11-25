# PGVector Dashboard

PGVector Dashboard est une interface Streamlit prête à l'emploi pour explorer vos embeddings PostgreSQL/pgvector : réduction dimensionnelle (UMAP, PCA, t-SNE), clustering (KMeans, HDBSCAN), vues 2D/3D et tableau interactif AGGrid.

## À quoi sert l'application ?
- Visualiser rapidement vos embeddings (2D ou 3D) pour comprendre leur organisation.
- Tester plusieurs méthodes de réduction (UMAP/PCA/t-SNE) et de clustering (KMeans/HDBSCAN).
- Filtrer, trier et exporter les points via un tableau AGGrid.
- Servir de brique "Extension Analyse embeddings" pour RAISE en se connectant au même PostgreSQL/pgvector.

## Installation manuelle
1. **Cloner le repo**
   ```bash
   git clone <repo>
   cd pgvector-dashboard
   ```
2. **Créer un environnement Python**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configurer la connexion PostgreSQL** (au choix)
   - Via variables d'environnement : `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_TABLE` (table contenant les colonnes `id`, `text`, `embedding`).
   - Ou via `./.streamlit/secrets.toml` :
     ```toml
     db_name = "ma_base"
     db_user = "postgres"
     db_password = "motdepasse"
     db_host = "localhost"
     db_port = 5432
     table_name = "my_embeddings_table"
     ```
5. **Lancer le dashboard**
   ```bash
   streamlit run main.py
   ```
   L'application est disponible sur http://localhost:8501.

## Guide d'utilisation
- **Source de données** : saisissez le nom de la table pgvector et un seed d'échantillonnage (pour les grandes tables un échantillon aléatoire est utilisé pour la visualisation).
- **Réduction dimensionnelle** : choisissez UMAP, PCA ou t-SNE, le nombre de dimensions (2D/3D) et ajustez les hyperparamètres (voisins/min_dist pour UMAP, perplexité pour t-SNE).
- **Clustering** : sélectionnez KMeans (avec `k`) ou HDBSCAN (taille minimale).
- **Visualisation** : onglet "Vue 2D/3D" pour explorer les points (couleur par cluster), "Tableau interactif" pour filtrer et exporter via AGGrid, "Résumé clusters" pour la distribution des clusters.

## Docker
Un container est fourni pour un déploiement rapide.

```bash
# Construction et lancement
docker-compose up --build
```

Les variables d'environnement de `docker-compose.yml` alimentent automatiquement la connexion pgvector (pensez à adapter `DB_HOST` si Postgres tourne ailleurs). Le service expose le port `8501`.

## Intégration RAISE (extension "Analyse embeddings")
- Déployez le container ou l'appli Streamlit à côté de votre instance RAISE et pointez `DB_*` vers la base pgvector utilisée par RAISE.
- Dans RAISE, ajoutez une entrée de menu ou un lien vers l'URL du dashboard (ex: `https://raise.exemple.fr/plugins/embeddings`).
- La vue AGGrid permet d'inspecter et exporter les points pour les croiser avec vos données RAISE.

## Dépannage
- **Connexion refusée** : vérifiez `DB_HOST`/`DB_PORT` et que l'extension `pgvector` est installée.
- **Colonnes manquantes** : la table doit exposer `id`, `text`, `embedding` (array vectoriel).
- **t-SNE lent** : réduisez la taille d'échantillon ou utilisez PCA/UMAP.
