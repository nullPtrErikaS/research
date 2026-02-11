# Embedding Run Report
_Generated: 2026-02-08T21:38:14_

## Summary
- Documents: 1491
- Clusters: 8 (column `cluster`)

## Plots
- [x] SVD scatter: visualizations/svd_scatter.png
- [x] t-SNE scatter: visualizations/tsne_scatter.png
- [x] UMAP scatter: visualizations/umap_scatter.png
- [x] t-SNE heatmap: visualizations/tsne_heatmap.png
- [x] UMAP heatmap: visualizations/umap_heatmap.png
- [x] t-SNE vs UMAP: visualizations/tsne_vs_umap.png

## Artifacts
- [x] TF-IDF (sparse): artifacts/tfidf_matrix.npz
- [ ] TF-IDF (dense fallback): artifacts/tfidf_matrix.npy
- [x] Vocabulary: artifacts/vocabulary.pkl
- [x] Vectorizer: artifacts/tfidf_vectorizer.pkl
- [x] SVD/PCA coords: artifacts/coords.npy
- [x] t-SNE coords: artifacts/coords_tsne.npy
- [ ] UMAP coords: artifacts/coords_umap.npy
- [x] NN indices: artifacts/nn_indices.npy
- [x] NN distances: artifacts/nn_distances.npy
- [x] Cluster labels: artifacts/cluster_labels.npy
- [x] Processed data: artifacts/processed_data.csv
- [x] Processed + clusters: artifacts/processed_data_with_clusters.csv

## Configs (snapshot)
- TFIDF_CONFIG: {'max_features': 5000, 'min_df': 2, 'max_df': 0.8, 'ngram_range': (1, 2)}
- TSNE_CONFIG: {'n_components': 2, 'perplexity': 30}
- UMAP_CONFIG: {'n_components': 2, 'n_neighbors': 15}
- CLUSTER_CONFIG: {'method': 'kmeans', 'n_clusters': 8}