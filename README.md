Research - Text Embedding & Clustering Pipeline

This repository contains a small exploratory pipeline for text processing,
TF-IDF vectorization, dimensionality reduction (SVD / t-SNE / UMAP), nearest
neighbours, and clustering. The `artifacts/` directory stores model data and
coordinates; `visualizations/` contains PNG plots.

Quick usage (recommended):

- Run the pipeline + evaluation (this will write artifacts and update report):

  ```powershell
  python -u .\scripts\run_all.py
  ```

- Run only evaluation (fast):

  ```powershell
  python -u .\scripts\run_all.py --skip-pipeline
  ```

- Dry-run (write config snapshot but do not execute pipeline or eval):

  ```powershell
  python -u .\scripts\run_all.py --dry-run
  ```

Key artifacts produced (under `artifacts/`):
- `tfidf_matrix.npz` or `tfidf_matrix.npy` — TF-IDF matrix (sparse/dense)
- `tfidf_vectorizer.pkl`, `vocabulary.pkl` — vectorizer artifacts
- `coords.npy`, `coords_tsne.npy`, `coords_umap.npy` — 2D embeddings
- `nn_indices.npy`, `nn_distances.npy` — nearest-neighbour outputs
- `cluster_labels.npy` — cluster labels
- `processed_data_with_clusters.csv` — processed document table with `cluster`
- `metrics.json` — evaluation metrics computed by `scripts/eval_metrics.py`
- `config.json` — snapshot of run flags produced by `scripts/run_all.py`

Notes and next steps:
- A small package wrapper `src/processor` provides a stable import point. An
  incremental modularization is in progress (preprocess module, etc.).
- To reproduce exact runs, re-use the saved `tfidf_vectorizer.pkl` and the
  coordinate files in `artifacts/`.

If you'd like I can:
- extract sample documents per cluster for inspection,
- continue splitting `parse.py` into modules (preprocess, vectorize, embed, viz),
- add small unit tests for critical functions (TF-IDF, PCA, neighbors).

