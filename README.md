# Short-Text Embedding Explorer

A modular pipeline and interactive visualization tool for analyzing short-text embeddings. This project enables researchers to compare dimensionality reduction techniques (PCA, t-SNE, UMAP), explore cluster structure, and analyze document neighborhoods across multiple projection methods simultaneously.

## What's Included

- **Pipeline** (`src/processor/`): Modular components for text preprocessing, TF-IDF vectorization, PCA/t-SNE/UMAP dimensionality reduction, k-NN computation, and clustering.
- **Explorer** (`prototype/streamlit_app.py`): Interactive Streamlit app with linked brushing across all three projection views, search/filtering, side-by-side document comparison, selection history/undo, and export functionality.
- **Artifacts**: Reproducible output bundles containing TF-IDF matrices, coordinate arrays, nearest-neighbor indices, cluster labels, and evaluation metrics.
- **Tests**: Regression, reproducibility, stability, and metric validation tests to catch silent drift and ensure consistent results.

## Quick Start

```powershell
# Clone and setup
git clone https://github.com/nullPtrErikaS/research.git
cd research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py

# Launch the explorer
python -m streamlit run prototype/streamlit_app.py

# Run tests
pytest tests/
```

## Key Features

- **Linked brushing**: Selections sync across PCA, t-SNE, and UMAP in real time
- **Search & filter**: By document ID, keyword, or cluster
- **Comparison panel**: Side-by-side documents with similarity metrics and neighbor overlap
- **Selection history**: Undo/redo without losing context
- **Artifact validation**: Automatic checks for row alignment and dimension consistency
- **Export**: Download selected document IDs as JSON

## Artifact Outputs

Under `artifacts/`:

| File | Description |
|------|-------------|
| `tfidf_matrix.npz` | TF-IDF vectors |
| `coords.npy`, `coords_tsne.npy`, `coords_umap.npy` | 2D coordinate arrays |
| `nn_indices.npy`, `nn_distances.npy` | k-NN results |
| `cluster_labels.npy` | Cluster assignments |
| `processed_data_with_clusters.csv` | Cleaned documents with metadata |
| `metrics.json`, `config.json` | Evaluation results and run configuration |

## Documentation

See [`docs/gleicher_frontend.html`](docs/gleicher_frontend.html) for the full project write-up, including methodology, results, lessons learned, and self-evaluation.

