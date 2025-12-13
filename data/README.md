# 20 Newsgroups Dataset

This directory contains the 20 Newsgroups dataset - a classic text classification dataset containing approximately 500 newsgroup documents across 20 different categories.

## Dataset Details

- **Source**: sklearn.datasets.fetch_20newsgroups
- **Size**: 500 documents (subset of the full training set)
- **Categories**: 20 newsgroup topics including:
  - Computer topics (graphics, hardware, mac, windows, etc.)
  - Recreation (autos, motorcycles, baseball, hockey)
  - Science (space, electronics, medicine, cryptography)
  - Politics, religion, and miscellaneous

## Files

- `newsgroups_20.csv` - Raw dataset with text and category columns
- `fetch_newsgroups.py` - Script to download and prepare the dataset
- `run_newsgroups_pipeline.py` - Full pipeline to process and embed the data

## Running the Pipeline

To process the newsgroups dataset and generate embeddings:

```powershell
python data/run_newsgroups_pipeline.py
```

This will:
1. Load and clean the text data
2. Preprocess (tokenize, remove stopwords, lemmatize)
3. Generate TF-IDF vectors
4. Perform dimensionality reduction (PCA, t-SNE, UMAP)
5. Cluster the documents
6. Save all artifacts to `artifacts/newsgroups/`

## Artifacts Generated

The pipeline produces the following outputs in `artifacts/newsgroups/`:

- `tfidf_matrix.npz` - TF-IDF sparse matrix
- `coords.npy` - 2D PCA coordinates
- `coords_tsne.npy` - 2D t-SNE coordinates
- `coords_umap.npy` - 2D UMAP coordinates
- `cluster_labels.npy` - K-means cluster assignments (20 clusters)
- `processed_data_with_clusters.csv` - Full processed dataset with clusters
- `doc_ids.txt` - Document IDs for reference

## Viewing in Streamlit App

To explore the newsgroups embeddings interactively, you'll need to either:

1. **Copy artifacts to main folder** (if app expects default location):
   ```powershell
   cp artifacts/newsgroups/* artifacts/
   ```

2. **Or modify the streamlit app** to point to the newsgroups subfolder

Then run:
```powershell
streamlit run prototype/streamlit_app.py
```

## Analysis Ideas

This dataset is great for:
- **Category separation**: See if computer vs science vs sports topics naturally cluster
- **Topic modeling**: Identify subtopics within categories
- **Outlier detection**: Find documents that don't fit their assigned category
- **Semantic search**: Find similar posts across different newsgroups
