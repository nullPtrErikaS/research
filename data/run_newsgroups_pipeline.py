"""Run full embedding pipeline on newsgroups dataset.

Usage:
    python data/run_newsgroups_pipeline.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import parse
from src.processor import vectorize, reduce, cluster

def main():
    print("=" * 70)
    print("NEWSGROUPS DATASET PIPELINE")
    print("=" * 70)
    
    # Setup paths
    data_path = Path(__file__).parent / 'newsgroups_20.csv'
    output_dir = Path(__file__).parent.parent / 'artifacts' / 'newsgroups'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and clean
    print("\n[1/6] Loading data...")
    df = parse.load_corpus(data_path)
    df = parse.normalize_corpus(df, text_col='text')
    
    # 2. Preprocess
    print("\n[2/6] Preprocessing...")
    if parse.NLTK_AVAILABLE:
        df = parse.preprocess_texts(
            df, 
            text_col='cleaned_text',
            config={
                'lowercase': True,
                'remove_stopwords': True,
                'min_word_length': 3,
                'lemmatize': True
            }
        )
        df['processed_text'] = df['preprocessed_text']
    else:
        print("  NLTK not available, skipping advanced preprocessing")
        df['processed_text'] = df['cleaned_text']
    
    # 3. TF-IDF vectorization
    print("\n[3/6] TF-IDF vectorization...")
    tfidf_matrix, vectorizer = vectorize.build_tfidf(
        df,
        text_col='processed_text',
        config={
            'max_features': 2000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': (1, 2)
        },
        output_dir=str(output_dir)
    )
    print(f"  Matrix shape: {tfidf_matrix.shape}")
    
    # 4. Dimensionality reduction (SVD first to 50D)
    print("\n[4/6] Dimensionality reduction...")
    coords_50d = reduce.run_pca(tfidf_matrix, n_components=50, output_dir=str(output_dir))
    
    # Project to 2D for visualization
    coords_2d = reduce.run_pca(coords_50d, n_components=2, output_dir=str(output_dir))
    
    # Optional: t-SNE
    try:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE projection...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        coords_tsne = tsne.fit_transform(coords_50d)
        np.save(output_dir / 'coords_tsne.npy', coords_tsne)
    except Exception as e:
        print(f"  t-SNE skipped: {e}")
    
    # Optional: UMAP
    try:
        import umap as _umap
        print("  Computing UMAP projection...")
        reducer = _umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        coords_umap = reducer.fit_transform(coords_50d)
        np.save(output_dir / 'coords_umap.npy', coords_umap)
    except Exception as e:
        print(f"  UMAP skipped: {e}")
    
    # 5. Clustering
    print("\n[5/6] Clustering...")
    cluster_labels = cluster.run_clustering(
        df,
        coords_50d,
        config={'method': 'kmeans', 'n_clusters': 20},
        output_dir=str(output_dir)
    )
    
    df['cluster'] = cluster_labels if cluster_labels is not None else 0
    n_clusters = len(np.unique(cluster_labels)) if cluster_labels is not None else 1
    print(f"  Found {n_clusters} clusters")
    
    # 6. Save artifacts
    print("\n[6/6] Saving artifacts...")
    
    # Save coordinates
    np.save(output_dir / 'coords.npy', coords_2d)
    print(f"  Saved coords.npy (shape: {coords_2d.shape})")
    
    # Save TF-IDF matrix
    if hasattr(tfidf_matrix, 'toarray'):
        from scipy.sparse import save_npz
        save_npz(output_dir / 'tfidf_matrix.npz', tfidf_matrix)
        print(f"  Saved tfidf_matrix.npz")
    else:
        np.save(output_dir / 'tfidf_matrix.npy', tfidf_matrix)
        print(f"  Saved tfidf_matrix.npy")
    
    # Save cluster labels
    np.save(output_dir / 'cluster_labels.npy', cluster_labels)
    
    # Save processed data with clusters
    df.to_csv(output_dir / 'processed_data_with_clusters.csv', index=False)
    print(f"  Saved processed_data_with_clusters.csv")
    
    # Save doc_ids
    with open(output_dir / 'doc_ids.txt', 'w') as f:
        f.write('\n'.join(df['doc_id']))
    print(f"  Saved doc_ids.txt")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 70)
    
    # Display summary
    print(f"\nDataset summary:")
    print(f"  Documents: {len(df)}")
    if cluster_labels is not None:
        print(f"  Clusters: {len(np.unique(cluster_labels))}")
    if 'category' in df.columns:
        print(f"  Categories: {df['category'].nunique()}")
        print(f"\nTop 5 categories:")
        print(df['category'].value_counts().head())

if __name__ == '__main__':
    main()
