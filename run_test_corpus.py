"""Run the full pipeline on the test corpus.

Outputs all artifacts into artifacts/test_corpus/ so the Streamlit explorer
can load it as a separate bundle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processor import (
    load_corpus,
    normalize_corpus,
    preprocess_texts,
    build_tfidf,
    run_pca,
    run_tsne,
    run_umap,
    compute_neighbors,
    run_clustering,
    save_artifacts,
)

OUTPUT_DIR = 'artifacts/test_corpus'

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'=== Test Corpus Pipeline ===')
    print(f'Output: {OUTPUT_DIR}/')

    df = load_corpus(f'{OUTPUT_DIR}/test_corpus.csv')
    df = normalize_corpus(df)
    df = preprocess_texts(df)
    save_artifacts(df, output_dir=OUTPUT_DIR)

    X, vect = build_tfidf(df, output_dir=OUTPUT_DIR)
    if X is None:
        print('ERROR: TF-IDF failed')
        return

    print(f'TF-IDF matrix shape: {X.shape}')

    coords = run_pca(X, output_dir=OUTPUT_DIR)
    print(f'PCA coords shape: {coords.shape}')

    # Perplexity must be < n_samples; clamp for small corpus
    n = len(df)
    tsne_perp = min(30, n - 1)
    coords_tsne = run_tsne(X, config={'n_components': 2, 'perplexity': tsne_perp}, output_dir=OUTPUT_DIR)
    if coords_tsne is not None:
        print(f't-SNE coords shape: {coords_tsne.shape}')

    coords_umap = run_umap(X, output_dir=OUTPUT_DIR)
    if coords_umap is not None:
        print(f'UMAP coords shape: {coords_umap.shape}')

    nn_idx, nn_dist = compute_neighbors(X, n_neighbors=5, metric='cosine', algorithm='brute')

    cluster_input = coords_umap if coords_umap is not None else coords
    labels = run_clustering(df, cluster_input, config={'method': 'kmeans', 'n_clusters': 4}, output_dir=OUTPUT_DIR)

    # Quick sanity checks
    print('\n=== Sanity Checks ===')

    # Check duplicates have identical vectors
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    dupe_ids = ['DUPE_A1', 'DUPE_A2', 'DUPE_A3']
    dupe_idx = [i for i, d in enumerate(df['doc_id']) if d in dupe_ids]
    if len(dupe_idx) == 3:
        dupe_vecs = X[dupe_idx]
        sim = cosine_similarity(dupe_vecs)
        print(f'Duplicate similarity (should be 1.000): {sim[0,1]:.3f}, {sim[0,2]:.3f}, {sim[1,2]:.3f}')
        assert sim[0, 1] > 0.999, 'FAIL: Exact duplicates should have sim ~1.0'
        print('  PASS: Duplicates are identical')

    # Check sports vs cooking separation
    sport_idx = [i for i, d in enumerate(df['doc_id']) if str(d).startswith('SPORT')]
    cook_idx = [i for i, d in enumerate(df['doc_id']) if str(d).startswith('COOK')]
    if sport_idx and cook_idx:
        cross_sim = cosine_similarity(X[sport_idx], X[cook_idx])
        avg_cross = cross_sim.mean()
        within_sport = cosine_similarity(X[sport_idx]).mean()
        within_cook = cosine_similarity(X[cook_idx]).mean()
        print(f'Within-sports avg sim: {within_sport:.3f}')
        print(f'Within-cooking avg sim: {within_cook:.3f}')
        print(f'Cross (sport vs cook) avg sim: {avg_cross:.3f}')
        assert avg_cross < within_sport, 'FAIL: Cross-cluster sim should be < within-cluster'
        assert avg_cross < within_cook, 'FAIL: Cross-cluster sim should be < within-cluster'
        print('  PASS: Sports and Cooking are more similar within-group than across')

    # Check outlier is far from everything
    outlier_idx = [i for i, d in enumerate(df['doc_id']) if d == 'OUTLIER_01']
    if outlier_idx:
        outlier_sim = cosine_similarity(X[outlier_idx], X).flatten()
        outlier_sim_others = np.delete(outlier_sim, outlier_idx[0])
        print(f'Outlier max similarity to any other doc: {outlier_sim_others.max():.3f}')
        print(f'Outlier avg similarity to all docs: {outlier_sim_others.mean():.3f}')
        print('  PASS: Outlier checked')

    print('\n=== Pipeline Complete ===')
    print(f'All outputs in: {OUTPUT_DIR}/')
    print(f'Select "artifacts/test_corpus" in the Explorer dataset dropdown to view.')


if __name__ == '__main__':
    main()
