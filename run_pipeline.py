from src.processor import (
    load_corpus,
    normalize_corpus,
    preprocess_texts,
    build_tfidf,
    run_pca,
    plot_scatter,
    plot_comparison,
    plot_heatmap,
    run_tsne,
    run_umap,
    compute_neighbors,
    run_clustering,
    write_summary_report,
    setup_dirs,
    save_artifacts,
)
    import os

if __name__ == '__main__':
    print('Running pipeline: setup -> preprocess -> tfidf -> pca -> plot')
    # ensure expected directories exist
    setup_dirs()
    df = load_corpus('all_guidelines.csv')
    df = normalize_corpus(df)
    df = preprocess_texts(df)
    # persist cleaned/preprocessed artifacts for reproducibility
    save_artifacts(df)
    X, vect = build_tfidf(df)
    if X is None:
        print('TF-IDF step did not run (missing scikit-learn)')
    else:
        print('TF-IDF complete')
    coords = run_pca(X)
    if coords is None:
        print('PCA/SVD step did not run')
    else:
        print('Dimensionality reduction produced coords with shape:', coords.shape)
        # derive default axis limits from SVD for optional reuse
        svd_limits = (coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max())
            # allow selecting distance metric via env var DIST_METRIC (default 'euclidean')
            DIST_METRIC = os.environ.get('DIST_METRIC', 'euclidean')
            print(f"Using distance metric: {DIST_METRIC}")
            plot_scatter(coords, df, filename='svd_scatter.png', title='SVD scatter', axis_limits=svd_limits,
                         label_col='Guideline + Slogan', annotate_top_n=12, metric=DIST_METRIC)

        # try additional embeddings if available
        coords_tsne = run_tsne(X)
        if coords_tsne is not None:
            tsne_limits = (coords_tsne[:,0].min(), coords_tsne[:,0].max(), coords_tsne[:,1].min(), coords_tsne[:,1].max())
                plot_scatter(coords_tsne, df, filename='tsne_scatter.png', title='t-SNE scatter', axis_limits=tsne_limits,
                             label_col='Guideline + Slogan', annotate_top_n=12, metric=DIST_METRIC)
            # also generate a density heatmap for t-SNE
            plot_heatmap(coords_tsne, filename='tsne_heatmap.png')

        coords_umap = run_umap(X)
        if coords_umap is not None:
            umap_limits = (coords_umap[:,0].min(), coords_umap[:,0].max(), coords_umap[:,1].min(), coords_umap[:,1].max())
                plot_scatter(coords_umap, df, filename='umap_scatter.png', title='UMAP scatter', axis_limits=umap_limits,
                             label_col='Guideline + Slogan', annotate_top_n=12, metric=DIST_METRIC)
            # and a density heatmap for UMAP
            plot_heatmap(coords_umap, filename='umap_heatmap.png')

        # side-by-side comparison if both embeddings exist
        if coords_tsne is not None and coords_umap is not None:
            # Use 'per' mode so each subplot uses its own limits, matching the standalone plots
            plot_comparison(
                coords_tsne, coords_umap, df,
                filename='tsne_vs_umap.png', titles=('t-SNE', 'UMAP'),
                mode='per', axis_limits_a=tsne_limits, axis_limits_b=umap_limits
            )

        # compute nearest neighbors on SVD coords (or fallback to TF-IDF reduced)
        nn_idx, nn_dist = compute_neighbors(coords if coords is not None else X, n_neighbors=10)
    nn_idx, nn_dist = compute_neighbors(coords if coords is not None else X, n_neighbors=10, metric=DIST_METRIC,
                       algorithm='brute' if DIST_METRIC == 'cosine' else 'auto')

        # clustering on UMAP coords if available, else SVD coords
        cluster_input = coords_umap if coords_umap is not None else coords
        labels = run_clustering(df, cluster_input)
    # write a tiny markdown report with pointers to outputs
    try:
        write_summary_report(df)
    except Exception as e:
        print('Could not write report:', e)
    print('Pipeline finished')
