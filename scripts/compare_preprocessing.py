"""Compare preprocessing variants on the corpus and record evaluation metrics.

Runs the core pipeline steps (preprocess -> tfidf -> pca/umap/tsne -> cluster)
for a small set of preprocessing configurations. Saves per-variant metrics to
`artifacts/preproc_sweep_results.json`.

Usage: python scripts/compare_preprocessing.py
"""
import json
import time
from pathlib import Path
import sys
import numpy as np

# ensure repo root is on sys.path so `src` package is importable when run from scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processor import (
    load_corpus,
    normalize_corpus,
    preprocess_texts,
    build_tfidf,
    run_pca,
    run_umap,
    run_tsne,
    compute_neighbors,
    run_clustering,
)

BASE = Path(__file__).resolve().parents[1] / 'artifacts'
BASE.mkdir(parents=True, exist_ok=True)


def ensure_svd_dense(X, n_components=50):
    from sklearn.decomposition import TruncatedSVD
    if X is None:
        return None
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1), random_state=42)
            Xred = svd.fit_transform(X)
            return Xred
        except Exception:
            try:
                return X.toarray()
            except Exception:
                return None
    if isinstance(X, np.ndarray):
        if X.shape[1] > n_components:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            return svd.fit_transform(X)
        return X
    try:
        return np.array(X)
    except Exception:
        return None


def trustworthiness_score(X_high, X_low, n_neighbors=5):
    try:
        from sklearn.manifold import trustworthiness
        return float(trustworthiness(X_high, X_low, n_neighbors=n_neighbors))
    except Exception:
        return None


def knn_overlap(A, B, k=10):
    try:
        from sklearn.neighbors import NearestNeighbors
        import numpy as _np
        if A is None or B is None:
            return None
        n = A.shape[0]
        k_use = min(k+1, n)
        nnA = NearestNeighbors(n_neighbors=k_use).fit(A).kneighbors(return_distance=False)
        nnB = NearestNeighbors(n_neighbors=k_use).fit(B).kneighbors(return_distance=False)
        overlaps = []
        for i in range(n):
            setA = set(nnA[i][1:])
            setB = set(nnB[i][1:])
            overlaps.append(len(setA & setB) / max(1, (k_use-1)))
        return float(_np.mean(overlaps))
    except Exception:
        return None


def cluster_separation(coords, labels):
    if coords is None or labels is None:
        return None
    labs = np.unique(labels)
    centroids = {}
    within = {}
    for l in labs:
        idx = np.where(labels == l)[0]
        if len(idx) == 0:
            continue
        pts = coords[idx]
        cent = pts.mean(axis=0)
        dists = np.linalg.norm(pts - cent, axis=1)
        centroids[l] = cent
        within[l] = float(dists.mean())
    cent_list = np.array(list(centroids.values())) if centroids else np.array([])
    if len(cent_list) < 2:
        mean_inter = 0.0
    else:
        from itertools import combinations
        pairs = list(combinations(range(len(cent_list)), 2))
        dists = [np.linalg.norm(cent_list[i]-cent_list[j]) for i,j in pairs]
        mean_inter = float(np.mean(dists))
    mean_within = float(np.mean(list(within.values()))) if within else 0.0
    sep = mean_inter / (mean_within + 1e-12) if (mean_within > 0) else None
    return {'mean_inter_centroid': mean_inter, 'mean_within_cluster': mean_within, 'separation_ratio': sep}


def load_tfidf_from_dir(d):
    try:
        from scipy import sparse
        p = Path(d) / 'tfidf_matrix.npz'
        if p.exists():
            return sparse.load_npz(str(p))
    except Exception:
        pass
    p2 = Path(d) / 'tfidf_matrix.npy'
    if p2.exists():
        return np.load(str(p2))
    return None


def run_variant(name, preproc_cfg):
    out = BASE / f'preproc_{name}'
    # ensure a clean output dir
    out.mkdir(exist_ok=True)

    df = load_corpus('all_guidelines.csv')
    df = normalize_corpus(df)
    if preproc_cfg is None:
        df = preprocess_texts(df)
    else:
        df = preprocess_texts(df, config=preproc_cfg)

    # Build TF-IDF into this variant folder
    X, vect = build_tfidf(df, text_col='preprocessed_text', config=None, output_dir=str(out))

    # PCA / SVD
    coords_pca = run_pca(X, output_dir=str(out))
    coords_tsne = run_tsne(X, output_dir=str(out))
    coords_umap = run_umap(X, output_dir=str(out))

    # Clustering (prefer umap coords)
    cluster_input = coords_umap if coords_umap is not None else coords_pca
    labels = run_clustering(df, cluster_input, output_dir=str(out))

    # Evaluation
    X50 = ensure_svd_dense(X, n_components=50)
    results = {'variant': name, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': {}}

    methods = {'pca': coords_pca, 'umap': coords_umap, 'tsne': coords_tsne}
    for mname, coords in methods.items():
        if coords is None:
            continue
        run = {}
        run['trustworthiness_k5'] = trustworthiness_score(X50, coords, n_neighbors=5) if X50 is not None else None
        run['knn_overlap_k10'] = knn_overlap(X50, coords, k=10) if X50 is not None else None
        run['n_points'] = int(coords.shape[0])
        run['cluster_separation'] = cluster_separation(coords, labels) if labels is not None else None
        results['runs'][mname] = run

    # pairwise overlaps
    available = {k:v for k,v in methods.items() if v is not None}
    keys = list(available.keys())
    pairs = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = keys[i]
            b = keys[j]
            pairs[f'{a}_vs_{b}'] = knn_overlap(available[a], available[b], k=10)
    results['pairwise_knn_overlap'] = pairs

    # save per-variant JSON
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    # define a small set of preprocessing variants to compare
    variants = {
        'default': None,
        'no_stopwords': {'lowercase': True, 'remove_stopwords': False, 'min_word_length': 3, 'lemmatize': True},
        'no_lemmatize': {'lowercase': True, 'remove_stopwords': True, 'min_word_length': 3, 'lemmatize': False},
        'no_lowercase': {'lowercase': False, 'remove_stopwords': True, 'min_word_length': 3, 'lemmatize': True},
        'minlen2': {'lowercase': True, 'remove_stopwords': True, 'min_word_length': 2, 'lemmatize': True},
    }

    all_results = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'variants': {}}
    for name, cfg in variants.items():
        print('Running variant:', name)
        res = run_variant(name, cfg)
        all_results['variants'][name] = res

    out_path = BASE / 'preproc_sweep_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print('Wrote sweep results to', out_path)


if __name__ == '__main__':
    main()
