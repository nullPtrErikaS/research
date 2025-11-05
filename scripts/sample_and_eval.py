"""Sample-and-evaluate script

Draw repeated random samples of documents and measure k-NN overlap
between high-dimensional TF-IDF (pre-reduced) and low-dimensional embeddings
(PCA/UMAP). Saves aggregated results to `artifacts/sample_eval_results.csv`.

Usage:
  python scripts/sample_and_eval.py --sample-size 25 --trials 30

Defaults are conservative; you can lower sample_size/trials for quick runs.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

ART = Path('artifacts')
ART.mkdir(exist_ok=True)
TFIDF_PATH = ART / 'tfidf_matrix.npz'
TFIDF_DENSE = ART / 'tfidf_matrix.npy'
COORDS_PCA = ART / 'coords.npy'
COORDS_UMAP = ART / 'coords_umap.npy'
COORDS_TSNE = ART / 'coords_tsne.npy'
OUT_CSV = ART / 'sample_eval_results.csv'


def load_tfidf():
    try:
        from scipy import sparse
        if TFIDF_PATH.exists():
            return sparse.load_npz(str(TFIDF_PATH))
    except Exception:
        pass
    if TFIDF_DENSE.exists():
        return np.load(str(TFIDF_DENSE))
    return None


def ensure_svd_dense(X, n_components=50):
    if X is None:
        return None
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1), random_state=42)
            return svd.fit_transform(X)
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


def knn_overlap(A, B, k=10):
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
    return float(np.mean(overlaps))


def sample_and_evaluate(sample_size=25, trials=30, k=10, seed=42):
    X = load_tfidf()
    if X is None:
        print('No TF-IDF matrix found; aborting sample eval')
        return None
    X50 = ensure_svd_dense(X, n_components=50)
    coords_pca = np.load(str(COORDS_PCA)) if COORDS_PCA.exists() else None
    coords_umap = np.load(str(COORDS_UMAP)) if COORDS_UMAP.exists() else None
    coords_tsne = np.load(str(COORDS_TSNE)) if COORDS_TSNE.exists() else None

    rng = random.Random(seed)
    n = X50.shape[0]
    results = []
    methods = [('pca', coords_pca), ('umap', coords_umap), ('tsne', coords_tsne)]
    available = [(name, coords) for name, coords in methods if coords is not None]
    if not available:
        print('No low-d coordinates found (UMAP/TSNE/PCA). Run embeddings first.')
        return None

    for t in range(trials):
        idx = sorted(rng.sample(range(n), min(sample_size, n)))
        X50_sub = X50[idx]
        row = {'trial': t, 'sample_size': len(idx)}
        for name, coords in available:
            coords_sub = coords[idx]
            ov = knn_overlap(X50_sub, coords_sub, k=k)
            row[f'{name}_knn_overlap_k{k}'] = ov
        results.append(row)

    # aggregate and save
    import csv
    keys = ['trial', 'sample_size'] + [f'{name}_knn_overlap_k{k}' for name, _ in available]
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('Wrote sample evaluation results to', OUT_CSV)
    return OUT_CSV


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=25)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    sample_and_evaluate(sample_size=args.sample_size, trials=args.trials, k=args.k, seed=args.seed)
