"""Evaluation script for embeddings.

Computes:
- TruncatedSVD pre-reduction to 50 dims (if TF-IDF matrix available)
- trustworthiness for each available embedding (sklearn.manifold.trustworthiness)
- mean k-NN overlap between embeddings and the pre-reduced high-dim representation
- cluster separation metrics (mean inter-centroid, mean within-cluster, separation ratio)

Writes results to `artifacts/metrics.json` and appends a short summary to `artifacts/report.md`.

Usage: python scripts/eval_metrics.py
"""
import os
import json
import time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'artifacts'
BASE.mkdir(parents=True, exist_ok=True)

TFIDF_PATH = BASE / 'tfidf_matrix.npz'
TFIDF_DENSE = BASE / 'tfidf_matrix.npy'
VECT_PATH = BASE / 'tfidf_vectorizer.pkl'
COORDS_PCA = BASE / 'coords.npy'
COORDS_UMAP = BASE / 'coords_umap.npy'
COORDS_TSNE = BASE / 'coords_tsne.npy'
LABELS = BASE / 'cluster_labels.npy'
OUT_JSON = BASE / 'metrics.json'
OUT_JSON_PER_RUN = BASE / f'metrics-{time.strftime("%Y%m%dT%H%M%S")}.json'
OUT_LOG_CSV = BASE / 'metrics_log.csv'
REPORT = BASE / 'report.md'

# Helpers

def load_tfidf():
    try:
        from scipy import sparse
        if TFIDF_PATH.exists():
            X = sparse.load_npz(str(TFIDF_PATH))
            return X
    except Exception:
        pass
    if TFIDF_DENSE.exists():
        return np.load(str(TFIDF_DENSE))
    return None


def ensure_svd_dense(X, n_components=50):
    from sklearn.decomposition import TruncatedSVD
    # If X is sparse, reduce to n_components
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
        # If already dense but high dim, try PCA-like reduction
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


def cluster_separation(coords, labels):
    import numpy as _np
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


if __name__ == '__main__':
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    results = {'timestamp': ts, 'runs': {}}

    X = load_tfidf()
    X50 = ensure_svd_dense(X, n_components=50)
    if X is None:
        print('No TF-IDF matrix found; some metrics will be skipped.')

    coords_pca = None
    coords_umap = None
    coords_tsne = None
    if COORDS_PCA.exists():
        coords_pca = np.load(str(COORDS_PCA))
    if COORDS_UMAP.exists():
        coords_umap = np.load(str(COORDS_UMAP))
    if COORDS_TSNE.exists():
        coords_tsne = np.load(str(COORDS_TSNE))

    labels = None
    if LABELS.exists():
        labels = np.load(str(LABELS))

    # Trustworthiness / kNN overlap
    methods = {'pca': coords_pca, 'umap': coords_umap, 'tsne': coords_tsne}
    for name, coords in methods.items():
        if coords is None:
            continue
        run = {}
        # trustworthiness against high-dim (X50)
        run['trustworthiness_k5'] = trustworthiness_score(X50, coords, n_neighbors=5) if X50 is not None else None
        run['trustworthiness_k10'] = trustworthiness_score(X50, coords, n_neighbors=10) if X50 is not None else None
        # knn overlap with high-d
        run['knn_overlap_k10'] = knn_overlap(X50, coords, k=10) if X50 is not None else None
        run['n_points'] = int(coords.shape[0])
        # cluster separation on this embedding
        sep = cluster_separation(coords, labels) if labels is not None else None
        run['cluster_separation'] = sep
        results['runs'][name] = run

    # pairwise overlap between embeddings (if multiple present)
    pairs = {}
    available = {k:v for k,v in methods.items() if v is not None}
    keys = list(available.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = keys[i]
            b = keys[j]
            val = knn_overlap(available[a], available[b], k=10)
            pairs[f'{a}_vs_{b}'] = val
    results['pairwise_knn_overlap'] = pairs

    # save JSON
    try:
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        # write a per-run JSON for historical tracking
        try:
            with open(OUT_JSON_PER_RUN, 'w', encoding='utf-8') as fr:
                json.dump(results, fr, indent=2)
        except Exception:
            pass
        print('Wrote metrics to', OUT_JSON)
    except Exception as e:
        print('Could not write metrics.json:', e)

    # append summary to report.md
    try:
        summary_lines = ["\n## Evaluation metrics (generated by scripts/eval_metrics.py)", f"Generated: {ts}", '']
        for m, r in results['runs'].items():
            summary_lines.append(f'### {m.upper()}')
            if r.get('n_points'):
                summary_lines.append(f'- n_points: {r.get("n_points")}')
            tw5 = r.get('trustworthiness_k5')
            tw10 = r.get('trustworthiness_k10')
            ko = r.get('knn_overlap_k10')
            if tw5 is not None:
                summary_lines.append(f'- trustworthiness (k=5): {tw5:.4f}')
            if tw10 is not None:
                summary_lines.append(f'- trustworthiness (k=10): {tw10:.4f}')
            if ko is not None:
                summary_lines.append(f'- mean k-NN overlap vs high-d (k=10): {ko:.4f}')
            sep = r.get('cluster_separation')
            if sep is not None:
                summary_lines.append(f'- mean inter-centroid distance: {sep.get("mean_inter_centroid"):.4f}')
                summary_lines.append(f'- mean within-cluster spread: {sep.get("mean_within_cluster"):.4f}')
                summary_lines.append(f'- separation ratio (inter/within): {sep.get("separation_ratio"):.3f}')
            summary_lines.append('')
        if pairs:
            summary_lines.append('### Pairwise embedding overlap')
            for k,v in pairs.items():
                summary_lines.append(f'- {k}: {v:.4f}')
            summary_lines.append('')

        with open(REPORT, 'a', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print('Appended metrics summary to', REPORT)
    except Exception as e:
        print('Could not append to report.md:', e)

    # append a one-line CSV log for quick history view
    try:
        import csv
        # flatten a few key metrics into a CSV row
        row = {
            'timestamp': ts,
            'n_points': sum(r.get('n_points', 0) for r in results['runs'].values()),
            'pca_trust_k5': results['runs'].get('pca', {}).get('trustworthiness_k5'),
            'umap_trust_k5': results['runs'].get('umap', {}).get('trustworthiness_k5'),
            'tsne_trust_k5': results['runs'].get('tsne', {}).get('trustworthiness_k5'),
        }
        write_header = not OUT_LOG_CSV.exists()
        with open(OUT_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print('Appended metrics summary to', OUT_LOG_CSV)
    except Exception:
        pass

    print('Done.')
