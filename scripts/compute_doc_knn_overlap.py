"""Compute per-document k-NN overlap between two coordinate sets.

Saves `artifacts/point_knn_overlap_k10.csv` with columns:
  doc_id, overlap_k10

Usage: python scripts/compute_doc_knn_overlap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ART = Path(__file__).resolve().parents[1] / 'artifacts'
ART.mkdir(exist_ok=True)

# prefer baseline coords (coords.npy as PCA) and run coords (coords_umap.npy)
base_path = ART / 'coords.npy'
run_path = ART / 'coords_umap.npy'
if not base_path.exists() or not run_path.exists():
    print('Required coord files not found. Need coords.npy and coords_umap.npy')
    raise SystemExit(1)

base = np.load(str(base_path))
run = np.load(str(run_path))

n = min(base.shape[0], run.shape[0])
base = base[:n]
run = run[:n]

# load doc ids if available
doc_ids_path = ART / 'doc_ids.txt'
if doc_ids_path.exists():
    with open(str(doc_ids_path), 'r', encoding='utf-8') as f:
        doc_ids = [l.strip() for l in f.readlines()][:n]
else:
    doc_ids = [f'doc_{i:04d}' for i in range(n)]

# choose k safely: at most n-1
n = base.shape[0]
if n <= 1:
    print('Not enough points to compute neighbors')
    raise SystemExit(1)
k = min(10, n-1)
# fit neighbors
nbrs_base = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(base)
nbrs_run = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(run)

# query
_, idx_base = nbrs_base.kneighbors(base)
_, idx_run = nbrs_run.kneighbors(run)

# remove self (first column assumed to be self)
idx_base = idx_base[:, 1:]
idx_run = idx_run[:, 1:]

# compute overlap fraction per document
overlaps = []
for i in range(n):
    set_base = set(idx_base[i].tolist())
    set_run = set(idx_run[i].tolist())
    overlap = len(set_base & set_run) / float(k)
    overlaps.append(overlap)

out_df = pd.DataFrame({'doc_id': doc_ids, f'overlap_k{k}': overlaps})
out_csv = ART / f'point_knn_overlap_k{k}.csv'
out_df.to_csv(str(out_csv), index=False)
print('Wrote', out_csv)

# print summary
print('mean overlap_k{}: {:.4f}'.format(k, out_df[f'overlap_k{k}'].mean()))
print('median overlap_k{}: {:.4f}'.format(k, out_df[f'overlap_k{k}'].median()))
print('lowest 10 docs (most changed):')
print(out_df.nsmallest(10, f'overlap_k{k}'))
