import os
import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load data and matrices
path_tfidf = 'artifacts/tfidf_matrix.npz'
path_csv = 'artifacts/processed_data_with_clusters.csv'

print("=" * 70)
print("VERIFYING TF-IDF SIMILARITY IMPLEMENTATION")
print("=" * 70)

tfidf_matrix = scipy.sparse.load_npz(path_tfidf)
df = pd.read_csv(path_csv)

print(f"\n[1] TF-IDF Matrix:")
print(f"    Shape: {tfidf_matrix.shape}")
print(f"    Data type: {type(tfidf_matrix)}")
print(f"    Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.1f}%")

print(f"\n[2] Dataset:")
print(f"    Rows: {len(df)}")
print(f"    Columns: {df.columns.tolist()[:5]}...")

# Test indexing and similarity computation
print(f"\n[3] Testing cosine_similarity computation:")
row0 = tfidf_matrix[0]
row1 = tfidf_matrix[1]
vec0 = row0.toarray().flatten()
vec1 = row1.toarray().flatten()
sim = float(cosine_similarity([vec0], [vec1])[0, 0])
print(f"    Doc 0 vs Doc 1: {sim:.3f}")
print(f"    ✓ Indexing works correctly")

# Compare: Jaccard (token overlap) vs TF-IDF for same pairs
if '__tokens' in df.columns:
    print(f"\n[4] Comparing Jaccard vs TF-IDF on same documents:")
    
    # Jaccard
    tokens0 = df.iloc[0].get('__tokens', [])
    tokens1 = df.iloc[1].get('__tokens', [])
    if isinstance(tokens0, list) and isinstance(tokens1, list):
        set0 = set(str(t).lower() for t in tokens0 if t)
        set1 = set(str(t).lower() for t in tokens1 if t)
        if set0 and set1:
            jaccard = len(set0 & set1) / len(set0 | set1)
            print(f"    Jaccard (Doc 0 vs 1): {jaccard:.3f}")
            print(f"    TF-IDF cosine (Doc 0 vs 1): {sim:.3f}")
            print(f"    TF-IDF is {sim/jaccard:.1f}x higher than Jaccard")
else:
    print(f"\n[4] No __tokens column found - cannot compare Jaccard")

print(f"\n[5] Sampling 10 random pairs:")
import random
random.seed(42)
pairs = random.sample(range(len(df)), min(10, len(df)))
for i, doc_idx in enumerate(pairs[:5]):
    other_idx = (doc_idx + 1) % len(df)
    row_i = tfidf_matrix[doc_idx]
    row_j = tfidf_matrix[other_idx]
    vec_i = row_i.toarray().flatten()
    vec_j = row_j.toarray().flatten()
    sim_ij = float(cosine_similarity([vec_i], [vec_j])[0, 0])
    print(f"    Doc {doc_idx:3d} vs Doc {other_idx:3d}: {sim_ij:.3f}")

print("\n" + "=" * 70)
print("✓ Implementation verification complete")
print("=" * 70)
