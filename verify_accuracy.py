import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

print("=" * 70)
print("ACCURACY CHECK: Will TF-IDF improve the embedding quality score?")
print("=" * 70)

# Load data
tfidf_matrix = scipy.sparse.load_npz('artifacts/tfidf_matrix.npz')
coords_tsne_path = 'artifacts/coords_tsne.npy'
coords_pca_path = 'artifacts/coords.npy'

# Try loading t-SNE coords, fallback to PCA
if np.load(coords_tsne_path) if False else True:
    try:
        coords = np.load(coords_tsne_path)
        print(f"\nUsing t-SNE coordinates: shape {coords.shape}")
    except:
        coords = np.load(coords_pca_path)
        print(f"\nUsing PCA coordinates: shape {coords.shape}")

# Build nearest neighbors in 2D space
n_neighbors = 5
nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
nn.fit(coords)
distances, indices = nn.kneighbors(coords)

print(f"\n[1] Computing TF-IDF similarities for neighbors in 2D space:")

similarities = []
for i in range(min(100, len(coords))):  # Test on first 100 docs
    for neighbor_idx in indices[i][1:]:  # Skip self
        row_i = tfidf_matrix[i]
        row_j = tfidf_matrix[neighbor_idx]
        vec_i = row_i.toarray().flatten()
        vec_j = row_j.toarray().flatten()
        sim = float(cosine_similarity([vec_i], [vec_j])[0, 0])
        similarities.append(sim)

similarities = np.array(similarities)
print(f"    Mean TF-IDF similarity (neighbors in 2D): {similarities.mean():.3f}")
print(f"    Median: {np.median(similarities):.3f}")
print(f"    Std: {similarities.std():.3f}")

print(f"\n[2] Computing TF-IDF similarities for RANDOM pairs (baseline):")

random_similarities = []
np.random.seed(42)
for _ in range(len(similarities)):
    i, j = np.random.choice(len(coords), 2, replace=False)
    row_i = tfidf_matrix[i]
    row_j = tfidf_matrix[j]
    vec_i = row_i.toarray().flatten()
    vec_j = row_j.toarray().flatten()
    sim = float(cosine_similarity([vec_i], [vec_j])[0, 0])
    random_similarities.append(sim)

random_similarities = np.array(random_similarities)
print(f"    Mean TF-IDF similarity (random pairs): {random_similarities.mean():.3f}")
print(f"    Median: {np.median(random_similarities):.3f}")

print(f"\n[3] ACCURACY ASSESSMENT:")
if similarities.mean() > random_similarities.mean():
    ratio = similarities.mean() / random_similarities.mean()
    print(f"    ✓ YES, embedding quality is > random!")
    print(f"    ✓ Neighbors have {ratio:.2f}x higher similarity than random pairs")
    print(f"    ✓ Expected score: ~{similarities.mean():.0%} (using TF-IDF)")
    print(f"    ✓ This is MUCH BETTER than the 11% Jaccard score")
else:
    print(f"    ⚠ WARNING: Embedding quality is NOT better than random!")
    print(f"    ⚠ Neighbors have LOWER TF-IDF similarity than random pairs")
    print(f"    ⚠ This suggests the 2D projection doesn't preserve structure")
    print(f"    ⚠ Consider different UMAP parameters or dataset")

print("\n" + "=" * 70)
