#!/usr/bin/env python3
"""Diagnose why t-SNE isn't loading in the Streamlit app."""

import os
import sys
import numpy as np

print("\n" + "="*70)
print("DIAGNOSTIC: Why is t-SNE Not Available?")
print("="*70)

# Step 1: Check paths
print("\n1️⃣ CHECKING FILE PATHS")
print("-" * 70)

candidates = [
    'artifacts/coords_tsne.npy',
    './artifacts/coords_tsne.npy',
    'streamlit_app/../artifacts/coords_tsne.npy',
]

for candidate in candidates:
    exists = os.path.exists(candidate)
    status = "✅" if exists else "❌"
    print(f"{status} {candidate:40} {'FOUND' if exists else 'NOT FOUND'}")

# Step 2: Try loading
print("\n2️⃣ ATTEMPTING TO LOAD")
print("-" * 70)

tsne_path = 'artifacts/coords_tsne.npy'
if os.path.exists(tsne_path):
    print(f"✅ File exists at: {os.path.abspath(tsne_path)}")
    try:
        coords_tsne = np.load(tsne_path)
        print(f"✅ Successfully loaded t-SNE")
        print(f"   - Shape: {coords_tsne.shape}")
        print(f"   - Dtype: {coords_tsne.dtype}")
        print(f"   - Memory: {coords_tsne.nbytes / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Failed to load: {type(e).__name__}: {e}")
else:
    print(f"❌ File NOT found at: {os.path.abspath(tsne_path)}")

# Step 3: Check other coordinate files
print("\n3️⃣ OTHER COORDINATE FILES")
print("-" * 70)

coord_files = [
    'artifacts/coords_tsne.npy',
    'artifacts/coords_umap.npy',
    'artifacts/coords.npy',
    'artifacts/tfidf_matrix.npz',
    'artifacts/cluster_labels.npy',
]

for fname in coord_files:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        print(f"✅ {fname:35} ({size_str})")
    else:
        print(f"❌ {fname:35} (NOT FOUND)")

# Step 4: Check if run_pipeline has been executed
print("\n4️⃣ CHECKING IF PIPELINE WAS RUN")
print("-" * 70)

pipeline_outputs = [
    'artifacts/processed_data.csv',
    'artifacts/processed_data_with_clusters.csv',
    'artifacts/coords_tsne.npy',
    'run_pipeline.py',
]

for item in pipeline_outputs:
    if os.path.exists(item):
        if os.path.isfile(item):
            size = os.path.getsize(item)
            print(f"✅ {item:40} ({size/1024:.1f} KB)")
        else:
            print(f"✅ {item:40} (directory)")
    else:
        print(f"❌ {item:40}")

print("\n5️⃣ SOLUTION")
print("-" * 70)

if not os.path.exists('artifacts/coords_tsne.npy'):
    print("""
❌ t-SNE coordinates have NOT been pre-computed yet!

SOLUTION: Run the pipeline to pre-compute t-SNE (one-time operation):
    
    python run_pipeline.py

This will:
  1. Process the newsgroups data
  2. Compute TF-IDF embeddings
  3. Fit t-SNE, UMAP, and PCA projections
  4. Save coordinates to artifacts/

Once complete, the Streamlit app will use pre-computed t-SNE and be
500-1000ms faster per interaction!
""")
else:
    print("""
✅ t-SNE file EXISTS!

If the Streamlit app is still using PCA fallback, the issue might be:
1. File path not being found by load_coordinate_files()
2. try_load() failing silently for another reason
3. Streamlit cache invalidation not working

Check the Streamlit app output for diagnostics:
  - Look for "Coordinate File Paths" expander
  - Look for "Coordinate Loading Status" expander
  - Check for warning messages about PCA fallback
""")

print("\n" + "="*70)
