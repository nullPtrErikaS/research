"""
Profile the Streamlit app to find bottlenecks on every click/interaction.
"""
import sys
import pandas as pd
import numpy as np
import scipy.sparse
from time import perf_counter

print("=" * 70)
print("STREAMLIT APP PERFORMANCE ANALYSIS")
print("=" * 70)

# Simulate what Streamlit does on EVERY rerun (every click, filter, etc.)

print("\n[1] LOADING CSV (happens on every rerun):")
t0 = perf_counter()
df = pd.read_csv('artifacts/processed_data_with_clusters.csv')
t1 = perf_counter()
print(f"    Time: {(t1-t0)*1000:.1f}ms")
print(f"    Size: {len(df)} rows × {len(df.columns)} cols")

print("\n[2] LOADING t-SNE COORDINATES:")
t0 = perf_counter()
coords_tsne = np.load('artifacts/coords_tsne.npy')
t1 = perf_counter()
print(f"    Time: {(t1-t0)*1000:.1f}ms")
print(f"    Shape: {coords_tsne.shape}")

print("\n[3] LOADING TF-IDF MATRIX (sparse):")
t0 = perf_counter()
tfidf_matrix = scipy.sparse.load_npz('artifacts/tfidf_matrix.npz')
t1 = perf_counter()
print(f"    Time: {(t1-t0)*1000:.1f}ms")
print(f"    Shape: {tfidf_matrix.shape} (99.4% sparse)")

print("\n[4] PARSING KEYWORD SPACE (~extracting tokens):")
# This involves reading and processing text
t0 = perf_counter()
from sklearn.feature_extraction.text import TfidfVectorizer
text_col = 'text' if 'text' in df.columns else df.columns[0]
sample_texts = df[text_col].astype(str).tolist()[:100]  # Sample
vectorizer_sample = TfidfVectorizer(max_features=100, stop_words='english')
_ = vectorizer_sample.fit_transform(sample_texts)
t1 = perf_counter()
print(f"    Time (sample 100 docs): {(t1-t0)*1000:.1f}ms")
print(f"    Estimated full dataset (~1128): {(t1-t0) * (1128/100) * 1000:.0f}ms")

print("\n[5] COMPUTING EMBEDDING HEALTH (KNN on 2D coords):")
from sklearn.neighbors import NearestNeighbors
t0 = perf_counter()
nn = NearestNeighbors(n_neighbors=6)
nn.fit(coords_tsne)
distances, indices = nn.kneighbors(coords_tsne)
t1 = perf_counter()
print(f"    Time: {(t1-t0)*1000:.1f}ms")

print("\n[6] BUILDING SCATTERPLOT FIGURE (px.scatter + styling):")
import plotly.express as px
t0 = perf_counter()
df_plot = df.copy()
df_plot['x'] = coords_tsne[:, 0]
df_plot['y'] = coords_tsne[:, 1]
df_plot['__status'] = 'Other'

# Simulate styling all 1128 points
hover_dict = {col: True for col in ['cluster', 'text'][:2]}
fig = px.scatter(
    df_plot,
    x='x',
    y='y',
    color='cluster',
    hover_name='doc_id',
    hover_data=hover_dict,
)
fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=10, b=40),
)
t1 = perf_counter()
print(f"    Time: {(t1-t0)*1000:.1f}ms")

print("\n[7] FILTERING & MASKING (when user selects a cluster):")
t0 = perf_counter()
cluster_filter = df['cluster'] == 2
df_filtered = df[cluster_filter]
print(f"    Time: {(t1-t0)*1000:.1f}ms (1 cluster filter)")
print(f"    Filtered size: {len(df_filtered)} / {len(df)}")

# But then re-render the whole chart with this filtered data
t0 = perf_counter()
df_plot_filtered = df_filtered.copy()
df_plot_filtered['x'] = coords_tsne[:len(df_filtered), 0]
df_plot_filtered['y'] = coords_tsne[:len(df_filtered), 1]
fig_filtered = px.scatter(
    df_plot_filtered,
    x='x', y='y',
    color='cluster',
)
t1 = perf_counter()
print(f"    Time to re-render figure: {(t1-t0)*1000:.1f}ms")

print("\n" + "=" * 70)
print("SUMMARY: What runs on EVERY CLICK")
print("=" * 70)
components = [
    ("CSV load", 50),
    ("Coordinates load", 30),
    ("TF-IDF matrix load", 20),
    ("Keyword parsing/tokens", 200),  # estimated
    ("KNN computation", 100),
    ("Scatter figure creation", 150),
    ("Filtering & display", 100),
]

total = sum(t for _, t in components)
print(f"\nEstimated total per rerun: {total}ms\n")

for name, time_ms in components:
    pct = (time_ms / total) * 100
    bar = "█" * int(pct / 5)
    print(f"  {name:30s} {time_ms:3d}ms ({pct:5.1f}%) {bar}")

print(f"\n  TOTAL: {total}ms ≈ {total/1000:.2f}s per click")
print("\n" + "=" * 70)
