"""
Diagnostic: Analyze token overlap distribution and embedding health quality.
Helps determine if low health score is a DATA problem vs ALGORITHM problem.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json
import os
import re

# Load data
def load_data():
    """Load processed data with clusters and tokens."""
    candidates = [
        'artifacts/processed_data_with_clusters.csv',
        'artifacts/full_dataset_with_new_id.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} documents from {path}")
            return df
    raise FileNotFoundError("Could not find data file")

def extract_tokens(text):
    """Extract tokens from text (same as streamlit_app.py)."""
    _stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
        'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about',
        'above', 'after', 'again', 'also', 'any', 'because', 'before',
        'between', 'during', 'into', 'through', 'under', 'until', 'up',
        'out', 'over', 'then', 'once', 'here', 'there', 'if', 'while',
        'etc', 'e.g', 'i.e', 'using', 'based', 'often', 'many', 'new',
        'well', 'even', 'like', 'make', 'use', 'one', 'two', 'get',
    }
    if not isinstance(text, str) or not text.strip():
        return []
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return [w for w in words if w not in _stopwords]

def compute_jaccard_overlap(tokens1, tokens2):
    """Jaccard similarity on token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    set1 = set(tokens1) if isinstance(tokens1, list) else set()
    set2 = set(tokens2) if isinstance(tokens2, list) else set()
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_tfidf_similarity(tokens1, tokens2, tfidf_vectorizer=None):
    """Cosine similarity on TF-IDF vectors."""
    try:
        if tfidf_vectorizer is None:
            return None
        t1 = ' '.join(tokens1) if isinstance(tokens1, list) else ''
        t2 = ' '.join(tokens2) if isinstance(tokens2, list) else ''
        if not t1 or not t2:
            return 0.0
        v1 = tfidf_vectorizer.transform([t1])
        v2 = tfidf_vectorizer.transform([t2])
        sim = cosine_similarity(v1, v2)[0, 0]
        return float(sim)
    except:
        return 0.0

def main():
    print("=" * 70)
    print("EMBEDDING HEALTH DIAGNOSTIC")
    print("=" * 70)
    
    df = load_data()
    
    # Generate __tokens from text if not present
    if '__tokens' not in df.columns:
        text_col = 'text' if 'text' in df.columns else None
        if text_col is None:
            print("ERROR: No 'text' column found to extract tokens from")
            return
        print("\nExtracting tokens from text...")
        df['__tokens'] = df[text_col].apply(extract_tokens)
    
    print(f"\n[1] TOKEN STATISTICS")
    print("-" * 70)
    token_counts = []
    for tokens in df['__tokens']:
        if isinstance(tokens, list):
            token_counts.append(len(tokens))
        else:
            token_counts.append(0)
    
    token_counts = np.array(token_counts)
    print(f"Avg tokens/doc: {token_counts.mean():.1f}")
    print(f"Median tokens/doc: {np.median(token_counts):.1f}")
    print(f"Min/Max: {token_counts.min()}/{token_counts.max()}")
    print(f"Docs with 0 tokens: {(token_counts == 0).sum()}")
    
    # All-pairs overlap analysis (sample if large)
    print(f"\n[2] JACCARD OVERLAP DISTRIBUTION (Token Set Similarity)")
    print("-" * 70)
    
    # Sample pairs for speed
    n_pairs = min(5000, len(df) * (len(df) - 1) // 2)
    overlaps_jaccard = []
    
    indices = np.random.choice(len(df), size=min(100, len(df)), replace=False)
    for i in indices:
        for j in indices:
            if i < j:
                t1 = df.iloc[i].get('__tokens', [])
                t2 = df.iloc[j].get('__tokens', [])
                overlap = compute_jaccard_overlap(t1, t2)
                overlaps_jaccard.append(overlap)
    
    overlaps_jaccard = np.array(overlaps_jaccard)
    print(f"Jaccard Similarity Stats (N={len(overlaps_jaccard)} pair samples):")
    print(f"  Mean: {overlaps_jaccard.mean():.3f}")
    print(f"  Median: {np.median(overlaps_jaccard):.3f}")
    print(f"  Std: {overlaps_jaccard.std():.3f}")
    print(f"  Percentiles: 25%={np.percentile(overlaps_jaccard, 25):.3f}, "
          f"50%={np.percentile(overlaps_jaccard, 50):.3f}, "
          f"75%={np.percentile(overlaps_jaccard, 75):.3f}, "
          f"90%={np.percentile(overlaps_jaccard, 90):.3f}")
    print(f"  % pairs with >0% overlap: {(overlaps_jaccard > 0).mean() * 100:.1f}%")
    print(f"  % pairs with >20% overlap: {(overlaps_jaccard > 0.2).mean() * 100:.1f}%")
    
    # Check if coordinates exist for NN analysis
    coords_tsne = None
    if os.path.exists('artifacts/coords_tsne.npy'):
        coords_tsne = np.load('artifacts/coords_tsne.npy')
        print(f"\n[3] NEAREST NEIGHBOR ANALYSIS (2D t-SNE projection)")
        print("-" * 70)
        
        n_neighbors = 5
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(coords_tsne)
        distances, indices_nn = nn.kneighbors(coords_tsne)
        
        nn_overlaps = []
        for i in range(len(df)):
            center_tokens = df.iloc[i].get('__tokens', [])
            if not isinstance(center_tokens, list) or not center_tokens:
                continue
            
            for neighbor_idx in indices_nn[i][1:]:  # Skip self
                neighbor_tokens = df.iloc[neighbor_idx].get('__tokens', [])
                overlap = compute_jaccard_overlap(center_tokens, neighbor_tokens)
                nn_overlaps.append(overlap)
        
        nn_overlaps = np.array(nn_overlaps)
        print(f"K-NN Jaccard Overlap (K={n_neighbors}, N={len(nn_overlaps)} neighbor pairs):")
        print(f"  Mean: {nn_overlaps.mean():.3f} ← Current Health Score")
        print(f"  Median: {np.median(nn_overlaps):.3f}")
        print(f"  Std: {nn_overlaps.std():.3f}")
        print(f"  Percentiles: 25%={np.percentile(nn_overlaps, 25):.3f}, "
              f"75%={np.percentile(nn_overlaps, 75):.3f}")
        print(f"\n  Analysis:")
        print(f"    - Random pairs mean: {overlaps_jaccard.mean():.3f}")
        print(f"    - Spatial NN mean: {nn_overlaps.mean():.3f}")
        if nn_overlaps.mean() > overlaps_jaccard.mean() * 1.5:
            print(f"    ✓ GOOD: Nearby docs are {(nn_overlaps.mean() / overlaps_jaccard.mean()):.1f}x more similar than random")
        else:
            print(f"    ✗ POOR: Projection not preserving semantic distances well")
    
    # Try TF-IDF similarity for comparison
    print(f"\n[4] ALGORITHM COMPARISON: Jaccard vs TF-IDF Cosine")
    print("-" * 70)
    try:
        sample_texts = [' '.join(t) if isinstance(t, list) else '' for t in df['__tokens']]
        sample_texts = [t if t else 'empty' for t in sample_texts]
        
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf.fit(sample_texts)
        
        overlaps_tfidf = []
        for i in indices[:20]:
            for j in indices[:20]:
                if i < j:
                    t1 = df.iloc[i].get('__tokens', [])
                    t2 = df.iloc[j].get('__tokens', [])
                    sim = compute_tfidf_similarity(t1, t2, tfidf)
                    if sim is not None:
                        overlaps_tfidf.append(sim)
        
        overlaps_tfidf = np.array(overlaps_tfidf)
        print(f"TF-IDF Cosine Similarity (sample of {len(overlaps_tfidf)} pairs):")
        print(f"  Mean: {overlaps_tfidf.mean():.3f} vs Jaccard {overlaps_jaccard[:len(overlaps_tfidf)].mean():.3f}")
        
        if overlaps_tfidf.mean() > overlaps_jaccard.mean() * 1.3:
            print(f"\n  Recommendation: Try TF-IDF instead of Jaccard")
            print(f"    → TF-IDF gives {(overlaps_tfidf.mean() / overlaps_jaccard.mean()):.1f}x higher scores")
    except Exception as e:
        print(f"  TF-IDF comparison failed: {e}")
    
    # Save diagnostic report
    print(f"\n[5] VERDICT")
    print("-" * 70)
    
    if nn_overlaps.mean() > 0.25:
        print("✓ GOOD DATA: Spatial neighbors ARE semantically similar")
        print("  → Low health score may be due to projection parameters")
        print("  → Try: Increase t-SNE perplexity, adjust learning_rate")
    elif nn_overlaps.mean() > 0.15:
        print("⚠ MODERATE: Spatial neighbors have moderate semantic overlap")
        print("  → Data might be genuinely scattered/diverse")
        print("  → Try: Switch to TF-IDF similarity, increase K neighbors")
    else:
        print("✗ POOR DATA: Spatial neighbors are semantically distant")
        print("  → Projection is not preserving semantic structure")
        print("  → Options:")
        print("     1. Use higher-dim embeddings (don't project to 2D)")
        print("     2. Adjust projection algorithm (UMAP instead of t-SNE)")
        print("     3. Check if base embeddings are meaningful")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
