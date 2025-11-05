"""Vectorization helpers: TF-IDF building and artifact saving.

This module contains `build_tfidf` moved out of the legacy `parse.py` as an
incremental modularization step.
"""
from pathlib import Path
import pickle
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def build_tfidf(df, text_col='preprocessed_text', config=None, output_dir='artifacts'):
    """Build TF-IDF matrix and save vectorizer + vocabulary.

    Returns: (X, vectorizer) where X is sparse matrix (or None if SKLearn
    is unavailable).
    """
    cfg = config or {'max_features': 5000, 'min_df': 2, 'max_df': 0.8, 'ngram_range': (1, 2)}
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed — cannot build TF-IDF. Install with: pip install scikit-learn")
        return None, None

    vect = TfidfVectorizer(max_features=cfg.get('max_features'),
                           min_df=cfg.get('min_df'),
                           max_df=cfg.get('max_df'),
                           ngram_range=cfg.get('ngram_range', (1, 1)))

    texts = df[text_col].fillna('').astype(str).tolist()
    X = vect.fit_transform(texts)

    Path(output_dir).mkdir(exist_ok=True)
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vect, f)

    # save vocabulary
    with open(f'{output_dir}/vocabulary.pkl', 'wb') as f:
        pickle.dump(vect.vocabulary_, f)

    # save sparse matrix
    try:
        from scipy import sparse
        sparse.save_npz(f'{output_dir}/tfidf_matrix.npz', X)
    except Exception:
        # fallback to dense save (only for small corpora)
        np.save(f'{output_dir}/tfidf_matrix.npy', X.toarray())

    print(f"Built TF-IDF matrix: shape={X.shape}")
    return X, vect
