import pandas as pd
import numpy as np
from src.processor import build_tfidf
from src.processor import run_pca, compute_neighbors


def test_pca_and_neighbors():
    # small synthetic data
    df = pd.DataFrame({'preprocessed_text': ['aa bb cc dd ee', 'aa bb cc', 'xx yy zz', 'xx zz', 'mm nn oo pp']})
    X, vect = build_tfidf(df, text_col='preprocessed_text', config={'max_features': 20, 'min_df':1, 'max_df':1.0, 'ngram_range':(1,1)})
    if X is None:
        return
    coords = run_pca(X, n_components=2, output_dir='artifacts')
    assert coords is not None
    idx, dist = compute_neighbors(coords, n_neighbors=2, output_dir='artifacts')
    assert idx is not None and dist is not None
    assert idx.shape[0] == coords.shape[0]
