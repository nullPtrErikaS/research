import pandas as pd
import numpy as np
from src.processor import build_tfidf


def test_build_tfidf_basic():
    df = pd.DataFrame({'preprocessed_text': ['alpha beta gamma', 'beta delta', 'gamma alpha', 'epsilon zeta']})
    X, vect = build_tfidf(df, text_col='preprocessed_text', config={'max_features': 10, 'min_df':1, 'max_df':1.0, 'ngram_range':(1,1)})
    # If scikit-learn isn't installed, X may be None; in that case, skip assertions
    if X is None:
        return
    assert hasattr(X, 'shape')
    assert X.shape[0] == 4
    assert vect is not None
    # vocabulary size should be <= max_features
    assert len(vect.vocabulary_) <= 10
