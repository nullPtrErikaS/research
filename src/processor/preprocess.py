"""Preprocessing module: tokenization, stopword removal, lemmatization.

This is an incremental modularization of the `preprocess_texts` behavior from
`parse.py`. It intentionally contains only what's needed to run preprocessing
so other modules can import it from `src.processor`.
"""
import re
from pathlib import Path

# optional NLTK detection
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

PREPROCESSING_CONFIG = {
    'lowercase': True,
    'remove_stopwords': True,
    'min_word_length': 3,
    'lemmatize': True
}


def ensure_nltk_resources():
    """Download minimal NLTK resources if missing (best-effort)."""
    if not NLTK_AVAILABLE:
        print("NLTK not available (install with: pip install nltk)")
        return
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        try:
            nltk.download('omw-1.4')
        except Exception:
            pass


def preprocess_texts(df, text_col='cleaned_text', config=PREPROCESSING_CONFIG):
    """Tokenize, remove stopwords, filter tokens, lemmatize and return DataFrame.

    Adds `tokens` and `preprocessed_text` columns to the DataFrame.
    """
    if not NLTK_AVAILABLE:
        print("NLTK not installed — skipping advanced preprocessing. Install with: pip install nltk")
        df['tokens'] = df[text_col].fillna('').astype(str).str.split()
        df['preprocessed_text'] = df[text_col].fillna('').astype(str)
        return df

    ensure_nltk_resources()
    stop_words = set(stopwords.words('english')) if config.get('remove_stopwords', True) else set()
    lemmatizer = WordNetLemmatizer() if config.get('lemmatize', True) else None

    tokens_out = []
    for text in df[text_col].fillna('').astype(str):
        if config.get('lowercase', True):
            text_proc = text.lower()
        else:
            text_proc = text
        try:
            toks = word_tokenize(text_proc)
        except LookupError:
            toks = re.findall(r"\b[\w']+\b", text_proc)
        cleaned = []
        for t in toks:
            if not re.match(r"^[A-Za-z]+$", t):
                continue
            if len(t) < config.get('min_word_length', 3):
                continue
            if t in stop_words:
                continue
            if lemmatizer is not None:
                t = lemmatizer.lemmatize(t)
            cleaned.append(t)
        tokens_out.append(cleaned)

    df['tokens'] = tokens_out
    df['preprocessed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    non_empty = (df['preprocessed_text'] != '').sum()
    print(f"Preprocessed {non_empty}/{len(df)} non-empty texts")
    return df
