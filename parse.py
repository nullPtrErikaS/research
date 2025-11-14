"""Text corpus analysis prototype.

Small one-file pipeline for loading guidelines, doing a bit of cleaning,
tokenization, TF-IDF, dimensionality reduction and simple plotting.

This is intentionally compact and pragmatic — helpful for exploratory
work. Outputs are written to ``artifacts/`` and ``visualizations/``.
"""

# Quick overview: load CSV -> clean -> tokenize -> TF-IDF -> reduce -> plot.
# Optional steps: t-SNE/UMAP, nearest-neighbors, clustering. Artifacts and
# plots are written under the project `artifacts/` and `visualizations/`.
import pandas as pd
import re
from pathlib import Path
import pickle
import numpy as np

# optional ML / NLP libs (import guarded so script still runs without them)
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA as SKPCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# optional projection/clustering libs (guarded)
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except Exception:
    TSNE_AVAILABLE = False

try:
    import umap as _umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    NEIGHBORS_AVAILABLE = True
except Exception:
    NEIGHBORS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    CLUSTERING_AVAILABLE = True
except Exception:
    CLUSTERING_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# ============================================================================
# DATA PARSING
# ============================================================================

def load_corpus(filepath):
    """Load a CSV into a DataFrame and add a stable `doc_id` column.

    Args:
        filepath: path to a CSV file readable by ``pandas.read_csv``.

    Returns:
        pd.DataFrame with an added `doc_id` column (doc_0000, doc_0001, ...).
    """
    df = pd.read_csv(filepath)
    # create a simple stable id for each row, useful for plotting/lookup
    df['doc_id'] = [f"doc_{i:04d}" for i in range(len(df))]
    print(f"Loaded {len(df)} docs")
    return df


def basic_clean(text):
    """Lightweight cleaning: strip URLs and collapse whitespace.

    Returns an empty string for missing values to keep downstream code
    consistent (avoids NaNs in text columns).
    """
    if pd.isna(text):
        # normalize missing -> empty string for later tokenization/vectorization
        return ""
    text = str(text)
    # drop obvious URLs
    text = re.sub(r'http\S+', '', text)
    # collapse repeated whitespace/newlines into single spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_corpus(df, text_col='Full Guideline'):
    """Apply cleaning to text column"""
    df['cleaned_text'] = df[text_col].apply(basic_clean)
    non_empty = (df['cleaned_text'] != '').sum()
    print(f"Cleaned {non_empty}/{len(df)} non-empty texts")
    return df


# ============================================================================
# PREPROCESSING 
# ============================================================================

# TODO: implement tokenization with nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# Config for later
PREPROCESSING_CONFIG = {
    'lowercase': True,
    'remove_stopwords': True,
    'min_word_length': 3,
    'lemmatize': True
}

# TODO: TF-IDF vectorization
# from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 2)
}
# TF-IDF -> sparse matrix. tweak max_features/min_df if vocab is wild.

# TODO: Dimensionality reduction
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

PCA_CONFIG = {'n_components': 2}
TSNE_CONFIG = {'n_components': 2, 'perplexity': 30}
UMAP_CONFIG = {'n_components': 2, 'n_neighbors': 15}
CLUSTER_CONFIG = {'method': 'kmeans', 'n_clusters': 8}

# reduce/embed: SVD for sparse -> dense. t-SNE is slow but pretty. UMAP is
# faster and usually fine. play with `perplexity`, `n_neighbors`, `min_dist`.


def _ensure_dense_for_embedding(X, n_components_for_init=50):
    """Return a dense (numpy) array suitable for TSNE/UMAP.

    If X is sparse, use TruncatedSVD to reduce to `n_components_for_init`.
    """
    if X is None:
        return None

    # If input is a sparse matrix (has toarray) we prefer to reduce its
    # dimensionality with TruncatedSVD rather than densify the full matrix
    # (which could be huge). The reduced dense output is friendlier for
    # UMAP/t-SNE which expect in-memory arrays.
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            svd = TruncatedSVD(n_components=min(n_components_for_init, X.shape[1]-1), random_state=42)
            Xred = svd.fit_transform(X)
            return Xred
        except Exception:
            # fallback: try to convert to dense array directly
            try:
                return X.toarray()
            except Exception:
                return None

    # already dense numpy array -> pass through
    if isinstance(X, np.ndarray):
        return X

    # last resort: attempt to coerce to numpy array
    try:
        return np.array(X)
    except Exception:
        return None

def save_artifacts(df, output_dir='artifacts'):
    """Save processed data"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # save doc ids
    with open(f'{output_dir}/doc_ids.txt', 'w') as f:
        for doc_id in df['doc_id']:
            f.write(doc_id + '\n')
    
    # save sample texts
    sample = df['cleaned_text'].head(5)
    with open(f'{output_dir}/sample_cleaned.txt', 'w') as f:
        for text in sample:
            f.write(text + '\n---\n')
    
    # save full dataframe as CSV
    df.to_csv(f'{output_dir}/processed_data.csv', index=False)
    
    print(f"Saved artifacts to {output_dir}/")
    # Note: caller may have added extra columns (tokens, cluster, etc.)
    # so processed_data.csv is a convenient snapshot of current DataFrame state.


def ensure_nltk_resources():
    """Download required NLTK resources if they are missing."""
    if not NLTK_AVAILABLE:
        print("NLTK not available (install with: pip install nltk)")
        return
    # This will try to download a small set of commonly-needed resources.
    # punkt (sentence tokenizer) sometimes requires extra 'punkt_tab' resource
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # some NLTK installs need punkt_tab for language models
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except Exception:
            # ignore if unavailable in this environment
            pass
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
        # for lemmatizer wordnet data
        try:
            nltk.download('omw-1.4')
        except Exception:
            pass


def preprocess_texts(df, text_col='cleaned_text', config=PREPROCESSING_CONFIG):
    """Tokenize, remove stopwords, filter short tokens, and lemmatize.

    Adds a `preprocessed_text` column containing space-joined tokens ready
    for vectorization, and a `tokens` column with the token lists.
    """
    if not NLTK_AVAILABLE:
        print("NLTK not installed — skipping advanced preprocessing. Install with: pip install nltk")
        # fallback: tokenization is a simple whitespace split so downstream
        # vectorizers still have something to work with.
        df['tokens'] = df[text_col].fillna('').astype(str).str.split()
        df['preprocessed_text'] = df[text_col].fillna('').astype(str)
        return df

    ensure_nltk_resources()
    stop_words = set(stopwords.words('english')) if config.get('remove_stopwords', True) else set()
    lemmatizer = WordNetLemmatizer() if config.get('lemmatize', True) else None

    # Tokenize and clean each document. We keep only alphabetic tokens,
    # enforce a minimum length and remove stopwords. Lemmatization is
    # applied if the WordNet lemmatizer is available.
    tokens_out = []

    for text in df[text_col].fillna('').astype(str):
        # optional lowercasing to reduce sparsity
        if config.get('lowercase', True):
            text_proc = text.lower()
        else:
            text_proc = text

        try:
            toks = word_tokenize(text_proc)
        except LookupError:
            # simple fallback if punkt/tokenizers aren't available
            toks = re.findall(r"\b[\w']+\b", text_proc)

        cleaned = []
        for t in toks:
            # keep alphabetic tokens only (drops punctuation/numbers)
            if not re.match(r"^[A-Za-z]+$", t):
                continue

            # enforce minimum token length
            if len(t) < config.get('min_word_length', 3):
                continue

            # basic stopword removal
            if t in stop_words:
                continue

            # optional lemmatization to reduce inflectional forms
            if lemmatizer is not None:
                t = lemmatizer.lemmatize(t)
            cleaned.append(t)

        tokens_out.append(cleaned)

    df['tokens'] = tokens_out
    # space-joined form is convenient for scikit-learn vectorizers
    df['preprocessed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    non_empty = (df['preprocessed_text'] != '').sum()
    print(f"Preprocessed {non_empty}/{len(df)} non-empty texts")
    return df


def build_tfidf(df, text_col='preprocessed_text', config=TFIDF_CONFIG, output_dir='artifacts'):
    """Wrapper that delegates to `src.processor.vectorize.build_tfidf`.

    Kept for backward compatibility with callers that import `build_tfidf`
    from `parse.py`.
    """
    try:
        from src.processor.vectorize import build_tfidf as _build
    except Exception:
        # fallback: use local implementation if new module missing
        print('Could not import src.processor.vectorize; ensure modularization applied correctly')
        return None, None
    return _build(df, text_col=text_col, config=config, output_dir=output_dir)
    # Vectorizer and vocabulary are saved so you can re-use the same
    # mapping later (e.g. to transform new documents without refitting).

# TF-IDF is sparse. reduce with SVD before turning dense for embeddings.


def run_pca(X, n_components=2, output_dir='artifacts'):
    """Wrapper that delegates to `src.processor.reduce.run_pca`.

    Maintains backward compatibility for callers importing `run_pca` from
    `parse.py`.
    """
    try:
        from src.processor.reduce import run_pca as _run_pca
    except Exception:
        print('Could not import src.processor.reduce.run_pca; ensure modularization applied correctly')
        return None
    return _run_pca(X, n_components=n_components, output_dir=output_dir)
    # coords.npy holds the low-dim representation (rows align with input
    # documents). Useful for quick plotting or as input to neighbors/clustering.

# run_pca: quick 2D layout -> artifacts/coords.npy. fallback for neighbors/clusters.


def plot_scatter(coords, df, output_dir='visualizations', title='PCA / SVD scatter', filename=None, axis_limits=None,
                 label_col=None, annotate_top_n=10, metric=None):
    """Compatibility wrapper delegating to `src.processor.viz.plot_scatter`.

    Additional parameters `label_col`, `annotate_top_n`, and `metric` are
    forwarded to the plotting helper to allow human-friendly labels and
    metric visibility in filenames/titles.
    """
    try:
        from src.processor.viz import plot_scatter as _plot
    except Exception:
        print('Could not import src.processor.viz.plot_scatter; ensure modularization applied correctly')
        return None
    return _plot(coords, df, output_dir=output_dir, title=title, filename=filename, axis_limits=axis_limits,
                 label_col=label_col, annotate_top_n=annotate_top_n, metric=metric)


def plot_comparison(coords_a, coords_b, df, output_dir='visualizations', filename='tsne_vs_umap.png', titles=(
    't-SNE', 'UMAP'), mode='shared', axis_limits_a=None, axis_limits_b=None):
    """Compatibility wrapper delegating to `src.processor.viz.plot_comparison`."""
    try:
        from src.processor.viz import plot_comparison as _plot
    except Exception:
        print('Could not import src.processor.viz.plot_comparison; ensure modularization applied correctly')
        return
    return _plot(coords_a, coords_b, df, output_dir=output_dir, filename=filename, titles=titles, mode=mode, axis_limits_a=axis_limits_a, axis_limits_b=axis_limits_b)


def plot_heatmap(coords, output_dir='visualizations', filename='embedding_heatmap.png', bins=200, cmap='viridis'):
    """Compatibility wrapper delegating to `src.processor.viz.plot_heatmap`."""
    try:
        from src.processor.viz import plot_heatmap as _plot
    except Exception:
        print('Could not import src.processor.viz.plot_heatmap; ensure modularization applied correctly')
        return
    return _plot(coords, output_dir=output_dir, filename=filename, bins=bins, cmap=cmap)


def write_summary_report(df, output_dir_art='artifacts', output_dir_vis='visualizations'):
    """Write a minimal markdown report listing produced artifacts/plots.

    Saves to artifacts/report.md
    """
    Path(output_dir_art).mkdir(exist_ok=True)
    vis = Path(output_dir_vis)
    art = Path(output_dir_art)

    def _exists(p):
        return (vis / p).exists() if not str(p).startswith(str(art)) else Path(p).exists()

    lines = []
    lines.append('# Embedding Run Report')
    try:
        from datetime import datetime
        lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    except Exception:
        lines.append("\n")

    # basics
    lines.append('## Summary')
    lines.append(f"- Documents: {len(df)}")
    if 'cluster' in df.columns:
        try:
            import numpy as _np
            n_clusters = len(_np.unique(_np.array(df['cluster'])))
        except Exception:
            n_clusters = len(set(list(df['cluster'])))
        lines.append(f"- Clusters: {n_clusters} (column `cluster`)\n")
    else:
        lines.append("- Clusters: (not computed)\n")

    # plots
    lines.append('## Plots')
    plots = [
        ('SVD scatter', vis / 'svd_scatter.png'),
        ('t-SNE scatter', vis / 'tsne_scatter.png'),
        ('UMAP scatter', vis / 'umap_scatter.png'),
        ('t-SNE heatmap', vis / 'tsne_heatmap.png'),
        ('UMAP heatmap', vis / 'umap_heatmap.png'),
        ('t-SNE vs UMAP', vis / 'tsne_vs_umap.png'),
    ]
    for label, path in plots:
        mark = '[x]' if path.exists() else '[ ]'
        lines.append(f"- {mark} {label}: {path.as_posix()}")
    lines.append("")

    # artifacts
    lines.append('## Artifacts')
    arts = [
        ('TF-IDF (sparse)', art / 'tfidf_matrix.npz'),
        ('TF-IDF (dense fallback)', art / 'tfidf_matrix.npy'),
        ('Vocabulary', art / 'vocabulary.pkl'),
        ('Vectorizer', art / 'tfidf_vectorizer.pkl'),
        ('SVD/PCA coords', art / 'coords.npy'),
        ('t-SNE coords', art / 'coords_tsne.npy'),
        ('UMAP coords', art / 'coords_umap.npy'),
        ('NN indices', art / 'nn_indices.npy'),
        ('NN distances', art / 'nn_distances.npy'),
        ('Cluster labels', art / 'cluster_labels.npy'),
        ('Processed data', art / 'processed_data.csv'),
        ('Processed + clusters', art / 'processed_data_with_clusters.csv'),
    ]
    for label, path in arts:
        mark = '[x]' if path.exists() else '[ ]'
        lines.append(f"- {mark} {label}: {path.as_posix()}")

    # configs snapshot
    lines.append('\n## Configs (snapshot)')
    lines.append(f"- TFIDF_CONFIG: {TFIDF_CONFIG}")
    lines.append(f"- TSNE_CONFIG: {TSNE_CONFIG}")
    lines.append(f"- UMAP_CONFIG: {UMAP_CONFIG}")
    lines.append(f"- CLUSTER_CONFIG: {CLUSTER_CONFIG}")

    report_path = art / 'report.md'
    try:
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"Wrote report to {report_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")

# big datasets -> downsample or set alpha so the plot isn't a black blob.


def run_tsne(X, config=TSNE_CONFIG, output_dir='artifacts'):
    """Compatibility wrapper delegating to `src.processor.embed.run_tsne`."""
    try:
        from src.processor.embed import run_tsne as _run
    except Exception:
        print('Could not import src.processor.embed.run_tsne; ensure modularization applied correctly')
        return None
    return _run(X, config=config, output_dir=output_dir)


def run_umap(X, config=UMAP_CONFIG, output_dir='artifacts'):
    """Compatibility wrapper delegating to `src.processor.embed.run_umap`."""
    try:
        from src.processor.embed import run_umap as _run
    except Exception:
        print('Could not import src.processor.embed.run_umap; ensure modularization applied correctly')
        return None
    return _run(X, config=config, output_dir=output_dir)


def compute_neighbors(X_or_coords, n_neighbors=10, output_dir='artifacts', metric='euclidean', algorithm='auto'):
    """Compatibility wrapper that delegates to `src.processor.neighbors.compute_neighbors`.

    Parameters below are passed through to the implementation in `src.processor.neighbors`.
    """
    try:
        from src.processor.neighbors import compute_neighbors as _compute
    except Exception:
        print('Could not import src.processor.neighbors.compute_neighbors; ensure modularization applied correctly')
        return None, None
    return _compute(X_or_coords, n_neighbors=n_neighbors, output_dir=output_dir, metric=metric, algorithm=algorithm)


def run_clustering(df, X_or_coords, config=CLUSTER_CONFIG, output_dir='artifacts'):
    """Compatibility wrapper that delegates to `src.processor.cluster.run_clustering`."""
    try:
        from src.processor.cluster import run_clustering as _run
    except Exception:
        print('Could not import src.processor.cluster.run_clustering; ensure modularization applied correctly')
        return None
    return _run(df, X_or_coords, config=config, output_dir=output_dir)



# ============================================================================
# REPO STRUCTURE
# ============================================================================

def setup_dirs():
    """Create project directories"""
    dirs = ['data', 'artifacts', 'notebooks', 'scripts', 'visualizations', 'models']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print(f"Created dirs: {', '.join(dirs)}")


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING PROTOTYPE")
    print("=" * 50)
    
    # setup
    setup_dirs()
    
    # load and clean
    df = load_corpus('all_guidelines.csv')
    df = normalize_corpus(df)

    # preprocess
    df = preprocess_texts(df, text_col='cleaned_text')

    # TF-IDF
    X, vect = build_tfidf(df, text_col='preprocessed_text')

    # PCA / SVD coords
    coords = run_pca(X, n_components=2)

    # Plot
    plot_scatter(coords, df, title='Guidelines: SVD scatter')

    # Save final artifacts (now includes tokens & preprocessed)
    save_artifacts(df)
        
    # check what metadata we have
    print(f"\nColumns: {list(df.columns)}")
    
    # save stuff
    save_artifacts(df)
    
    print("\n" + "=" * 50)
    print("DONE")