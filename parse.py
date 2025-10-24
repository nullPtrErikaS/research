"""
Text corpus analysis prototype
"""
# quick pipeline: load csv, clean text, tokenize, TF-IDF, reduce, plot.
# can also run t-SNE/UMAP or cluster if you feel like it. outputs go in
# artifacts/ and visualizations/ (messy but useful).
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
    """Load CSV and add stable doc IDs"""
    df = pd.read_csv(filepath)
    df['doc_id'] = [f"doc_{i:04d}" for i in range(len(df))]
    print(f"Loaded {len(df)} docs")
    return df


def basic_clean(text):
    """cleaning - removes URLs, extra whitespace"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)  # urls
    text = re.sub(r'\s+', ' ', text).strip()  # whitespace
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
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            svd = TruncatedSVD(n_components=min(n_components_for_init, X.shape[1]-1), random_state=42)
            Xred = svd.fit_transform(X)
            return Xred
        except Exception:
            try:
                return X.toarray()
            except Exception:
                return None
    if isinstance(X, np.ndarray):
        return X
    # last resort
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


def ensure_nltk_resources():
    """Download required NLTK resources if they are missing."""
    if not NLTK_AVAILABLE:
        print("NLTK not available (install with: pip install nltk)")
        return
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
        # fallback: use cleaned_text as preprocessed_text
        df['tokens'] = df[text_col].fillna('').astype(str).str.split()
        df['preprocessed_text'] = df[text_col].fillna('').astype(str)
        return df

    ensure_nltk_resources()
    stop_words = set(stopwords.words('english')) if config.get('remove_stopwords', True) else set()
    lemmatizer = WordNetLemmatizer() if config.get('lemmatize', True) else None

    # tokenize, drop stopwords, lemmatize if available. makes things less noisy.

    tokens_out = []

    for text in df[text_col].fillna('').astype(str):
        if config.get('lowercase', True):
            text_proc = text.lower()
        else:
            text_proc = text

        try:
            toks = word_tokenize(text_proc)
        except LookupError:
            # fallback simple tokenizer: split on non-word characters
            toks = re.findall(r"\b[\w']+\b", text_proc)
        cleaned = []
        for t in toks:
            # keep alphabetic tokens only
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


def build_tfidf(df, text_col='preprocessed_text', config=TFIDF_CONFIG, output_dir='artifacts'):
    """Build TF-IDF matrix and save vectorizer + vocabulary.

    Returns: (X, vectorizer) where X is sparse matrix
    """
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed — cannot build TF-IDF. Install with: pip install scikit-learn")
        return None, None

    vect = TfidfVectorizer(max_features=config.get('max_features'),
                           min_df=config.get('min_df'),
                           max_df=config.get('max_df'),
                           ngram_range=config.get('ngram_range', (1, 1)))

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

# TF-IDF is sparse. reduce with SVD before turning dense for embeddings.


def run_pca(X, n_components=2, output_dir='artifacts'):
    """Run dimensionality reduction. Use TruncatedSVD if input is sparse.

    Returns coordinates (n_samples, n_components) as numpy array.
    """
    if X is None:
        print("No matrix provided to run_pca")
        return None

    Path(output_dir).mkdir(exist_ok=True)

    try:
        # prefer TruncatedSVD for sparse matrices
        if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            coords = svd.fit_transform(X)
            # save model
            with open(f'{output_dir}/svd_model.pkl', 'wb') as f:
                pickle.dump(svd, f)
        else:
            pca = SKPCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(X)
            with open(f'{output_dir}/pca_model.pkl', 'wb') as f:
                pickle.dump(pca, f)
    except Exception as e:
        print(f"Dimensionality reduction failed: {e}")
        return None

    # coords -> save
    np.save(f'{output_dir}/coords.npy', coords)
    print(f"Saved coords shape={coords.shape} to {output_dir}")
    return coords

# run_pca: quick 2D layout -> artifacts/coords.npy. fallback for neighbors/clusters.


def plot_scatter(coords, df, output_dir='visualizations', title='PCA / SVD scatter', filename=None, axis_limits=None):
    """Make a simple 2D scatter plot and save to output_dir."""
    if coords is None:
        print("No coords to plot")
        return
    Path(output_dir).mkdir(exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed — cannot produce plot. Install with: pip install matplotlib")
        return

    plt.figure(figsize=(8, 6))
    x, y = coords[:, 0], coords[:, 1]
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    if axis_limits is not None:
        xmin, xmax, ymin, ymax = axis_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # optionally label a few points
    for i, doc_id in enumerate(df['doc_id'].head(10)):
        plt.annotate(doc_id, (x[i], y[i]), fontsize=6, alpha=0.8)

    if filename is None:
        outpath = f'{output_dir}/scatter.png'
    else:
        outpath = str(Path(output_dir) / filename)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved scatter to {outpath}")


def plot_comparison(coords_a, coords_b, df, output_dir='visualizations', filename='tsne_vs_umap.png', titles=(
    't-SNE', 'UMAP'), mode='shared', axis_limits_a=None, axis_limits_b=None):
    """Side-by-side scatter comparison with shared axis limits.

    Assumes coords are aligned to the same document ordering.
    """
    if coords_a is None or coords_b is None:
        print("Missing coords for comparison plot")
        return
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed — cannot produce comparison plot")
        return
    Path(output_dir).mkdir(exist_ok=True)

    import matplotlib.pyplot as plt  # ensure local binding
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=(mode=='shared'), sharey=(mode=='shared'))
    axes[0].scatter(coords_a[:, 0], coords_a[:, 1], s=8, alpha=0.7)
    axes[0].set_title(titles[0])
    axes[1].scatter(coords_b[:, 0], coords_b[:, 1], s=8, alpha=0.7)
    axes[1].set_title(titles[1])
    if mode == 'shared':
        xmin = min(coords_a[:, 0].min(), coords_b[:, 0].min())
        xmax = max(coords_a[:, 0].max(), coords_b[:, 0].max())
        ymin = min(coords_a[:, 1].min(), coords_b[:, 1].min())
        ymax = max(coords_a[:, 1].max(), coords_b[:, 1].max())
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    else:
        # per-embedding limits; use provided or compute from data
        if axis_limits_a is None:
            axis_limits_a = (coords_a[:,0].min(), coords_a[:,0].max(), coords_a[:,1].min(), coords_a[:,1].max())
        if axis_limits_b is None:
            axis_limits_b = (coords_b[:,0].min(), coords_b[:,0].max(), coords_b[:,1].min(), coords_b[:,1].max())
        axes[0].set_xlim(axis_limits_a[0], axis_limits_a[1])
        axes[0].set_ylim(axis_limits_a[2], axis_limits_a[3])
        axes[1].set_xlim(axis_limits_b[0], axis_limits_b[1])
        axes[1].set_ylim(axis_limits_b[2], axis_limits_b[3])
    for ax in axes:
        ax.set_xlabel('dim1')
        ax.set_ylabel('dim2')
    fig.tight_layout()
    outpath = str(Path(output_dir) / filename)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {outpath}")


def plot_heatmap(coords, output_dir='visualizations', filename='embedding_heatmap.png', bins=200, cmap='viridis'):
    """Density heatmap from 2D coords using a 2D histogram."""
    if coords is None:
        print("No coords for heatmap")
        return
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed — cannot produce heatmap")
        return
    Path(output_dir).mkdir(exist_ok=True)

    import matplotlib.pyplot as plt  # ensure local binding
    x = coords[:, 0]
    y = coords[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    # transpose H because imshow expects [row, col] -> y, x
    im = ax.imshow(H.T, origin='lower', cmap=cmap,
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect='auto')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_title('density heatmap')
    fig.colorbar(im, ax=ax, shrink=0.8, label='count')
    outpath = str(Path(output_dir) / filename)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {outpath}")


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
    """Run t-SNE on X (or a reduced version) and save coords."""
    # t-SNE: pretty but slow. use for viz, not as features.
    if not TSNE_AVAILABLE:
        print("scikit-learn TSNE not available install scikit-learn>=0.24")
        return None

    Xdense = _ensure_dense_for_embedding(X, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare dense input for t-SNE")
        return None

    cfg = config.copy()
    n_components = cfg.pop('n_components', 2)
    try:
        tsne = TSNE(n_components=n_components, **cfg, random_state=42)
        coords = tsne.fit_transform(Xdense)
    except TypeError:
        # older sklearn versions accept perplexity, learning_rate etc.; fallback
        tsne = TSNE(n_components=n_components, random_state=42)
        coords = tsne.fit_transform(Xdense)

    Path(output_dir).mkdir(exist_ok=True)
    np.save(f'{output_dir}/coords_tsne.npy', coords)
    print(f"Saved t-SNE coords shape={coords.shape} to {output_dir}/coords_tsne.npy")
    return coords


def run_umap(X, config=UMAP_CONFIG, output_dir='artifacts'):
    """Run UMAP on X (or a reduced version) and save coords."""
    if not UMAP_AVAILABLE:
        print("umap-learn not available install with: pip install umap-learn")
        return None

    Xdense = _ensure_dense_for_embedding(X, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare dense input for UMAP")
        return None

    cfg = config.copy()
    n_components = cfg.pop('n_components', 2)
    try:
        reducer = _umap.UMAP(n_components=n_components, **cfg, random_state=42)
        coords = reducer.fit_transform(Xdense)
    except Exception as e:
        print(f"UMAP failed: {e}")
        return None

    Path(output_dir).mkdir(exist_ok=True)
    np.save(f'{output_dir}/coords_umap.npy', coords)
    print(f"Saved UMAP coords shape={coords.shape} to {output_dir}/coords_umap.npy")
    return coords


def compute_neighbors(X_or_coords, n_neighbors=10, output_dir='artifacts'):
    """Compute nearest neighbors on provided data (dense or coords) and save indices/distances."""
    if not NEIGHBORS_AVAILABLE:
        print("sklearn NearestNeighbors not available install scikit-learn")
        return None, None

    if X_or_coords is None:
        print("No input provided to compute_neighbors")
        return None, None

    Xdense = _ensure_dense_for_embedding(X_or_coords, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare input for nearest-neighbors")
        return None, None

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, Xdense.shape[0]-1))
    nn.fit(Xdense)
    distances, indices = nn.kneighbors(Xdense)

    Path(output_dir).mkdir(exist_ok=True)
    np.save(f'{output_dir}/nn_indices.npy', indices)
    np.save(f'{output_dir}/nn_distances.npy', distances)
    print(f"Saved nearest-neighbor indices/distances to {output_dir}")
    return indices, distances


def run_clustering(df, X_or_coords, config=CLUSTER_CONFIG, output_dir='artifacts'):
    """Run clustering, add labels to df, and save results.

    Supports 'kmeans' (sklearn) and 'hdbscan' (if installed).
    Returns labels (numpy array) or None.
    """
    if X_or_coords is None:
        print("No input provided to run_clustering")
        return None

    Xdense = _ensure_dense_for_embedding(X_or_coords, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare input for clustering")
        return None

    method = config.get('method', 'kmeans')
    labels = None

    if method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            print("hdbscan not available; falling back to kmeans")
            method = 'kmeans'
        else:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=config.get('min_cluster_size', 5))
                labels = clusterer.fit_predict(Xdense)
            except Exception as e:
                print(f"hdbscan failed: {e}")
                labels = None

    if method == 'kmeans':
        if not CLUSTERING_AVAILABLE:
            print("sklearn KMeans not available install scikit-learn")
            return None
        n_clusters = config.get('n_clusters', 8)
        try:
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(Xdense)
        except Exception as e:
            print(f"KMeans failed: {e}")
            labels = None

    if labels is not None:
        # attach to df
        try:
            df['cluster'] = labels.tolist()
        except Exception:
            df['cluster'] = list(labels)

        Path(output_dir).mkdir(exist_ok=True)
        np.save(f'{output_dir}/cluster_labels.npy', labels)
        # update processed_data.csv if it exists in-memory df
        try:
            df.to_csv(f'{output_dir}/processed_data_with_clusters.csv', index=False)
        except Exception:
            pass

        print(f"Saved cluster labels (method={method}) to {output_dir}/cluster_labels.npy")
    else:
        print("No cluster labels produced")

    return labels



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