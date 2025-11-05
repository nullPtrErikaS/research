"""Embedding helpers: t-SNE and UMAP wrappers extracted from parse.py.

This module provides `run_tsne` and `run_umap` which prepare dense inputs
and run the respective reducers, saving coords to `artifacts/`.
"""
from pathlib import Path
import numpy as np

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
    from sklearn.decomposition import TruncatedSVD
except Exception:
    TruncatedSVD = None


def _ensure_dense_for_embedding(X, n_components_for_init=50):
    if X is None:
        return None
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            if TruncatedSVD is not None:
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
    try:
        return np.array(X)
    except Exception:
        return None


def run_tsne(X, config=None, output_dir='artifacts'):
    cfg = (config or {'n_components': 2, 'perplexity': 30}).copy()
    if not TSNE_AVAILABLE:
        print("scikit-learn TSNE not available install scikit-learn>=0.24")
        return None

    Xdense = _ensure_dense_for_embedding(X, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare dense input for t-SNE")
        return None

    n_components = cfg.pop('n_components', 2)
    try:
        tsne = TSNE(n_components=n_components, **cfg, random_state=42)
        coords = tsne.fit_transform(Xdense)
    except TypeError:
        # fallback for older sklearn API
        tsne = TSNE(n_components=n_components, random_state=42)
        coords = tsne.fit_transform(Xdense)

    Path(output_dir).mkdir(exist_ok=True)
    np.save(f'{output_dir}/coords_tsne.npy', coords)
    print(f"Saved t-SNE coords shape={coords.shape} to {output_dir}/coords_tsne.npy")
    return coords


def run_umap(X, config=None, output_dir='artifacts'):
    cfg = (config or {'n_components': 2, 'n_neighbors': 15}).copy()
    if not UMAP_AVAILABLE:
        print("umap-learn not available install with: pip install umap-learn")
        return None

    Xdense = _ensure_dense_for_embedding(X, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare dense input for UMAP")
        return None

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
