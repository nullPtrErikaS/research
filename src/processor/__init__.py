"""Lightweight processor package that re-exports functions from the
legacy `parse.py` implementation and from new modular implementations.

This file provides a stable import surface while the codebase is being
incrementally split into `src/processor/*` modules.
"""

from parse import (
    load_corpus,
    normalize_corpus,
    plot_scatter,
    plot_comparison,
    plot_heatmap,
    run_tsne,
    run_umap,
    write_summary_report,
    setup_dirs,
    save_artifacts,
)

# New modular implementations (prefer these as refactor continues)
from .preprocess import preprocess_texts
from .vectorize import build_tfidf
from .reduce import run_pca
from .neighbors import compute_neighbors
from .cluster import run_clustering
from .embed import run_tsne, run_umap
from .viz import plot_scatter, plot_comparison, plot_heatmap

__all__ = [
    'load_corpus',
    'normalize_corpus',
    'preprocess_texts',
    'build_tfidf',
    'run_pca',
    'plot_scatter',
    'plot_comparison',
    'plot_heatmap',
    'run_tsne',
    'run_umap',
    'compute_neighbors',
    'run_clustering',
    'write_summary_report',
    'setup_dirs',
    'save_artifacts',
]
