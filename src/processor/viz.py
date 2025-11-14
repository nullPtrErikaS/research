"""Visualization helpers: scatter and heatmap plotting.

Extracted from the legacy `parse.py` to centralize plotting utilities.
"""
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def plot_scatter(coords, df, output_dir='visualizations', title='scatter', filename=None, axis_limits=None,
                 label_col=None, annotate_top_n=10, metric=None):
    """Plot 2D scatter of `coords` with optional labels taken from `df`.

    label_col : str or None
        Column name in `df` to use for point labels (e.g. 'Guideline + Slogan').
        Falls back to 'doc_id' if not present. If None, no labels are used except
        for the first `annotate_top_n` points.
    annotate_top_n : int
        Number of points to annotate with labels from `label_col` (default 10).
    metric : str or None
        Optional metric name to include in the output filename and printed title.
    """
    if coords is None:
        print('No coords to plot')
        return
    Path(output_dir).mkdir(exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib not installed — cannot produce plot. Install with: pip install matplotlib')
        return

    plt.figure(figsize=(8, 6))
    x, y = coords[:, 0], coords[:, 1]
    plt.scatter(x, y, s=10, alpha=0.7)
    # Make metric visible in title when provided
    t = title
    if metric:
        t = f"{title} (metric={metric})"
    plt.title(t)
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    if axis_limits is not None:
        xmin, xmax, ymin, ymax = axis_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # Choose label column (fallback to doc_id)
    if label_col is None:
        label_col = 'doc_id'
    if label_col not in df.columns:
        # fallback
        label_col = 'doc_id'

    # Annotate first N points with human-friendly names (truncate long labels)
    def _shorten(s, n=80):
        s = str(s)
        return s if len(s) <= n else s[: n-1].rstrip() + '…'

    n_to_annotate = min(max(0, int(annotate_top_n)), coords.shape[0])
    for i in range(n_to_annotate):
        label = df.iloc[i].get(label_col, df.iloc[i].get('doc_id', str(i)))
        label = _shorten(label, n=60)
        plt.annotate(label, (x[i], y[i]), fontsize=6, alpha=0.9)

    # Build output filename: if not provided, include metric when available
    if filename is None:
        metric_tag = f'_{str(metric).replace(" ","_")}' if metric else ''
        outpath = f'{output_dir}/scatter{metric_tag}.png'
    else:
        outpath = str(Path(output_dir) / filename)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'Saved scatter to {outpath}')


def plot_comparison(coords_a, coords_b, df, output_dir='visualizations', filename='tsne_vs_umap.png', titles=('A','B'), mode='shared', axis_limits_a=None, axis_limits_b=None):
    if coords_a is None or coords_b is None:
        print('Missing coords for comparison plot')
        return
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib not installed — cannot produce comparison plot')
        return
    Path(output_dir).mkdir(exist_ok=True)

    import matplotlib.pyplot as plt
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
    print(f'Saved comparison plot to {outpath}')


def plot_heatmap(coords, output_dir='visualizations', filename='embedding_heatmap.png', bins=200, cmap='viridis'):
    if coords is None:
        print('No coords for heatmap')
        return
    if not MATPLOTLIB_AVAILABLE:
        print('matplotlib not installed — cannot produce heatmap')
        return
    Path(output_dir).mkdir(exist_ok=True)

    import matplotlib.pyplot as plt
    x = coords[:, 0]
    y = coords[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    fig, ax = plt.subplots(figsize=(6, 5))
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
    print(f'Saved heatmap to {outpath}')
