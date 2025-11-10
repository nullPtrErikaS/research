"""Compute per-document coordinate drift between a baseline and current run.

This script looks for baseline coordinates in `artifacts/point_coords_baseline.npy`.
If missing, it will create the baseline from `artifacts/coords.npy` (PCA/SVD coords).
It then loads the most recent coords (prefers UMAP, then t-SNE, then PCA) and
computes per-document L2 displacement. Outputs:
- `artifacts/point_drift.csv` with columns (doc_id, baseline_x, baseline_y, run_x, run_y, displacement)
- `visualizations/point_drift.png` color-coded scatter highlighting large movers.

Usage: python scripts/compute_point_drift.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parents[1] / 'artifacts'
VIS = Path(__file__).resolve().parents[1] / 'visualizations'
ART.mkdir(exist_ok=True)
VIS.mkdir(exist_ok=True)


def load_coords_preference():
    # prefer specific embeddings if present
    prefer = ['coords_umap.npy', 'coords_tsne.npy', 'coords.npy']
    for p in prefer:
        fp = ART / p
        if fp.exists():
            try:
                arr = np.load(str(fp))
                print('Loaded coords from', fp.name)
                return arr, fp.name
            except Exception:
                continue
    return None, None


def main():
    # Baseline path
    baseline_path = ART / 'point_coords_baseline.npy'

    # If baseline missing, try to create from coords.npy
    if not baseline_path.exists():
        pca_path = ART / 'coords.npy'
        if pca_path.exists():
            try:
                arr = np.load(str(pca_path))
                np.save(str(baseline_path), arr)
                print('Created baseline coords from coords.npy')
            except Exception as e:
                print('Could not create baseline coords:', e)
        else:
            print('No baseline coords found and coords.npy missing; aborting')
            return

    baseline = np.load(str(baseline_path))
    run_coords, run_name = load_coords_preference()
    if run_coords is None:
        print('No run coordinates found (coords_umap/coords_tsne/coords.npy). Aborting.')
        return

    # Align shapes
    if baseline.shape != run_coords.shape:
        n = min(baseline.shape[0], run_coords.shape[0])
        baseline = baseline[:n]
        run_coords = run_coords[:n]

    # Load doc ids if available
    doc_ids_path = ART / 'doc_ids.txt'
    if doc_ids_path.exists():
        with open(str(doc_ids_path), 'r', encoding='utf-8') as f:
            doc_ids = [l.strip() for l in f.readlines()][:baseline.shape[0]]
    else:
        doc_ids = [str(i) for i in range(baseline.shape[0])]

    # Compute displacements
    diffs = run_coords - baseline
    displacements = np.linalg.norm(diffs, axis=1)

    df = pd.DataFrame({
        'doc_id': doc_ids,
        'baseline_x': baseline[:,0],
        'baseline_y': baseline[:,1],
        'run_x': run_coords[:,0],
        'run_y': run_coords[:,1],
        'displacement': displacements,
    })

    out_csv = ART / 'point_drift.csv'
    df.to_csv(str(out_csv), index=False)
    print('Wrote point drift CSV to', out_csv)

    # Make a scatter plot colored by displacement (percentile-based clipping)
    vmax = np.percentile(displacements, 99)
    vmin = np.percentile(displacements, 1)

    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(baseline[:,0], baseline[:,1], c=displacements, cmap='viridis', s=10, vmin=vmin, vmax=vmax)
    ax.set_title(f'Point drift (baseline -> {run_name})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('L2 displacement')

    # highlight top movers
    top_idx = np.argsort(displacements)[-20:]
    ax.scatter(baseline[top_idx,0], baseline[top_idx,1], facecolors='none', edgecolors='red', s=50, linewidths=1.2)
    for i in top_idx:
        ax.text(baseline[i,0], baseline[i,1], str(doc_ids[i]), fontsize=6)

    out_png = VIS / 'point_drift.png'
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
    print('Wrote point drift plot to', out_png)


if __name__ == '__main__':
    main()
