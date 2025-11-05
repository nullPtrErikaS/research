import numpy as np
import pandas as pd
from pathlib import Path

from src.processor import plot_scatter, plot_heatmap

VIS = Path('visualizations')


def test_plot_scatter_and_heatmap(tmp_path):
    # create tiny coords and df
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]])
    df = pd.DataFrame({'doc_id': ['a', 'b', 'c']})
    outdir = str(tmp_path / 'viz_out')
    plot_scatter(coords, df, output_dir=outdir, filename='test_scatter.png', title='test')
    plot_heatmap(coords, output_dir=outdir, filename='test_heatmap.png', bins=10)
    assert (Path(outdir) / 'test_scatter.png').exists()
    assert (Path(outdir) / 'test_heatmap.png').exists()
