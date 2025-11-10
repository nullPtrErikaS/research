from pathlib import Path
import numpy as np

ART = Path(__file__).resolve().parents[2] / 'artifacts'


def test_nn_artifacts_exist_and_finite():
    idx = ART / 'nn_indices.npy'
    dist = ART / 'nn_distances.npy'
    assert idx.exists() and dist.exists(), 'Nearest-neighbor artifacts missing'
    inds = np.load(str(idx))
    d = np.load(str(dist))
    assert np.isfinite(d).all(), 'nn_distances contains non-finite values'
    assert inds.shape == d.shape, 'nn indices/distances shapes mismatch'
