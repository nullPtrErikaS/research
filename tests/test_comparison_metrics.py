"""
Tests for the two-document comparison panel distance metrics.

The app's compute_pair_metrics() function (defined inline inside the Document
Details tab) returns:
  - 'cosine'    : cosine similarity from sklearn.metrics.pairwise.cosine_similarity
  - 'euclidean' : L2 norm from numpy.linalg.norm

The projection-distance helper p_dist() (also inline) computes the Euclidean
distance between two points in each 2-D projection space.

These tests reproduce that logic directly so we can verify correctness with
known vectors, without needing to import the Streamlit app script.
"""

import math
import unittest

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Replication of the app's distance helpers
# ---------------------------------------------------------------------------

def compute_pair_metrics(vec_a, vec_b):
    """Mirror of the app's compute_pair_metrics (embedding-space metrics)."""
    cosine_val = float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0])
    euclid_val = float(np.linalg.norm(vec_a - vec_b))
    return {"cosine": cosine_val, "euclidean": euclid_val}


def projection_distance(row_a, row_b, col_x, col_y):
    """Mirror of the app's p_dist() helper for projection-space distances."""
    return math.sqrt(
        (row_a[col_x] - row_b[col_x]) ** 2 + (row_a[col_y] - row_b[col_y]) ** 2
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors_have_similarity_one(self):
        v = np.array([1.0, 2.0, 3.0])
        metrics = compute_pair_metrics(v, v)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)

    def test_opposite_vectors_have_similarity_minus_one(self):
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([-1.0, 0.0, 0.0])
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["cosine"], -1.0, places=6)

    def test_orthogonal_vectors_have_similarity_zero(self):
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["cosine"], 0.0, places=6)

    def test_known_cosine_value(self):
        # cos([1,1], [1,0]) = 1/sqrt(2) ≈ 0.7071
        v = np.array([1.0, 1.0])
        w = np.array([1.0, 0.0])
        metrics = compute_pair_metrics(v, w)
        expected = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(metrics["cosine"], expected, places=6)

    def test_cosine_is_scale_invariant(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 4.0, 6.0])  # same direction, twice the magnitude
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)


class TestEuclideanDistance(unittest.TestCase):

    def test_same_vector_has_zero_distance(self):
        v = np.array([3.0, 4.0])
        metrics = compute_pair_metrics(v, v)
        self.assertAlmostEqual(metrics["euclidean"], 0.0, places=6)

    def test_unit_vectors_along_axes(self):
        # distance between (1,0) and (0,1) = sqrt(2)
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["euclidean"], math.sqrt(2), places=6)

    def test_pythagorean_triple(self):
        # (0,0,0) to (3,4,0) = 5.0
        v = np.zeros(3)
        w = np.array([3.0, 4.0, 0.0])
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["euclidean"], 5.0, places=6)

    def test_euclidean_is_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        m_vw = compute_pair_metrics(v, w)
        m_wv = compute_pair_metrics(w, v)
        self.assertAlmostEqual(m_vw["euclidean"], m_wv["euclidean"], places=6)


class TestProjectionDistance(unittest.TestCase):
    """Tests for the p_dist() helper that operates on 2-D projection coordinates."""

    def _make_row(self, tx, ty, ux, uy, px, py):
        return {"tsne_x": tx, "tsne_y": ty, "umap_x": ux, "umap_y": uy, "pca_x": px, "pca_y": py}

    def test_same_point_zero_distance(self):
        row = self._make_row(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        for cx, cy in [("tsne_x", "tsne_y"), ("umap_x", "umap_y"), ("pca_x", "pca_y")]:
            self.assertAlmostEqual(projection_distance(row, row, cx, cy), 0.0, places=6)

    def test_known_tsne_distance(self):
        a = self._make_row(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        b = self._make_row(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(projection_distance(a, b, "tsne_x", "tsne_y"), 5.0, places=6)

    def test_known_umap_distance(self):
        a = self._make_row(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        b = self._make_row(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(
            projection_distance(a, b, "umap_x", "umap_y"), math.sqrt(2), places=6
        )

    def test_known_pca_distance(self):
        a = self._make_row(0.0, 0.0, 0.0, 0.0, -1.0, -1.0)
        b = self._make_row(0.0, 0.0, 0.0, 0.0, 2.0, 3.0)
        # sqrt((2-(-1))^2 + (3-(-1))^2) = sqrt(9+16) = 5
        self.assertAlmostEqual(projection_distance(a, b, "pca_x", "pca_y"), 5.0, places=6)

    def test_all_three_projections_independent(self):
        """Each projection's distance is computed from its own coordinate pair."""
        a = self._make_row(0.0, 0.0,  0.0, 0.0,  0.0, 0.0)
        b = self._make_row(1.0, 0.0,  0.0, 2.0,  3.0, 4.0)
        self.assertAlmostEqual(projection_distance(a, b, "tsne_x", "tsne_y"), 1.0, places=6)
        self.assertAlmostEqual(projection_distance(a, b, "umap_x", "umap_y"), 2.0, places=6)
        self.assertAlmostEqual(projection_distance(a, b, "pca_x",  "pca_y"),  5.0, places=6)


class TestBothMetricsTogether(unittest.TestCase):
    """Sanity-check that both metrics are returned correctly in the same call."""

    def test_returns_both_keys(self):
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        metrics = compute_pair_metrics(v, w)
        self.assertIn("cosine", metrics)
        self.assertIn("euclidean", metrics)

    def test_high_cosine_low_euclidean_for_similar_scaled_vectors(self):
        v = np.array([10.0, 10.0])
        w = np.array([10.001, 10.001])
        metrics = compute_pair_metrics(v, w)
        self.assertGreater(metrics["cosine"], 0.999)
        self.assertLess(metrics["euclidean"], 0.1)

    def test_low_cosine_can_coexist_with_low_euclidean(self):
        # Two vectors very close in space but pointing in opposite directions
        # is mathematically impossible for unit vectors, but for tiny magnitudes
        # they can both be small.  This verifies the metrics are independent.
        v = np.array([0.001, 0.0])
        w = np.array([-0.001, 0.0])
        metrics = compute_pair_metrics(v, w)
        self.assertAlmostEqual(metrics["cosine"], -1.0, places=4)
        self.assertAlmostEqual(metrics["euclidean"], 0.002, places=6)


if __name__ == "__main__":
    unittest.main()
