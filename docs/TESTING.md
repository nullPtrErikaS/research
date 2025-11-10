# Testing and Baselines

This document explains the project's testing strategy, where baselines live, and how to run/update them.

Key artifacts
- `artifacts/metrics.json` — latest computed metrics (trustworthiness, k-NN overlap, separation).
- `tests/fixtures/metrics_baseline.json` — committed baseline used by regression tests.
- `artifacts/point_drift.csv` and `visualizations/point_drift.png` — per-document drift diagnostics.

Running tests
- Run the full test suite:
```powershell
pytest -q
```

Numeric regression
- `tests/test_metrics_regression.py` compares `artifacts/metrics.json` to the committed baseline with tolerances:
  - trustworthiness (k=5): ±0.02
  - k-NN overlap (k=10): ±0.03
  - separation ratio: ±10% (relative)
- To update the baseline: run `python scripts/eval_metrics.py` to produce a new `artifacts/metrics.json`, inspect results, then copy it to `tests/fixtures/metrics_baseline.json` and commit.

Reproducibility & stability
- `tests/test_pipeline_reproducible.py` runs the evaluation twice and asserts numeric equality (useful because evaluation reads saved coords only).
- `tests/test_metric_stability.py` runs 10 quick evaluation repeats and checks the std of UMAP trustworthiness is small (default threshold 0.05).

Nearest-neighbor & distance checks
- `tests/test_distance_metric_consistency.py` verifies nearest-neighbor artifacts exist and distances are finite.

Point-drift
- Use `python scripts/compute_point_drift.py` to compute per-doc displacement and create `artifacts/point_drift.csv` and `visualizations/point_drift.png`.

CI recommendations
- Keep CI jobs light: prefer loading precomputed artifacts in CI tests rather than re-running UMAP/t-SNE in full. Run heavier sweeps (preprocessing/parameter sweeps) on schedule or locally.
