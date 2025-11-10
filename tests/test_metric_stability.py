import runpy
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2] / 'artifacts'


def run_eval_and_get(metric='umap'):
    runpy.run_path('scripts/eval_metrics.py', run_name='__main__')
    data = json.load(open(BASE / 'metrics.json', 'r', encoding='utf-8'))
    val = data.get('runs', {}).get(metric, {}).get('trustworthiness_k5')
    return val


def test_stability_10_runs():
    vals = []
    for i in range(10):
        v = run_eval_and_get('umap')
        if v is not None:
            vals.append(v)
    if len(vals) >= 2:
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        # expect std to be small; if not, the test will warn by failing
        assert std < 0.05, f'UMAP trustworthiness std too large across runs: {std}'
