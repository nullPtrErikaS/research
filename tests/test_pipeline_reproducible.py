import runpy
import json
from pathlib import Path
import time


BASE = Path(__file__).resolve().parents[2] / 'artifacts'


def run_eval_and_load():
    # run eval script which reads saved coords and writes metrics.json
    runpy.run_path('scripts/eval_metrics.py', run_name='__main__')
    # give a small sleep to ensure file written (script returns after writing)
    return json.load(open(BASE / 'metrics.json', 'r', encoding='utf-8'))


def test_eval_reproducible():
    # run twice and compare numeric values exactly (evaluation reads artifacts only)
    a = run_eval_and_load()
    time.sleep(0.1)
    b = run_eval_and_load()

    assert 'runs' in a and 'runs' in b
    for k in a['runs'].keys():
        ar = a['runs'][k]
        br = b['runs'][k]
        for metric in ['trustworthiness_k5', 'knn_overlap_k10']:
            assert (ar.get(metric) is None and br.get(metric) is None) or abs(ar.get(metric, 0) - br.get(metric, 0)) < 1e-9
