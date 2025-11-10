import json
from pathlib import Path
import math


BASE = Path(__file__).resolve().parents[2] / 'artifacts'
FIX = Path(__file__).resolve().parents[1] / 'fixtures'


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def almost_equal(a, b, tol=1e-8):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol


def test_metrics_against_baseline():
    baseline = load_json(FIX / 'metrics_baseline.json')
    assert (BASE / 'metrics.json').exists(), 'metrics.json missing in artifacts/'
    current = load_json(BASE / 'metrics.json')

    # For each method, check trust_k5 and knn_overlap_k10 within tolerances
    tolerances = {'trustworthiness_k5': 0.02, 'knn_overlap_k10': 0.03}

    for method, bvals in baseline.get('runs', {}).items():
        cvals = current.get('runs', {}).get(method)
        assert cvals is not None, f'{method} missing in current metrics'
        # trustworthiness
        b_tw = bvals.get('trustworthiness_k5')
        c_tw = cvals.get('trustworthiness_k5')
        assert c_tw is not None and b_tw is not None
        assert abs(c_tw - b_tw) <= tolerances['trustworthiness_k5'], f'{method} trust_k5 drift: {c_tw} vs {b_tw}'

        # knn overlap
        b_ko = bvals.get('knn_overlap_k10')
        c_ko = cvals.get('knn_overlap_k10')
        if b_ko is not None:
            assert c_ko is not None
            assert abs(c_ko - b_ko) <= tolerances['knn_overlap_k10'], f'{method} knn_overlap_k10 drift: {c_ko} vs {b_ko}'

        # separation ratio relative tolerance 10%
        b_sep = bvals.get('cluster_separation', {}).get('separation_ratio')
        c_sep = cvals.get('cluster_separation', {}).get('separation_ratio')
        if b_sep is not None and c_sep is not None:
            rel = abs(c_sep - b_sep) / max(1e-8, abs(b_sep))
            assert rel <= 0.10, f'{method} separation ratio drift too large: {c_sep} vs {b_sep}'

    # basic property checks
    for m, r in current.get('runs', {}).items():
        tw = r.get('trustworthiness_k5')
        ko = r.get('knn_overlap_k10')
        assert tw is None or (0.0 <= tw <= 1.0)
        assert ko is None or (0.0 <= ko <= 1.0)
