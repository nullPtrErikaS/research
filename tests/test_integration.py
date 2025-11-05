import runpy
import sys
from pathlib import Path

ART = Path('artifacts')


def test_run_all_eval_only():
    # run the orchestrator in eval-only mode
    old_argv = sys.argv.copy()
    try:
        sys.argv = ['scripts/run_all.py', '--skip-pipeline']
        runpy.run_path('scripts/run_all.py', run_name='__main__')
    finally:
        sys.argv = old_argv

    cfg = ART / 'config.json'
    metrics = ART / 'metrics.json'
    assert cfg.exists(), f'Expected {cfg} to exist after run_all'
    assert metrics.exists(), f'Expected {metrics} to exist after run_all' 
