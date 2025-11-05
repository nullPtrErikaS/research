"""Orchestrator to run the pipeline and evaluation.

Usage:
  python scripts/run_all.py [--skip-pipeline] [--skip-eval] [--dry-run]

By default it runs the pipeline (run_pipeline.py) and then the evaluation
script (scripts/eval_metrics.py). It snapshots a small `artifacts/config.json`
with the timestamp and flags used.
"""
import argparse
import json
from pathlib import Path
import time
import runpy
import sys

ARTIFACTS = Path('artifacts')
ARTIFACTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Run pipeline and evaluation')
    parser.add_argument('--skip-pipeline', action='store_true', help='Skip running run_pipeline.py')
    parser.add_argument('--skip-eval', action='store_true', help='Skip running scripts/eval_metrics.py')
    parser.add_argument('--dry-run', action='store_true', help='Do not execute; just write config and print actions')
    args = parser.parse_args()

    cfg = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'skip_pipeline': bool(args.skip_pipeline),
        'skip_eval': bool(args.skip_eval),
        'dry_run': bool(args.dry_run),
    }

    cfg_path = ARTIFACTS / 'config.json'
    try:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        print('Wrote config snapshot to', cfg_path)
    except Exception as e:
        print('Could not write config.json:', e)

    if args.dry_run:
        print('Dry-run: nothing executed')
        return

    if not args.skip_pipeline:
        print('\n=== Running pipeline (run_pipeline.py) ===')
        try:
            runpy.run_path('run_pipeline.py', run_name='__main__')
        except Exception as e:
            print('Pipeline execution failed:', e)

    if not args.skip_eval:
        print('\n=== Running evaluation (scripts/eval_metrics.py) ===')
        try:
            runpy.run_path('scripts/eval_metrics.py', run_name='__main__')
        except Exception as e:
            print('Evaluation execution failed:', e)

    print('\nAll requested steps completed.')


if __name__ == '__main__':
    main()
