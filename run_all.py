#!/usr/bin/env python3
"""
Simple wrapper to launch the Streamlit prototype: `python run_all.py`
This runs `streamlit run prototype/streamlit_app.py` using the current Python interpreter.
"""
import os
import sys
import subprocess

SCRIPT = os.path.join(os.path.dirname(__file__), 'prototype', 'streamlit_app.py')
if not os.path.exists(SCRIPT):
    print('Prototype script not found at', SCRIPT)
    sys.exit(1)

try:
    # Use `-m streamlit` to ensure we use the streamlit installed in the current environment
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', SCRIPT], check=True)
except subprocess.CalledProcessError as e:
    print('Streamlit exited with', e.returncode)
    sys.exit(e.returncode)
except Exception as e:
    print('Error launching Streamlit:', e)
    print('Make sure you have the prototype requirements installed:')
    print('  pip install -r prototype/requirements.txt')
    sys.exit(1)
