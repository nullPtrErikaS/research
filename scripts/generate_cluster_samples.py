"""Generate a small CSV with sample documents per cluster.

This script tries to read `artifacts/processed_data_with_clusters.csv` (preferred).
If missing, it will attempt to load `artifacts/processed_data.csv` and `artifacts/cluster_labels.npy`.
It then writes `artifacts/cluster_samples.csv` containing up to `N_SAMPLES` examples per cluster.
"""
import csv
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('artifacts')
BASE.mkdir(exist_ok=True)
OUT = BASE / 'cluster_samples.csv'

N_SAMPLES = 3

# try preferred file
pref = BASE / 'processed_data_with_clusters.csv'
fallback = BASE / 'processed_data.csv'
labels_np = BASE / 'cluster_labels.npy'

if pref.exists():
    df = pd.read_csv(pref)
elif fallback.exists() and labels_np.exists():
    df = pd.read_csv(fallback)
    try:
        labels = np.load(str(labels_np))
        df['cluster'] = labels.tolist()
    except Exception as e:
        print('Could not load cluster labels:', e)
        df['cluster'] = -1
else:
    print('No processed_data CSV found in artifacts/. Run the pipeline first.')
    raise SystemExit(1)

# ensure a minimal set of text columns present
text_cols = [c for c in ['Full Guideline', 'cleaned_text', 'preprocessed_text'] if c in df.columns]
if not text_cols:
    print('No text columns found (expected one of Full Guideline / cleaned_text / preprocessed_text)')
    raise SystemExit(1)

out_rows = []
for cluster in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == cluster]
    if sub.empty:
        continue
    samples = sub.head(N_SAMPLES)
    for _, row in samples.iterrows():
        out_rows.append({
            'cluster': int(cluster),
            'doc_id': row.get('doc_id', ''),
            'text_short': (row.get('cleaned_text') or row.get('preprocessed_text') or '')[:400],
            'text_full': row.get('Full Guideline', '') if 'Full Guideline' in row else (row.get('cleaned_text','') or row.get('preprocessed_text','')),
        })

# write CSV
with open(OUT, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['cluster', 'doc_id', 'text_short', 'text_full'])
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)

print('Wrote cluster samples to', OUT)
