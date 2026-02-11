
import os
import numpy as np
import pandas as pd

def check_dir(d):
    print(f"\n--- Checking {d} ---")
    
    doc_ids_path = os.path.join(d, 'doc_ids.txt')
    csv_path = os.path.join(d, 'processed_data_with_clusters.csv')
    tsne_path = os.path.join(d, 'coords_tsne.npy')
    
    doc_ids = []
    if os.path.exists(doc_ids_path):
        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            doc_ids = [l.strip() for l in f if l.strip()]
        print(f"doc_ids.txt: {len(doc_ids)} lines")
    else:
        print("doc_ids.txt: MISSING")
        
    df = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"csv: {len(df)} rows")
        except Exception as e:
            print(f"csv error: {e}")
    else:
        print("csv: MISSING")
        
    if os.path.exists(tsne_path):
        try:
            arr = np.load(tsne_path)
            print(f"tsne.npy shape: {arr.shape}")
        except Exception as e:
             print(f"tsne error: {e}")
    else:
        print("tsne.npy: MISSING")

    # Check order
    if doc_ids and df is not None:
        if len(doc_ids) != len(df):
            print("!!! LENGTH MISMATCH !!!")
        
        # Check first 5
        print("First 5 doc_ids.txt:", doc_ids[:5])
        print("First 5 csv doc_ids:", df['doc_id'].tolist()[:5] if 'doc_id' in df.columns else "No doc_id col")
        
        if 'doc_id' in df.columns:
            matches = 0
            for i in range(min(len(doc_ids), len(df))):
                if str(doc_ids[i]) == str(df.iloc[i]['doc_id']):
                    matches += 1
            print(f"Position matches in first {min(len(doc_ids), len(df))}: {matches}")

check_dir('c:\\Users\\wolff\\research\\artifacts')
check_dir('c:\\Users\\wolff\\research\\artifacts\\preproc_default')
