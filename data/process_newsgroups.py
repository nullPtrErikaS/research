"""Run the pipeline on the newsgroups dataset."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path to import parse module
sys.path.insert(0, str(Path(__file__).parent.parent))

import parse

def main():
    # Load the newsgroups dataset
    data_path = Path(__file__).parent / 'newsgroups_20.csv'
    df = pd.read_csv(data_path)
    
    # Add doc_id
    df['doc_id'] = [f"doc_{i:04d}" for i in range(len(df))]
    print(f"Loaded {len(df)} docs from newsgroups dataset")
    
    # Clean the text column
    df['cleaned_text'] = df['text'].apply(parse.basic_clean)
    non_empty = (df['cleaned_text'] != '').sum()
    print(f"Cleaned {non_empty}/{len(df)} non-empty texts")
    
    # Save processed data
    output_dir = Path(__file__).parent.parent / 'artifacts' / 'newsgroups'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'processed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    # Display some stats
    print(f"\nDataset summary:")
    print(f"  Total documents: {len(df)}")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  Category distribution:\n{df['category'].value_counts().head(10)}")

if __name__ == '__main__':
    main()
