"""Fetch the 20 Newsgroups dataset and save it as CSV."""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path

# Fetch the dataset
print("Fetching 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(
    subset='train',
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame({
    'text': newsgroups.data,
    'category': [newsgroups.target_names[i] for i in newsgroups.target]
})

# Clean up very short texts
df = df[df['text'].str.len() > 50].reset_index(drop=True)

# Limit to 500 samples for a small dataset
df = df.head(500).reset_index(drop=True)

# Save to CSV
output_path = Path(__file__).parent / 'newsgroups_20.csv'
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} samples to {output_path}")
print(f"Categories: {df['category'].nunique()}")
print(f"\nFirst few rows:")
print(df.head())
