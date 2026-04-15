# Short-Text Embedding Explorer

An interactive Streamlit tool for exploring short-text embedding spaces. Researchers run an offline preprocessing and embedding pipeline, then use the app to inspect projections, compare documents, and export findings.

## Features

**Projections & navigation**
- Linked brushing across PCA, t-SNE, and UMAP — selections sync in real time
- Click, box, and lasso selection; additive selection mode
- Expand any projection to full width; collapse back to 3-column view
- Single-focus view with large plot and inline inspector panel

**Selection tools**
- Undo (last 10 selections), Clear, Random sample, Invert selection
- Jump to a specific document by ID; pin up to 15 documents so they stay highlighted across selections
- Saved Groups: save named subsets, rename them, annotate with notes, load with one click
- Keyword preview tooltip on each saved group button (top 5 terms)

**Search & filter**
- Full-text / doc-id / keyword search with optional regex
- OR / AND keyword logic toggle
- Filter by cluster, keyword tags, categorical metadata, and numeric ranges (e.g. Year)

**Visualization modes**
- Color by: Selection status, Semantic clusters, Saved Groups, or Year (with integer-labeled colorbar)
- Adjustable point opacity and size
- Optional hover tooltips

**Analysis panels** (tabs below the plots)
- **Document Details**: paged table of selected docs; single-doc neighbor list; two-doc comparison with cosine similarity, Euclidean distance, and projection distances; midpoint finder; multi-doc group cohesion score
- **Clusters**: cluster stats bar chart, distance heatmap, centroid summary for large selections
- **Top Keywords**: TF-IDF ranked keywords for the current selection (computed against full-corpus IDF); keyword details table

**Orient Me** (sidebar)
- One-click cluster selection buttons, auto-collapses after use

**Session tools**
- Session Reasoning Trail: timestamped action log, restore any past state, direct-download TXT export
- Save/load named sessions to disk (preserves selection, saved groups, notes, and pinned docs)
- Pipeline snapshots and snapshot diff view

**Pipeline & artifacts**
- Artifact bundle selector: switch between preprocessing variants
- Variant diff expander (always visible, shows "no differences" when on default)
- Embedding health score banner (neighborhood coherence, color-coded)
- Pipeline configuration panel with stage lock/unlock
- Reproducibility config export

## Installation

```bash
git clone https://github.com/nullPtrErikaS/research.git
cd research

python -m venv .venv
# Activate:
#   Windows: .\.venv\Scripts\Activate.ps1
#   Unix:    source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

**1. Run the offline pipeline** (required before launching the app):
```bash
python run_pipeline.py
```
This produces artifacts in `artifacts/` — embeddings, projection coordinates, cluster labels, and token data.

**2. Launch the app:**
```bash
streamlit run streamlit_app/streamlit_app.py
```

## Repo structure

```
streamlit_app/
  streamlit_app.py        # Main app — all core logic
  pages/
    1_Documentation.py    # In-app documentation page
artifacts/                # Precomputed pipeline outputs (not committed)
tests/
  test_selection_logic.py
  test_orient_verify.py
  test_regression.py
  test_comparison_metrics.py
scripts/
  run_all.py
run_pipeline.py           # Preprocessing + embedding pipeline
parse.py                  # Tokenization, lemmatization, stop word filtering
config/
  default.yaml
docs/
  gleicher_frontend.html
```

## Tests

```bash
pytest tests/
pytest tests/ -v  # verbose
```

## License

MIT License.
