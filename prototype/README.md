Embeddings Explorer — Prototype

Quick prototype viewer (Streamlit) for exploring embeddings and keeping linked selections in sync across dimensionality reductions.

## How to run

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r prototype/requirements.txt
```

2. Launch the Streamlit experience from the repo root:

```powershell
streamlit run prototype/streamlit_app.py
```

The app auto-detects `artifacts/*.npy` and `processed_data_with_clusters.csv`. If they are missing, a tiny synthetic dataset is generated so the UI still works.

## Feature highlights

- **Linked point selection:** clicking any point in t-SNE, UMAP, or PCA instantly highlights the same `doc_id` everywhere; linked brushing (lasso/box) works via Plotly selections.
- **Hover insights:** tooltips surface the `doc_id`, truncated snippet, cluster, and key metadata fields so you can triage points without leaving the scatterplots.
- **Search & filtering:** search by doc_id, keyword tags, or free-text phrases; combine with cluster, keyword-tag, and metadata filters (categorical + numeric sliders) to zero in on a cohort.
- **Chunk ↔ parent linking:** optional chunk/parent mapping means selecting a chunk auto-selects its parent document (and vice versa) across all plots.
- **Comparison workbench:** side-by-side panels show snippets, metadata, cosine similarity, Euclidean distance, and neighbor tables for two arbitrary documents.
- **Neighbor context:** nearest-neighbor tables update with the current filtered subset so you can inspect local neighborhoods without leaving the app.
- **Parameter playground:** tweak PCA components plus UMAP/t-SNE knobs and optionally recompute previews (requires `umap-learn`).

## Boxer vs Streamlit

| Criteria | Boxer | Streamlit | Notes |
| --- | --- | --- | --- |
| Interaction model | Low-level Canvas-style primitives; flexible but verbose | High-level widgets + Plotly integrations out of the box | Streamlit let us wire linked brushing with `streamlit-plotly-events` quickly |
| State handling | Requires custom Redux-like plumbing | `st.session_state` keeps shared selection/search state trivial | Our prototype syncs selections + filters without extra libs |
| Ecosystem | Smaller, fewer third-party components | Mature ecosystem, easy deployment via Streamlit Cloud | Plotly, Vega-Lite, Altair, and pydeck all first-class |
| Hosting | Needs bespoke hosting story | Streamlit sharing / community cloud friendly | Lowers effort for stakeholder demos |

**Decision:** keep Streamlit for the prototype. It already delivers the required linked interactions, deploys fast for stakeholder reviews, and keeps the codebase Python-native alongside the rest of the repo. Boxer can be revisited if we need fully bespoke WebGL rendering later.

## Usage notes

- This is a lightweight prototype meant for interactive demos, not large-scale production datasets.
- Provide `doc_ids.txt` that lines up with the `.npy` coordinate files to guarantee correct cross-view alignment.
- Install `umap-learn` if you want the sidebar parameter tweaks to recompute UMAP previews; otherwise PCA fallbacks are used.
