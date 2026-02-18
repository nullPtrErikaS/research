# Short-Text Embedding Explorer

A modular pipeline and interactive visualization tool for analyzing short-text embeddings. This project enables researchers to compare dimensionality reduction techniques (PCA, t-SNE, UMAP), explore cluster structure, and analyze document neighborhoods across multiple projection methods simultaneously.

Build Status
(Add your CI/CD badge here if using one)

Table of Contents
- Introduction
- Features
- Installation
- Usage
- Screenshots
- Tests
- Contributing
- License

## Features

- **Linked brushing**: Selections sync across PCA, t-SNE, and UMAP in real time
- **Search & filter**: By document ID, keyword, or cluster
- **Comparison panel**: Side-by-side documents with similarity metrics and neighbor overlap
- **Selection history**: Undo/redo without losing context
- **Artifact validation**: Automatic checks for row alignment and dimension consistency
- **Export**: Download selected document IDs as JSON

## Installation

```bash
# Clone
git clone https://github.com/nullPtrErikaS/research.git
cd research

# Setup environment (with venv)
python -m venv .venv
# Activate: (Windows: .\.venv\Scripts\Activate.ps1 // Unix: source .venv/bin/activate)
pip install -r requirements.txt

# Or install as a package (recommended):
pip install .
```

## Usage

1. **Run Pipeline** (Preprocessing + Embedding):
```bash
python run_pipeline.py
```
This generates artifacts in `artifacts/`.

2. **Run Explorer App**:
```bash
streamlit run streamlit_app/streamlit_app.py
```

## Screenshots

(Place high-contrast screenshots or GIFs here visualizing the app)

## Tests

Run the test suite to ensure stability:
```bash
pytest tests/
```

## Contributing

See CONTRIBUTING.md for details.

## License

MIT License.
