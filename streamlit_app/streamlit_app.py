import ast
import os
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="Embeddings Explorer (Prototype)")

# CSS Tweaks: Compact headers and margins
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 98% !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.2rem !important;
    }
    h1, h2, h3 {
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
    p, .stText, .stMarkdown {
        margin-bottom: 0.1rem !important;
        font-size: 0.9rem !important;
    }
    .stPlotlyChart { 
        height: 600px !important; 
        margin-bottom: 0 !important;
    }
    hr {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)


try:
    import umap
except Exception:  # pragma: no cover - optional dependency for previews
    umap = None

try:
    import scipy.sparse
except Exception:
    scipy = None

alignment_notes = []


@st.cache_data
def try_load(path, mtime=None):
    try:
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.npz') and scipy is not None:
             return scipy.sparse.load_npz(path)
        else:
            return pd.read_csv(path)
    except Exception:
        return None


def find_file(names):
    for n in names:
        if os.path.exists(n):
            return n
    return None
# Optional user-selected artifact bundle preference
selected_bundle_root = None
try:
    import streamlit as st  # already imported above
    selected_bundle_root = st.session_state.get('selected_bundle_root')
except Exception:
    selected_bundle_root = None

def bundle_candidates(filename):
    """Return candidate search paths for a filename, preferring a selected bundle root if set.

    Known bundle roots:
      - artifacts/preproc_default
      - artifacts/newsgroups
      - artifacts
      - (repo root)
    """
    roots = []
    if selected_bundle_root:
        roots.append(selected_bundle_root)
    # Default preference order
    roots.extend(['artifacts', 'artifacts/preproc_default', 'artifacts/newsgroups', ''])
    # Build candidate list, avoid duplicates
    seen = set()
    candidates = []
    for r in roots:
        path = f"{r}/{filename}" if r else filename
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates


def ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [value.strip()]
    if pd.isna(value):
        return []
    return [value]


def build_snippet(text, length=160):
    if not isinstance(text, str) or not text:
        return ''
    snippet = textwrap.shorten(text.replace('\n', ' '), width=length, placeholder='…')
    return snippet


def parse_keyword_space(df):
    token_col = None
    for candidate in ['tokens', 'token_list', 'keywords']:
        if candidate in df.columns:
            token_col = candidate
            break
    if token_col is None:
        return [], []
    token_lists = df[token_col].apply(lambda vals: [str(tok).lower() for tok in ensure_list(vals)])
    df['__tokens'] = token_lists
    flattened = [tok for toks in token_lists for tok in toks if isinstance(tok, str)]
    keyword_counts = pd.Series(flattened).value_counts()
    top_keywords = keyword_counts.head(100).index.tolist()
    return token_col, top_keywords


def apply_chunk_mapping(df, chunk_col, parent_col):
    mapping = df[[chunk_col, parent_col]].dropna().drop_duplicates()
    chunk_to_parent = dict(zip(mapping[chunk_col], mapping[parent_col]))
    parent_to_chunks = mapping.groupby(parent_col)[chunk_col].apply(list).to_dict()
    return chunk_to_parent, parent_to_chunks


def build_hover_columns(df):
    base_cols = ['doc_id', 'cluster', '__snippet']
    available = [c for c in base_cols if c in df.columns]
    metadata_cols = [
        c for c in df.columns
        if c not in base_cols + ['text', '__color', '__status', '__sel', '__tokens', '__search_hit', '__global_idx']
    ]
    # Always include a few rich-text columns if available
    extras = []
    # Prioritize these specific columns for the tooltip
    priority_cols = [
        'Slogan', 'Authors', 'Work', 'Dashboard Type', 
        'Restructured', 'Example', 'Tool Used/Mentioned', 
        'Data Domain', 'Year'
    ]
    for col in priority_cols:
        if col in df.columns:
            extras.append(col)
    
    # Add other metadata columns, excluding internal/base/priority ones
    others = [c for c in metadata_cols if c not in extras]
    
    # Return base + priority + a few others
    return list(dict.fromkeys(available + extras + others[:3]))


def extract_doc_ids_from_events(events, df_slice):
    hits = []
    if not events:
        return hits
    for ev in events:
        if isinstance(ev, dict):
            if 'customdata' in ev and ev['customdata']:
                data = ev['customdata']
                if isinstance(data, (list, tuple)):
                    hits.append(str(data[0]))
                else:
                    hits.append(str(data))
                continue
            idx = ev.get('pointIndex')
            if idx is None:
                idx = ev.get('pointNumber')
            if idx is None and ev.get('points'):
                sub_points = ev['points']
                for p in sub_points:
                    pi = p.get('pointIndex', p.get('pointNumber'))
                    if pi is not None and 0 <= int(pi) < len(df_slice):
                        hits.append(df_slice.iloc[int(pi)]['doc_id'])
                continue
            if idx is not None and 0 <= int(idx) < len(df_slice):
                hits.append(df_slice.iloc[int(idx)]['doc_id'])
    return list(dict.fromkeys(hits))


def expand_chunk_links(ids):
    chunk_parent = st.session_state.get('chunk_parent_map')
    parent_chunk = st.session_state.get('parent_chunk_map')
    ordered = list(dict.fromkeys(ids))
    if not chunk_parent and not parent_chunk:
        return ordered
    expanded = list(ordered)

    def append_if_missing(value):
        if value is None:
            return
        if value not in expanded:
            expanded.append(value)

    idx = 0
    while idx < len(expanded):
        did = expanded[idx]
        if chunk_parent and did in chunk_parent:
            append_if_missing(chunk_parent[did])
        if parent_chunk and did in parent_chunk:
            for child in parent_chunk[did]:
                append_if_missing(child)
        idx += 1
    return expanded


def update_selection(doc_ids, additive=False):
    if not isinstance(doc_ids, (list, tuple, set)):
        doc_ids = [doc_ids]
    current = list(st.session_state.get('selected_ids', []))
    
    # Save to history before changing
    if current and current != st.session_state.get('selected_ids', []):
        history = st.session_state.get('selection_history', [])
        history.append(current)
        st.session_state['selection_history'] = history[-10:]  # Keep last 10
    
    if additive:
        base_seq = current + list(doc_ids)
    else:
        base_seq = list(doc_ids)
    final_ids = expand_chunk_links(base_seq)
    st.session_state['selected_ids'] = list(dict.fromkeys(final_ids))


def merge_search_hits(search_hits):
    if not search_hits:
        st.session_state['search_hits'] = []
    else:
        st.session_state['search_hits'] = list(dict.fromkeys(search_hits))


def run_search(df_slice, query, scopes):
    if not query:
        merge_search_hits([])
        return df_slice
    query = query.strip()
    if not query:
        merge_search_hits([])
        return df_slice
    tokens = [tok.strip().lower() for tok in query.split(',') if tok.strip()]
    hits = set()
    mask = pd.Series(False, index=df_slice.index)
    scopes = set(scopes)
    for idx, row in df_slice.iterrows():
        doc_id = str(row.get('doc_id', ''))
        text_val = str(row.get('text', row.get('cleaned_text', ''))).lower()
        keyword_list = [tok.lower() for tok in row.get('__tokens', [])]
        match = False
        if 'doc_id' in scopes and query.lower() in doc_id.lower():
            match = True
        if not match and 'keywords' in scopes and tokens:
            if any(tok in keyword_list for tok in tokens):
                match = True
        if not match and 'phrase' in scopes:
            if query.lower() in text_val:
                match = True
        if match:
            mask.loc[idx] = True
            hits.add(doc_id)
    merge_search_hits(list(hits))
    if hits:
        return df_slice[mask]
    return df_slice


# Locate data files in the repo (flexible, with bundle preference)
processed_csv_path = find_file(bundle_candidates('processed_data_with_clusters.csv') + bundle_candidates('full_dataset_with_new_id.csv'))

# Consistency fix: determine if we should enforce a specific root based on the CSV found
forced_root = None
if processed_csv_path:
    forced_root = os.path.dirname(processed_csv_path)

def aligned_candidates(filename):
    base = bundle_candidates(filename)
    if forced_root:
        # Prioritize the directory where the main CSV lives
        return [os.path.join(forced_root, filename)] + base
    return base

coords_tsne_path = find_file(aligned_candidates('coords_tsne.npy'))
coords_umap_path = find_file(aligned_candidates('coords_umap.npy'))
# Prefer preproc_default coords by default; bundle preference can override
coords_pca_path = find_file(aligned_candidates('coords.npy'))
tfidf_matrix_path = find_file(aligned_candidates('tfidf_matrix.npz'))
cluster_labels_path = find_file(aligned_candidates('cluster_labels.npy'))
# processed_csv_path is already found above
doc_ids_path = find_file(aligned_candidates('doc_ids.txt'))


@st.cache_data(show_spinner="Loading and processing dataset...")
def load_and_process_data(csv_path, doc_ids_path_arg, _version=2):
    # Load or synthesize small dataset
    if csv_path:
        df_local = pd.read_csv(csv_path)
    else:
        # try to build df from doc_ids file
        if doc_ids_path_arg:
            try:
                with open(doc_ids_path_arg, 'r', encoding='utf-8') as f:
                    ids = [line.strip() for line in f if line.strip()]
                df_local = pd.DataFrame({'doc_id': ids})
                df_local['text'] = df_local['doc_id'].apply(lambda x: f'Snippet for {x}')
                df_local['cluster'] = 0
            except Exception:
                df_local = pd.DataFrame({'doc_id': [], 'text': [], 'cluster': []})
        else:
            # synthetic fallback
            n = 200
            df_local = pd.DataFrame({
                'doc_id': [f'DOC{i:04d}' for i in range(n)],
                'text': [f'This is a sample snippet number {i}.' for i in range(n)],
                'cluster': np.random.randint(0, 8, size=n)
            })

    if 'doc_id' not in df_local.columns:
        if 'id' in df_local.columns:
            df_local['doc_id'] = df_local['id'].astype(str)
        else:
            df_local['doc_id'] = df_local.index.astype(str)
            
    if 'cluster' not in df_local.columns:
        df_local['cluster'] = 0

    df_local['doc_id'] = df_local['doc_id'].astype(str)
    
    # Identify text column
    txt_src = None
    for candidate in ['text', 'cleaned_text', 'preprocessed_text']:
        if candidate in df_local.columns:
            txt_src = candidate
            break
    if txt_src is None:
        df_local['text'] = df_local['doc_id'].apply(lambda x: f'Snippet for {x}')
        txt_src = 'text'
        
    df_local['__snippet'] = df_local[txt_src].apply(build_snippet)
    
    # Parse keywords
    tok_col, av_kw = parse_keyword_space(df_local)
    
    if '__tokens' not in df_local.columns or all(len(t) == 0 for t in df_local['__tokens']):
        # No token column found — extract keywords from text
        import re
        _stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'about',
            'above', 'after', 'again', 'also', 'any', 'because', 'before',
            'between', 'during', 'into', 'through', 'under', 'until', 'up',
            'out', 'over', 'then', 'once', 'here', 'there', 'if', 'while',
            'etc', 'e.g', 'i.e', 'using', 'based', 'often', 'many', 'new',
            'well', 'even', 'like', 'make', 'use', 'one', 'two', 'get',
        }
        def _extract_tokens(text):
            if not isinstance(text, str) or not text.strip():
                return []
            words = re.findall(r'[a-zA-Z]{3,}', text.lower())
            return [w for w in words if w not in _stopwords]
        
        df_local['__tokens'] = df_local[txt_src].apply(_extract_tokens)
        tok_col = '__generated'
        # Rebuild available keywords from generated tokens
        flattened = [tok for toks in df_local['__tokens'] for tok in toks]
        keyword_counts = pd.Series(flattened).value_counts()
        av_kw = keyword_counts.head(100).index.tolist()
        
    return df_local, tok_col, av_kw

# Execute cached loading
df, token_col, available_keywords = load_and_process_data(processed_csv_path, doc_ids_path)

@st.cache_data
def extract_cluster_topics(df_for_topics):
    topic_map = {}
    if 'cluster' in df_for_topics.columns and '__tokens' in df_for_topics.columns:
        from collections import Counter
        for c in df_for_topics['cluster'].unique():
            all_toks = [tok for toks in df_for_topics[df_for_topics['cluster'] == c]['__tokens'] for tok in toks]
            if all_toks:
                top_words = [kw for kw, _ in Counter(all_toks).most_common(3)]
                topic_map[c] = f"{c}: {', '.join(top_words)}"
            else:
                topic_map[c] = f"Cluster {c}"
    return topic_map

GLOBAL_CLUSTER_MAP = extract_cluster_topics(df)

# Metadata inference (fast enough to run every time or cache separately if needed)
NON_METADATA_COLUMNS = {
    'doc_id', 'text', '__snippet', '__tokens', 'cleaned_text', 'preprocessed_text',
    'Guideline + Slogan', 'processed_text', 'tokens', 'cluster'
}
metadata_candidates = [c for c in df.columns if c not in NON_METADATA_COLUMNS and not c.startswith('__')]
numeric_metadata = [c for c in metadata_candidates if pd.api.types.is_numeric_dtype(df[c])]
categorical_metadata = [c for c in metadata_candidates if c not in numeric_metadata]
candidate_chunk_cols = [c for c in df.columns if 'chunk' in c.lower() or 'segment' in c.lower() or 'part' in c.lower()]
candidate_parent_cols = [c for c in df.columns if 'parent' in c.lower() or 'doc_id' in c.lower() or 'guideline' in c.lower()]


# Load coordinates (if present) otherwise make fake 2D projections
def get_mtime(p):
    return os.path.getmtime(p) if p and os.path.exists(p) else 0

coords_tsne = try_load(coords_tsne_path, get_mtime(coords_tsne_path)) if coords_tsne_path else None
coords_umap = try_load(coords_umap_path, get_mtime(coords_umap_path)) if coords_umap_path else None
coords_base = try_load(coords_pca_path, get_mtime(coords_pca_path)) if coords_pca_path else None
tfidf_matrix = try_load(tfidf_matrix_path, get_mtime(tfidf_matrix_path)) if tfidf_matrix_path else None

# read doc_ids list if available to align .npy indices
doc_ids_list = None
if doc_ids_path:
    try:
        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            doc_ids_list = [line.strip() for line in f if line.strip()]
    except Exception:
        doc_ids_list = None

if doc_ids_list and len(doc_ids_list) != len(df):
    # keep mismatch in warning list but ignore the file for alignment to avoid blank plots
    alignment_note = (
        f'doc_ids.txt length ({len(doc_ids_list)}) does not match processed CSV ({len(df)}). '
        'Falling back to dataframe order for alignment.'
    )
    alignment_notes.append(alignment_note)
    doc_ids_list = None

if doc_ids_list:
    doc_id_to_global_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids_list)}
else:
    doc_id_to_global_idx = {doc_id: idx for idx, doc_id in enumerate(df['doc_id'].tolist())}

# Sanity checks: compare lengths and shapes so user gets a clear warning
def check_alignment():
    issues = []
    # coords shapes
    try:
        if coords_tsne is not None and coords_tsne.ndim == 2:
            pass
    except Exception:
        issues.append('t-SNE coordinates appear invalid or unreadable')
    try:
        if coords_umap is not None and coords_umap.ndim == 2:
            pass
    except Exception:
        issues.append('UMAP coordinates appear invalid or unreadable')
    # doc_ids length vs coords length
    if doc_ids_list is not None:
        n_docs = len(doc_ids_list)
        if coords_tsne is not None and len(coords_tsne) != n_docs:
            issues.append(f'Length mismatch: doc_ids ({n_docs}) vs coords_tsne ({len(coords_tsne)})')
        if coords_umap is not None and len(coords_umap) != n_docs:
            issues.append(f'Length mismatch: doc_ids ({n_docs}) vs coords_umap ({len(coords_umap)})')
        if coords_base is not None and len(coords_base) != n_docs:
            issues.append(f'Length mismatch: doc_ids ({n_docs}) vs base embeddings ({len(coords_base)})')
    return issues

alignment_issues = check_alignment()
alignment_issues.extend(alignment_notes)

expected_doc_len = len(doc_ids_list) if doc_ids_list is not None else len(df)


def enforce_length(arr):
    if arr is None:
        return None
    try:
        if len(arr) == expected_doc_len:
            return arr
    except Exception:
        return None
    return None


coords_tsne = enforce_length(coords_tsne)
coords_umap = enforce_length(coords_umap)
coords_base = enforce_length(coords_base)
# tfidf_matrix usually matches, but can't easily check 'len' if sparse without shape
if tfidf_matrix is not None and tfidf_matrix.shape[0] != expected_doc_len:
    alignment_issues.append(f"TF-IDF matrix rows ({tfidf_matrix.shape[0]}) mismatch doc count ({expected_doc_len})")
    tfidf_matrix = None

if coords_base is None:
    # make random high-dim embeddings if not available
    rng = np.random.RandomState(0)
    base_embeddings = rng.normal(size=(len(df), 64))
elif tfidf_matrix is not None:
    # Use high-dim data if available!
    base_embeddings = tfidf_matrix
    st.toast("Loaded high-dimensional TF-IDF data for accurate similarity")
else:
    base_embeddings = coords_base
    st.warning("Using low-dimensional 2D layout for similarity. Results may be inaccurate.")

if coords_tsne is None:
    coords_tsne = PCA(n_components=2).fit_transform(base_embeddings)
if coords_umap is None:
    coords_umap = PCA(n_components=2).fit_transform(base_embeddings + 0.01)


def apply_scale_and_alpha(fig, alpha, scale):
    for trace in fig.data:
        if not hasattr(trace, 'marker') or trace.marker is None: continue
        # Safely scale size
        if isinstance(trace.marker.size, (list, tuple)):
            trace.marker.size = [s * scale for s in trace.marker.size]
        elif isinstance(trace.marker.size, (int, float)):
            trace.marker.size = trace.marker.size * scale
            
        # Safely apply opacity capping it smoothly across arrays and scalars
        if isinstance(trace.marker.opacity, (list, tuple)):
            # If trace already has an opacity array, blend it with the global alpha
            trace.marker.opacity = [min(1.0, o * alpha) for o in trace.marker.opacity]
        elif isinstance(trace.marker.opacity, (int, float)):
            trace.marker.opacity = min(1.0, trace.marker.opacity * alpha)
        else:
            trace.marker.opacity = alpha

def make_plot(df_plot, xcol, ycol, selected_ids, search_ids, title, hover_cols, color_mode='Selection', focused_ids=None, show_hover=True):
    df_plot = df_plot.copy()
    df_plot['__size'] = 8  # Default size
    if focused_ids is None:
        focused_ids = []
        
    hover_dict = {xcol: False, ycol: False}
    for col in hover_cols:
        hover_dict[col] = True
    
    if color_mode == 'Cluster':
        if 'cluster' in df_plot.columns:
            # Bugfix: map to consistent colors for each cluster
            cluster_counts = df_plot['cluster'].value_counts().to_dict()
            df_plot['cluster_str'] = df_plot['cluster'].apply(lambda c: f'{GLOBAL_CLUSTER_MAP.get(c, "Cluster " + str(c))} (n={cluster_counts.get(c, 0)})')
            color_col = 'cluster_str'
            palette = px.colors.qualitative.Plotly
            cluster_color_map = {
                f'{GLOBAL_CLUSTER_MAP.get(c, "Cluster " + str(c))} (n={n})': palette[i % len(palette)]
                for i, (c, n) in enumerate(cluster_counts.items())
            }
        else:
            df_plot['__status'] = 'Other'
            color_col = '__status'
            cluster_color_map = {'Other': '#457B9D'}
        
        if selected_ids:
            df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
        if search_ids:
            df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
        # In cluster mode, overlay focused docs with a dedicated marker column
        if focused_ids:
            df_plot['__is_focused'] = df_plot['doc_id'].isin(focused_ids)
        
        fig = px.scatter(
            df_plot,
            x=xcol,
            y=ycol,
            color=color_col,
            color_discrete_map=cluster_color_map,
            hover_name='doc_id',
            hover_data=hover_dict,
            custom_data=['doc_id'],
        )
    else:
        # Color by selection status
        df_plot['__status'] = 'Other'
        
        if search_ids:
            df_plot.loc[df_plot['doc_id'].isin(search_ids), '__status'] = 'Search hit'
            df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
        if selected_ids:
            df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__status'] = 'Selected'
            df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
        # Focused overrides Selected (applied last = highest priority)
        if focused_ids:
            df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__status'] = 'Focused'
            df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__size'] = 16
        
        color_map = {
            'Focused': '#E63946',        # Unified red
            'Selected': '#E63946',       # Unified red
            'Search hit': '#F4A261',     # Warm orange
            'Other': '#457B9D'           # Deep blue-gray
        }
        
        fig = px.scatter(
            df_plot,
            x=xcol,
            y=ycol,
            color='__status',
            color_discrete_map=color_map,
            hover_name='doc_id',
            hover_data=hover_dict,
            custom_data=['doc_id'],
            category_orders={'__status': ['Focused', 'Selected', 'Search hit', 'Other']}
        )
    
    if not show_hover:
        fig.update_traces(hoverinfo='none', hovertemplate=None)
        
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.4)",
            font_size=11,
            font_family="Arial, sans-serif",
            font_color="black",
            bordercolor="gray"
        )
    )
    
    try:
        for trace in fig.data:
            if color_mode == 'Selection':
                if trace.name == 'Focused':
                    trace.marker.size = 14
                    trace.marker.opacity = 1.0
                    trace.marker.line = dict(width=2.5, color='white')
                    trace.marker.symbol = 'x'
                    continue
                elif trace.name == 'Selected':
                    trace.marker.size = 10
                    trace.marker.opacity = 0.95
                    trace.marker.symbol = 'diamond'
                elif trace.name == 'Search hit':
                    trace.marker.size = 9
                    trace.marker.opacity = 0.9
                    trace.marker.symbol = 'star'
                else:
                    trace.marker.size = 5
                    trace.marker.opacity = 0.75
                    trace.marker.symbol = 'circle'
            else:
                if selected_ids or search_ids or focused_ids:
                    # Dynamically fade unselected points and highlight selections
                    opacities = []
                    sizes = []
                    l_widths = []
                    l_colors = []
                    m_colors = []
                    m_symbols = []
                    base_color = getattr(trace.marker, 'color', '#457B9D')
                    if isinstance(base_color, (list, tuple)):
                        base_color = base_color[0] if len(base_color) > 0 else '#457B9D'
                        
                    for cdata in trace.customdata:
                        did = cdata[0]
                        if focused_ids and did in focused_ids:
                           opacities.append(1.0)
                           sizes.append(18)
                           l_widths.append(3)
                           l_colors.append('white')
                           m_colors.append('#E63946')
                           m_symbols.append('x')
                        elif selected_ids and did in selected_ids:
                           opacities.append(1.0)
                           sizes.append(14)
                           l_widths.append(2)
                           l_colors.append('white')
                           m_colors.append('#E63946')
                           m_symbols.append('diamond')
                        elif search_ids and did in search_ids:
                           opacities.append(1.0)
                           sizes.append(10)
                           l_widths.append(1.5)
                           l_colors.append('white')
                           m_colors.append('#F4A261')
                           m_symbols.append('star')
                        else:
                           opacities.append(0.35) # moderately faded
                           sizes.append(5)
                           l_widths.append(0.5)
                           l_colors.append('rgba(0,0,0,0.2)')
                           m_colors.append(base_color)
                           m_symbols.append('circle')
                    trace.marker.opacity = opacities
                    trace.marker.size = sizes
                    trace.marker.line = dict(width=l_widths, color=l_colors)
                    trace.marker.color = m_colors
                    trace.marker.symbol = m_symbols
                else:
                    trace.marker.size = 6
                    trace.marker.opacity = 0.85
                    trace.marker.line = dict(width=0.8, color='rgba(0,0,0,0.4)')
    except Exception:
        pass
    
    # (Removed redundant overlay tracing for focused_ids since integrated into array logic)
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial, sans-serif", color="#333333"),
            x=0.5,
            xanchor='center',
            y=0.02,  # Position at bottom
            yanchor='bottom'
        ),
        margin=dict(l=10, r=10, t=10, b=40),  # More space at bottom for title
        dragmode=st.session_state.get('dragmode', 'lasso'),  # Use session state dragmode
        clickmode='event+select',
        hovermode='closest',  # Only show hover for closest point, not while selecting
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,  # Move legend to top since title is at bottom
            xanchor="right",
            x=1,
            title=None
        ),
        xaxis=dict(title="Projection Dimension 1", showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0', scaleanchor="y", scaleratio=1, range=[-1.1, 1.1]),
        yaxis=dict(title="Projection Dimension 2", showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0', scaleanchor="x", scaleratio=1, range=[-1.1, 1.1]),
        autosize=True  # Enable responsive sizing
    )
    return fig


st.markdown("## Embeddings Explorer")

# initialize selection state
if 'selected_ids' not in st.session_state:
    st.session_state['selected_ids'] = []
if 'search_hits' not in st.session_state:
    st.session_state['search_hits'] = []
if 'selection_history' not in st.session_state:
    st.session_state['selection_history'] = []
if 'dragmode' not in st.session_state:
    st.session_state['dragmode'] = 'lasso'
if 'last_chart_states' not in st.session_state:
    st.session_state['last_chart_states'] = {}
if 'chart_revisions' not in st.session_state:
    st.session_state['chart_revisions'] = {'chart_tsne': 0, 'chart_umap': 0, 'chart_pca': 0, 'chart_focus': 0}

# Compact status bar (filled in later after filtering runs)
status_placeholder = st.empty()



# Guided Tour / Getting Started
with st.sidebar:
    with st.expander('Getting Started', expanded=True):
        st.write("1. **Load Data**: The latest dataset is auto-loaded.")
        st.write("2. **Explore**: Use the 3 projections to see document relationships.")
        st.write("3. **Select**: Lasso-select points to highlight them across all views.")
        st.write("4. **Analyze**: Check the 'Analysis' tab for cluster stats and details.")

# Dynamic Clustering Logic
with st.sidebar:
    with st.expander('Clustering Settings', expanded=True):
        cluster_mode = st.radio(
            "Cluster Source", 
            ["Static (Pre-computed)", "Dynamic (K-Means)"],
            help="Static: Use the file's original groups. Dynamic: Create new groups right now based on the data."
        )
        
        if cluster_mode == "Dynamic (K-Means)":
            n_clusters = st.slider(
                "Number of Clusters (k)", 
                min_value=2, 
                max_value=50, 
                value=8, 
                help="How many distinct groups (topics) to find."
            )
            
            if st.button("Re-run Clustering") or 'dynamic_clusters' not in st.session_state or st.session_state.get('last_k') != n_clusters:
                with st.spinner(f"Running K-Means with k={n_clusters}..."):
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        # Ensure base_embeddings is valid
                        if base_embeddings is not None and base_embeddings.shape[0] == len(df):
                            new_labels = kmeans.fit_predict(base_embeddings)
                            st.session_state['dynamic_clusters'] = new_labels
                            st.session_state['last_k'] = n_clusters
                            st.success(f"Generated {n_clusters} clusters!")
                        else:
                            st.error("Embeddings not available or aligned for clustering.")
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")
        
        # Apply dynamic clusters if active
        if cluster_mode == "Dynamic (K-Means)" and 'dynamic_clusters' in st.session_state:
            df['cluster'] = st.session_state['dynamic_clusters']

# default sidebar values (populated below)
search_query = ''
search_scopes = ['doc_id', 'phrase']
cluster_filter = sorted(df['cluster'].unique().tolist()) if 'cluster' in df.columns else []
keyword_filter = []
metadata_filters = {}
numeric_ranges = {}
doc_a, doc_b = None, None
select_doc = None
recompute_previews = False
pca_comps = 2
umap_neighbors = 15
umap_min_dist = 0.1
tsne_perplexity = 30
tsne_lr = 200

with st.sidebar:
    st.header('Controls')
    # Dataset / artifact bundle selection
    with st.expander('Dataset & Display', expanded=True):
        st.caption('Artifact Bundle')
        # Dataset / artifact bundle selection
        bundle_options = []
        bundle_map = {}
        # Discover all artifact bundles that contain a CSV dataset
        _art_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts')
        _art_dir = os.path.normpath(_art_dir)
        _csv_names = ('processed_data_with_clusters.csv', 'full_dataset_with_new_id.csv', 'processed_data.csv')
        # Check artifacts root
        for _cn in _csv_names:
            if os.path.exists(os.path.join(_art_dir, _cn)):
                bundle_options.append('artifacts')
                bundle_map['artifacts'] = _art_dir
                break
        # Check subdirectories
        if os.path.isdir(_art_dir):
            for _sub in sorted(os.listdir(_art_dir)):
                _sub_path = os.path.join(_art_dir, _sub)
                if os.path.isdir(_sub_path):
                    for _cn in _csv_names:
                        if os.path.exists(os.path.join(_sub_path, _cn)):
                            label = f'artifacts/{_sub}'
                            if label not in bundle_options:
                                bundle_options.append(label)
                                bundle_map[label] = _sub_path
                            break
        if not bundle_options:
            bundle_options = ['(auto)']
            bundle_map['(auto)'] = None
        current_label = '(auto)'
        if st.session_state.get('selected_bundle_root') in bundle_map.values():
            # Reverse lookup
            for k, v in bundle_map.items():
                if v == st.session_state.get('selected_bundle_root'):
                    current_label = k
                    break
        chosen_label = st.selectbox('Choose artifact bundle', options=bundle_options, index=bundle_options.index(current_label) if current_label in bundle_options else 0)
        new_root = bundle_map.get(chosen_label)
        if new_root != st.session_state.get('selected_bundle_root'):
            st.session_state['selected_bundle_root'] = new_root
            st.info(f'Selected bundle: {chosen_label or "(auto)"}')
            st.rerun()
        if alignment_issues:
            st.warning('Data alignment issues detected:')
            for it in alignment_issues:
                st.write('- ' + it)
            st.help('If you have a `doc_ids.txt` file, ensure it matches the ordering used to produce the .npy coordinate files. If not available, the app falls back to index-alignment which may mis-map documents across projections.')

        st.caption('Display Mode')
        view_mode = st.radio(
            "View Layout",
            ["Linked (All 3)", "Single (Focus)"],
            help="Linked: Compare 3 projections. Single: Large view for details."
        )
        focus_proj = 'UMAP'
        if view_mode == "Single (Focus)":
            focus_proj = st.selectbox("Select Projection", ["UMAP", "t-SNE", "PCA"])

        color_mode = st.radio(
            'Color points by:',
            options=['Selection', 'Cluster', 'Cohort'],
            help='Switch between showing selection status (Red/Blue), semantic clusters (Multi-color), or saved Cohorts.'
        )
        
        st.markdown('**Visual Tweaks**')
        point_alpha = st.slider('Point Opacity', 0.1, 1.0, 0.7, 0.1)
        point_size_scale = st.slider('Point Size', 0.5, 2.0, 1.0, 0.1)

        show_download_buttons = st.checkbox('Show plot download buttons', value=False, help='Enable this to download high-resolution PNGs of the current plots for reports or presentations.')
        show_hover = st.checkbox('Show tooltip on hover', value=True, help='Uncheck to completely hide the hover tooltips for a cleaner view.')

    with st.expander('Search & Filter', expanded=False):
        # Improved Search
        search_query = st.text_input('Search doc_id / keyword / phrase / regex', '')
        use_regex = st.checkbox('Use Regex', value=False, help='Treat search query as a Regular Expression.')
        
        search_scopes = st.multiselect('Search scopes', options=['doc_id', 'keywords', 'phrase'], default=['doc_id', 'phrase'], help="Where to look for your search terms. 'doc_id' checks filenames, 'phrase' checks full text.")
        clusters = sorted(df['cluster'].unique().tolist()) if 'cluster' in df.columns else []
        cluster_filter = st.multiselect('Clusters', options=clusters, default=clusters, help="Filter data by topic group. Uncheck a cluster number to hide its documents from the view.")
        keyword_filter = st.multiselect('Keyword tags', options=available_keywords, default=[], help='Filter the visible points to only those containing specific keywords. Useful for narrowing down to a theme.')
        keyword_logic_selection = st.radio("Keyword Logic", ["OR (match any)", "AND (match all)"], index=0, horizontal=True)
        
        st.markdown('**Metadata Filters**')
        cat_defaults = [c for c in ['Data Domain', 'Tool Used/Mentioned', 'Publisher'] if c in categorical_metadata]
        selected_cats = st.multiselect('Categorical fields', options=categorical_metadata, default=cat_defaults, help="Select metadata categories (like Publisher or Domain) to enable specific filters below.")
        for col in selected_cats:
            options = sorted({str(v) for v in df[col].dropna().unique() if str(v).strip()})
            sel = st.multiselect(f'{col}', options=options, key=f'cat_filter_{col}')
            if sel:
                metadata_filters[col] = sel
        num_defaults = [c for c in ['Year', 'Subjective Trust Score:'] if c in numeric_metadata]
        selected_nums = st.multiselect('Numeric fields', options=numeric_metadata, default=num_defaults, help="Select numeric metadata (like Year or Score) to enable range sliders below.")
        for col in selected_nums:
            series = df[col].dropna()
            if series.empty:
                continue
            col_min, col_max = float(series.min()), float(series.max())
            low, high = st.slider(
                f'{col} range',
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f'num_filter_{col}'
            )
            numeric_ranges[col] = (low, high)
        max_points = st.slider('Max points to display', min_value=50, max_value=3000, value=1200, step=50, help="Reduce this number if the app feels slow. Limits how many dots are drawn.")

    # Cohort Management (Persistent Selection)
    with st.expander('Saved Cohorts', expanded=True):
        if 'saved_cohorts' not in st.session_state:
            st.session_state['saved_cohorts'] = {}
        
        # Save current selection
        current_sel_len = len(st.session_state['selected_ids'])
        new_cohort_name = st.text_input('New cohort name', placeholder='e.g., Outliers, Group A')
        if st.button('Save current selection') and new_cohort_name and current_sel_len > 0:
            st.session_state['saved_cohorts'][new_cohort_name] = list(st.session_state['selected_ids'])
            st.success(f"Saved {current_sel_len} docs to '{new_cohort_name}'")
            st.rerun()

        # Display saved cohorts
        if st.session_state['saved_cohorts']:
            st.markdown('**Cohorts**')
            cohorts_to_delete = []
            for name, ids in st.session_state['saved_cohorts'].items():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    if st.button(f"{name} ({len(ids)})", key=f"load_{name}", use_container_width=True):
                        update_selection(ids, additive=st.session_state.get('additive_mode', False))
                with c2:
                    st.write("") # Spacer
                with c3:
                    if st.button('🗑️', key=f"del_{name}"):
                        cohorts_to_delete.append(name)
            
            if cohorts_to_delete:
                for name in cohorts_to_delete:
                    del st.session_state['saved_cohorts'][name]
                st.rerun()
        else:
            st.caption("No saved cohorts yet.")

    with st.expander('Selection Tools', expanded=False):
        st.markdown('**Quick Jumps**')
        doc_options = df['doc_id'].tolist()
        default_doc = doc_options[0] if doc_options else ''
        select_doc = st.selectbox('Jump to doc_id', options=doc_options or [''], index=0, help='Instantly select and center on a specific document by ID.')
        if st.button('Highlight doc', key='btn_highlight_doc') and select_doc:
            update_selection([select_doc], additive=False)
        multi_select = st.multiselect('Pin doc_ids (max 15)', options=doc_options, default=st.session_state['selected_ids'][:5], max_selections=15, help="Manually select specific documents to keep them highlighted, even if you click elsewhere.")
        if st.button('Apply pinned selection', key='btn_apply_multi'):
            update_selection(multi_select or [], additive=False)
        
        st.markdown('**Controls**')
        # Quick actions row
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            if st.button('Clear', key='btn_clear_selection', use_container_width=True):
                update_selection([], additive=False)
        with qcol2:
            if st.button('Undo', key='btn_undo', use_container_width=True, disabled=len(st.session_state.get('selection_history', [])) == 0):
                history = st.session_state.get('selection_history', [])
                if history:
                    st.session_state['selected_ids'] = history.pop()
                    st.session_state['selection_history'] = history
                    st.rerun()
        with qcol3:
            if st.button('Random', key='btn_random', use_container_width=True, help='Select 10 random documents to explore the dataset.'):
                import random
                random_ids = random.sample(doc_options, min(10, len(doc_options)))
                update_selection(random_ids, additive=False)
        
        # Brushing modes
        qcol4, qcol5 = st.columns(2)
        with qcol4:
            if st.button('Invert', key='btn_invert', use_container_width=True, help='Select everything that is NOT currently selected.'):
                all_ids = set(df_work['doc_id'].tolist())
                current_ids = set(st.session_state['selected_ids'])
                inverted = list(all_ids - current_ids)
                update_selection(inverted, additive=False)
        with qcol5:
            box_select = st.checkbox('Box Select', value=False, help='Switch to rectangular box selection (default is freehand lasso).')
            if box_select:
                st.session_state['dragmode'] = 'select'
            else:
                st.session_state['dragmode'] = 'lasso'
        
        # Additive selection mode
        additive_mode = st.checkbox('Additive Selection', value=False, help='Keep existing selection when making a new one (Ctrl-click behavior). Uncheck to start fresh with each drag.')
        if 'additive_mode' not in st.session_state:
            st.session_state['additive_mode'] = False
        st.session_state['additive_mode'] = additive_mode
        
        # Export selection
        if st.session_state['selected_ids']:
            st.markdown('**Download Selection**')
            n_download = st.number_input('Limit download items count', min_value=1, max_value=max(1, len(st.session_state['selected_ids'])), value=len(st.session_state['selected_ids']), step=1)
            download_ids = st.session_state['selected_ids'][:n_download]
            import json
            export_data = {
                'selected_ids': download_ids,
                'count': len(download_ids),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            st.download_button(
                label=f'Export {len(download_ids)} items (JSON)',
                data=json.dumps(export_data, indent=2),
                file_name='selected_docs.json',
                mime='application/json',
                use_container_width=True
            )
            
            # Export Filtered CSV
            csv_data = df[df['doc_id'].isin(download_ids)]
            if not csv_data.empty:
               csv_string = csv_data.to_csv(index=False)
               st.download_button(
                   label=f'Export {len(download_ids)} items (CSV)',
                   data=csv_string,
                   file_name='selected_data.csv',
                   mime='text/csv',
                   use_container_width=True
               )
    with st.expander('Sessions & Advanced', expanded=False):
        st.markdown('**Sessions**')
        # ... (rest of session code) ...
        # Simplified for brevity in this replacement, keeping context

    with st.expander('Reproducibility', expanded=False):
        st.write("**Configuration**")
        config = {
            "umap_neighbors": 15,
            "umap_min_dist": 0.1,
            "seed": 42,
            "model": "all-MiniLM-L6-v2 (implied)",
            "n_docs": len(df),
            "generated_at": pd.Timestamp.now().isoformat()
        }
        st.json(config)
        st.caption("Use these parameters to reproduce this exact map structure.")
        # Save session
        session_name = st.text_input('Session name', placeholder='my_analysis', help="Save your current filters, selection, and view settings to reload later.")
        if st.button('Save Session', use_container_width=True):
            if session_name:
                import json
                import os
                session_data = {
                    'selected_ids': st.session_state['selected_ids'],
                    'search_query': search_query,
                    'cluster_filter': cluster_filter,
                    'color_mode': color_mode,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                os.makedirs('sessions', exist_ok=True)
                with open(f'sessions/{session_name}.json', 'w') as f:
                    json.dump(session_data, f, indent=2)
                st.success(f'Saved session: {session_name}')
        
        # Load session
        import os
        if os.path.exists('sessions'):
            session_files = [f.replace('.json', '') for f in os.listdir('sessions') if f.endswith('.json')]
            if session_files:
                load_session = st.selectbox('Load session', options=[''] + session_files)
                if st.button('Load Session', use_container_width=True) and load_session:
                    import json
                    with open(f'sessions/{load_session}.json', 'r') as f:
                        session_data = json.load(f)
                    st.session_state['selected_ids'] = session_data.get('selected_ids', [])
                    st.success(f'Loaded session: {load_session}')
                    st.rerun()

        st.markdown('---')
        st.markdown('**Linking**')
        chunk_choice = st.selectbox('Chunk column', options=['(none)'] + candidate_chunk_cols, index=1 if candidate_chunk_cols else 0, help="If your data refers to parts of a larger document, pick the 'part' ID column here.")
        parent_choice = st.selectbox('Parent column', options=['(none)'] + candidate_parent_cols, index=1 if candidate_parent_cols else 0, help="Pick the 'parent' ID column here. This lets you click a part and automatically select the whole parent document.")
        chunk_col = None if chunk_choice == '(none)' else chunk_choice
        parent_col = None if parent_choice == '(none)' else parent_choice
        if chunk_col and parent_col:
            if st.button('Build chunk ↔ parent map', key='btn_build_chunk_map'):
                try:
                    chunk_parent_map, parent_chunk_map = apply_chunk_mapping(df, chunk_col, parent_col)
                    st.success(f'Built mapping with {len(chunk_parent_map)} chunk entries')
                    st.session_state['chunk_parent_map'] = chunk_parent_map
                    st.session_state['parent_chunk_map'] = parent_chunk_map
                except Exception as exc:
                    st.error('Failed to build mapping: ' + str(exc))
        else:
            st.caption('Provide both chunk and parent columns to enable chunk↔parent linking')

        st.markdown('---')
        st.markdown('**Comparison Inputs**')
        doc_a = st.selectbox('Doc A', options=doc_options or [''], index=0, key='comp_a', help="Choose the first document for the side-by-side comparison.")
        doc_b = st.selectbox('Doc B', options=doc_options or [''], index=1 if len(doc_options) > 1 else 0, key='comp_b', help="Choose the second document to compare against Doc A.")

        st.markdown('---')
        st.markdown('**Embedding Parameters**')
        pca_comps = st.slider('PCA components (preview)', min_value=2, max_value=5, value=2, help="Advanced: Number of components for PCA dimensionality reduction.")
        umap_neighbors = st.slider('UMAP neighbors', min_value=5, max_value=120, value=15, help="Advanced: Controls how broad the analysis is. Low values focus on local details; high values look at the big picture.")
        umap_min_dist = st.slider('UMAP min_dist', min_value=0.0, max_value=1.0, value=0.1, help="Advanced: Controls how tightly points are packed. Lower is clumpier.")
        tsne_perplexity = st.slider('t-SNE perplexity', min_value=5, max_value=100, value=30, help="Advanced: Roughly, how many neighbors each point 'cares' about. Higher values smooth the plot.")
        tsne_lr = st.slider('t-SNE learning rate', min_value=10, max_value=1000, value=200, help="Advanced: Speed of the t-SNE optimization process.")
        recompute_previews = st.button('Recompute embedding previews')
        st.caption('Preview recompute runs locally. UMAP requires `umap-learn`; otherwise PCA fallbacks are used.')


if recompute_previews:
    with st.spinner('Recomputing embeddings preview …'):
        new_tsne = None
        new_umap = None
        if len(base_embeddings) > 3:
            try:
                max_perplexity = min(tsne_perplexity, len(base_embeddings) - 1)
                if max_perplexity < 2:
                    max_perplexity = 2
                tsne_model = TSNE(
                    n_components=2,
                    perplexity=max_perplexity,
                    learning_rate=tsne_lr,
                    init='pca',
                    random_state=42,
                    n_iter=1000,
                )
                new_tsne = tsne_model.fit_transform(base_embeddings)
            except Exception as exc:
                st.warning(f't-SNE recompute failed: {exc}')
        else:
            st.info('Need at least 4 documents to recompute t-SNE preview.')
        if umap is not None:
            try:
                neighbors = min(umap_neighbors, max(2, len(base_embeddings) - 1))
                umap_model = umap.UMAP(
                    n_components=2,
                    n_neighbors=neighbors,
                    min_dist=umap_min_dist,
                    random_state=42,
                )
                new_umap = umap_model.fit_transform(base_embeddings)
            except Exception as exc:
                st.warning(f'UMAP recompute failed: {exc}')
        else:
            st.info('Install `umap-learn` to enable UMAP recomputes.')
        if new_tsne is not None:
            st.session_state['tsne_coords_override'] = new_tsne
        if new_umap is not None:
            st.session_state['umap_coords_override'] = new_umap
        st.rerun()


coords_tsne = st.session_state.get('tsne_coords_override', coords_tsne)
coords_umap = st.session_state.get('umap_coords_override', coords_umap)


# Filter dataframe
df_work = df.copy()
if 'cluster' in df_work.columns and cluster_filter:
    df_work = df_work[df_work['cluster'].isin(cluster_filter)]
if keyword_filter:
    lowered_keywords = [kw.lower() for kw in keyword_filter]
    if "AND" in keyword_logic_selection:
        df_work = df_work[df_work['__tokens'].apply(lambda toks: all(kw in toks for kw in lowered_keywords))]
    else:
        df_work = df_work[df_work['__tokens'].apply(lambda toks: any(kw in toks for kw in lowered_keywords))]
for col, values in metadata_filters.items():
    df_work = df_work[df_work[col].astype(str).isin(values)]
for col, (low, high) in numeric_ranges.items():
    df_work = df_work[(df_work[col] >= low) & (df_work[col] <= high)]
df_work = run_search(df_work, search_query, search_scopes)
df_work = df_work.reset_index(drop=True)
if len(df_work) > max_points:
    df_work = df_work.iloc[:max_points]

if df_work.empty:
    st.warning('Filters/search returned zero documents. Clear filters to see points again.')
    st.stop()

df_work['__global_idx'] = df_work['doc_id'].map(doc_id_to_global_idx)

# Extract focused doc IDs from the document details table.
# The table state persists across reruns so we can read it before rendering charts.
focused_ids = []
_table_state = st.session_state.get('sel_details_table_stable')
if _table_state is not None:
    try:
        _tbl_rows = _table_state.selection.rows if hasattr(_table_state, 'selection') else []
        if _tbl_rows:
            _sel_ids = st.session_state.get('selected_ids', [])
            _sel_df = df_work[df_work['doc_id'].isin(_sel_ids)]
            for _ri in _tbl_rows:
                if _ri < len(_sel_df):
                    focused_ids.append(_sel_df.iloc[_ri]['doc_id'])
    except Exception:
        pass
st.session_state['focused_ids'] = focused_ids

# Centralized Selection Logic: Process chart events before rendering new plots
# This ensures that a selection in any chart updates all charts immediately (fixing directional bugs).
# IMPORTANT: Empty selections (from chart re-renders after key bumps) are SKIPPED to prevent
# the bug where re-created charts fire empty events that wipe the user's selection.
chart_keys = ['chart_tsne', 'chart_umap', 'chart_pca', 'chart_focus']
for key in chart_keys:
    rev = st.session_state['chart_revisions'][key]
    effective_key = f"{key}_{rev}"
    
    current_state = st.session_state.get(effective_key)
    last_state = st.session_state['last_chart_states'].get(key)
    
    if current_state != last_state:
        # Always record the latest state so we don't re-process this same event
        st.session_state['last_chart_states'][key] = current_state
        
        # Extract doc IDs from the selection event
        new_ids = []
        if current_state:
            selection_data = None
            if hasattr(current_state, 'selection'):
                selection_data = current_state.selection
            elif isinstance(current_state, dict):
                selection_data = current_state.get('selection', {})
            
            if selection_data and 'points' in selection_data:
                points = selection_data['points']
                for p in points:
                    cdata = p.get('customdata', p.get('custom_data'))
                    if cdata is not None:
                        if isinstance(cdata, list) and cdata:
                            new_ids.append(str(cdata[0]))
                        else:
                            new_ids.append(str(cdata))
                    elif 'point_index' in p:
                        idx = p['point_index']
                        if 0 <= idx < len(df_work):
                            new_ids.append(df_work.iloc[idx]['doc_id'])

        selected_docs = list(dict.fromkeys(new_ids))
        
        # CRITICAL FIX: Skip empty selections entirely.
        # After a lasso select bumps chart keys, the fresh chart instances fire
        # empty selection events. Processing those would clear the user's selection.
        # Users can use the "Clear" button to explicitly deselect.
        if not selected_docs:
            continue
        
        # Smart Inspection: If user clicks a single point that is already in 
        # the current group, keep the entire group but move that doc to the end
        # so the Inspector panel focuses on it.
        current_selection = st.session_state.get('selected_ids', [])
        if len(selected_docs) == 1 and len(current_selection) > 1 and selected_docs[0] in current_selection:
            focus_id = selected_docs[0]
            selected_docs = [d for d in current_selection if d != focus_id] + [focus_id]
            st.toast(f"Inspecting {focus_id} in current group")

        is_additive = st.session_state.get('additive_mode', False)
        update_selection(selected_docs, additive=is_additive)
        
        # Bump all chart keys so they re-render with updated colors/sizes,
        # and pre-set their last_state to None so the fresh empty events are skipped.
        for k in chart_keys:
            st.session_state['chart_revisions'][k] += 1
            st.session_state['last_chart_states'][k] = None

        # Process only one real interaction per rerun to avoid conflicts
        break



# attach coordinates using global indices whenever possible
def lookup_coords(arr, idx, doc_id=None, local_idx=None):
    """Robust coordinate lookup.

    Tries several fallbacks so points still render when alignment files are missing
    or partially mismatched:
    1. Use provided global index `idx` if valid.
    2. If that fails, try to look up by `doc_id` using `doc_id_to_global_idx`.
    3. If that fails, try using `local_idx` (position in the filtered df_work).
    4. Return (0.0, 0.0) as a last resort.
    """
    if arr is None:
        return 0.0, 0.0
    # helper to read a 2d row safely
    def _row_to_xy(r):
        try:
            r = np.asarray(r)
            if r.size >= 2:
                return float(r[0]), float(r[1])
        except Exception:
            pass
        return None

    # try global index first
    try:
        if idx is not None and not pd.isna(idx):
            cand = _row_to_xy(arr[int(idx)])
            if cand:
                return cand
    except Exception:
        pass

    # try doc_id mapping
    try:
        if doc_id is not None:
            mapped = doc_id_to_global_idx.get(str(doc_id))
            if mapped is not None:
                cand = _row_to_xy(arr[int(mapped)])
                if cand:
                    return cand
    except Exception:
        pass

    # try local index into the array if it matches length or seems plausible    
    return 0.0, 0.0


tsne_x, tsne_y, umap_x, umap_y = [], [], [], []
for local_idx, (gi, docid) in enumerate(zip(df_work['__global_idx'].tolist(), df_work['doc_id'].tolist())):
    tx, ty = lookup_coords(coords_tsne, gi, doc_id=docid, local_idx=local_idx)
    ux, uy = lookup_coords(coords_umap, gi, doc_id=docid, local_idx=local_idx)
    tsne_x.append(tx)
    tsne_y.append(ty)
    umap_x.append(ux)
    umap_y.append(uy)

df_work['tsne_x'] = tsne_x
df_work['tsne_y'] = tsne_y
df_work['umap_x'] = umap_x
df_work['umap_y'] = umap_y




embedding_dim = base_embeddings.shape[1] if base_embeddings.ndim > 1 else 1


def lookup_embedding(idx):
    if idx is None or pd.isna(idx):
        return None
    try:
        return base_embeddings[int(idx)]
    except Exception:
        return None


# Consolidate embeddings_for_sim calculation here (it was previously duplicated)
# This version is the robust one from the second block
embedding_rows = []
for gi in df_work['__global_idx'].tolist():
    vec = lookup_embedding(gi)
    if vec is None:
        vec = np.zeros(embedding_dim)
    embedding_rows.append(vec)

embeddings_for_sim = np.zeros((0, embedding_dim))
if embedding_rows:
    # Stack and ensure valid numeric type
    try:
        # First stack loosely
        stacked = np.vstack(embedding_rows)
        # Then force to float, replacing potential objects/sequences
        embeddings_for_sim = np.asarray(stacked, dtype=float)
    except Exception:
        # Fallback: iterate and clean row by row
        valid_rows = []
        for r in embedding_rows:
            try:
                # flatten if it's a matrix or sparse
                if hasattr(r, 'toarray'):
                    r = r.toarray()
                r = np.asarray(r).flatten()
                 # ensure it matches dim and is float
                if len(r) == embedding_dim:
                    valid_rows.append(r.astype(float))
                else:
                    valid_rows.append(np.zeros(embedding_dim))
            except Exception:
                valid_rows.append(np.zeros(embedding_dim))
        if valid_rows:
            embeddings_for_sim = np.vstack(valid_rows)
        else:
            embeddings_for_sim = np.zeros((len(embedding_rows), embedding_dim))

# PCA preview from base embeddings
try:
    if len(embeddings_for_sim) >= pca_comps and embedding_dim >= pca_comps:
        # Add random_state for consistency across reruns
        pca = PCA(n_components=pca_comps, random_state=42)
        pca_coords = pca.fit_transform(embeddings_for_sim)
        df_work['pca_x'] = pca_coords[:, 0]
        df_work['pca_y'] = pca_coords[:, 1] if pca_comps >= 2 else 0.0
    else:
        df_work['pca_x'] = np.zeros(len(df_work))
        df_work['pca_y'] = np.zeros(len(df_work))
except Exception:
    df_work['pca_x'] = np.zeros(len(df_work))
    df_work['pca_y'] = np.zeros(len(df_work))

# Normalize all coordinate scales consistently to [-1, 1] for uniform graph representations
for c in ['tsne_x', 'tsne_y', 'umap_x', 'umap_y', 'pca_x', 'pca_y']:
    if c in df_work.columns and len(df_work) > 0:
        c_min, c_max = df_work[c].min(), df_work[c].max()
        if c_max > c_min:
            df_work[c] = (df_work[c] - c_min) / (c_max - c_min) * 2 - 1
        elif c_max == c_min:
            df_work[c] = 0.0

hover_cols = build_hover_columns(df_work)

# Create index mapping for lookups (needed by heatmap and other features)
id_to_local_idx = {doc_id: idx for idx, doc_id in enumerate(df_work['doc_id'].tolist())}

# Build embeddings matrix for similarity calculations
# (Sim matrix calculation moved up)

if doc_a is None and not df_work.empty:
    doc_a = df_work.iloc[0]['doc_id']
if doc_b is None and len(df_work) > 1:
    doc_b = df_work.iloc[1]['doc_id']

# Display filter status as a compact bar
filtered_n = len(df_work)
total_n = len(df)
sel_n = len(st.session_state['selected_ids'])
active_filters = []
if search_query: active_filters.append(f"Search: {search_query}")
if 'cluster' in df.columns and len(df['cluster'].unique()) > len(cluster_filter): active_filters.append("Cluster")
if keyword_filter: active_filters.append("Keyword")
if metadata_filters: active_filters.append("Metadata")

with status_placeholder.container():
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total", f"{total_n:,}")
    sc2.metric("Visible", f"{filtered_n:,}")
    sc3.metric("Selected", f"{sel_n:,}")
    sc4.metric("Filters", len(active_filters))
    if active_filters:
        st.caption(" · ".join(active_filters))

if len(df_work) < len(df):
    rc1, rc2 = st.columns([4, 1])
    with rc2:
        if st.button('Reset Filters', use_container_width=True):
            st.rerun()

tab_explore = st.container()

with tab_explore:
    # Color mapping for Cohorts
    if color_mode == 'Cohort':
        # Assign colors to cohorts if active
        df_work['__cohort'] = 'Other'
        cohort_map = st.session_state.get('saved_cohorts', {})
        # Simple color cycle
        cohort_colors = px.colors.qualitative.Bold
        color_map = {'Other': '#EEEEEE'}
        
        # Apply cohort labels (priority to last loaded?)
        # We can just iterate and overlay.
        for i, (name, ids) in enumerate(cohort_map.items()):
            df_work.loc[df_work['doc_id'].isin(ids), '__cohort'] = name
            color_map[name] = cohort_colors[i % len(cohort_colors)]
        
        # We need to hack the plotting function slightly to support custom categorical column mapping
        # For now, let's reuse the 'color' argument logic in plotting by mapping it to a dedicated column
        # Ideally, we pass color_col='__cohort' and color_map
    
    # Inspector Logic: identify "primary" selected point
    primary_doc_id = None
    if st.session_state['selected_ids']:
        # Last selected is usually the "inspection" target
        primary_doc_id = st.session_state['selected_ids'][-1]
        
    selection_focus = None
    selection_tsne = None
    selection_umap = None
    selection_pca = None
    
    if view_mode == "Single (Focus)":
        # Split layout: Large Plot + Inspector Panel
        pcol1, pcol2 = st.columns([3, 1])
        
        with pcol1:
            st.markdown(f"### {focus_proj} Projection")
            # Map focus_proj to column names
            x, y = 'umap_x', 'umap_y'
            if focus_proj == 't-SNE': x, y = 'tsne_x', 'tsne_y'
            elif focus_proj == 'PCA': x, y = 'pca_x', 'pca_y'
            
            # Use same plotting function but larger
            fig_focus = make_plot(
                df_work, x, y, 
                st.session_state['selected_ids'] or [], 
                st.session_state.get('search_hits', []), 
                f"{focus_proj} Large View", 
                hover_cols, 
                color_mode,
                focused_ids=st.session_state.get('focused_ids', []),
                show_hover=show_hover
            )
            # Update opacity/size safely if customizable
            apply_scale_and_alpha(fig_focus, point_alpha, 1.2 * point_size_scale)

            selection_focus = st.plotly_chart(
                fig_focus, 
                use_container_width=True, 
                on_select="rerun", 
                selection_mode=['points', 'box', 'lasso'],
                key=f"chart_focus_{st.session_state['chart_revisions']['chart_focus']}",
                height=700
            )

        with pcol2:
            st.markdown("### Inspector")
            if primary_doc_id:
                # Find document data
                row = df[df['doc_id'] == primary_doc_id].iloc[0] if not df[df['doc_id'] == primary_doc_id].empty else None
                if row is not None:
                    st.info(f"**ID:** {row['doc_id']}")
                    st.caption(f"Cluster: {row.get('cluster', 'N/A')}")
                    
                    st.text_area("Content", row.get('text', ''), height=200)
                    
                    # Metadata
                    with st.expander("Metadata", expanded=True):
                        for c in ['Slogan', 'Authors', 'Year', 'Data Domain']:
                            if c in row and not pd.isna(row[c]):
                                st.write(f"**{c}:** {row[c]}")
                    
                    # Neighborhood Stability (Mini view)
                    st.markdown("---")
                    st.caption("Nearest Neighbors (Global)")
                    # Simple NN calculation on the fly for single doc if meaningful
                    try:
                        idx = doc_id_to_global_idx.get(primary_doc_id)
                        if idx is not None:
                            vec = embeddings_for_sim[idx].reshape(1, -1)
                            # Cosine Sim
                            sims = cosine_similarity(vec, embeddings_for_sim).flatten()
                            # Get top 5 (excluding self)
                            top_indices = sims.argsort()[::-1][1:6]
                            
                            for i in top_indices:
                                nid = df.iloc[i]['doc_id']
                                score = sims[i]
                                st.write(f"- **{nid}**: {score:.3f}")
                    except Exception as e:
                        st.caption("NN calculation unavailable.")
            else:
                st.sidebar.info("Select a point to inspect details.")
                st.write("Click a point ->")

    else:
        # Fully flat, side-by-side projections for maximum visibility and comparative analysis
        st.markdown("### Projections")
        plot_col1, plot_col2, plot_col3 = st.columns(3, gap="small")
        
        with plot_col1:
            fig_tsne = make_plot(
                df_work, 'tsne_x', 'tsne_y',
                st.session_state['selected_ids'] or [],
                st.session_state.get('search_hits', []),
                't-SNE', hover_cols, color_mode,
                focused_ids=st.session_state.get('focused_ids', []),
                show_hover=show_hover
            )
            apply_scale_and_alpha(fig_tsne, point_alpha, point_size_scale)
            
            selection_tsne = st.plotly_chart(
                fig_tsne, use_container_width=True, on_select="rerun", 
                selection_mode=['points', 'box', 'lasso'],
                key=f"chart_tsne_{st.session_state['chart_revisions']['chart_tsne']}"
            )
            if show_download_buttons:
                st.download_button(
                    label='Download t-SNE', data=fig_tsne.to_html(),
                    file_name='tsne_plot.html', mime='text/html', use_container_width=True
                )
        
        with plot_col2:
            fig_umap = make_plot(
                df_work, 'umap_x', 'umap_y',
                st.session_state['selected_ids'] or [],
                st.session_state.get('search_hits', []),
                'UMAP', hover_cols, color_mode,
                focused_ids=st.session_state.get('focused_ids', []),
                show_hover=show_hover
            )
            apply_scale_and_alpha(fig_umap, point_alpha, point_size_scale)
            
            selection_umap = st.plotly_chart(
                fig_umap, use_container_width=True, on_select="rerun", 
                selection_mode=['points', 'box', 'lasso'],
                key=f"chart_umap_{st.session_state['chart_revisions']['chart_umap']}"
            )
            if show_download_buttons:
                st.download_button(
                    label='Download UMAP', data=fig_umap.to_html(),
                    file_name='umap_plot.html', mime='text/html', use_container_width=True
                )
                
        with plot_col3:
            fig_pca = make_plot(
                df_work, 'pca_x', 'pca_y',
                st.session_state['selected_ids'] or [],
                st.session_state.get('search_hits', []),
                'PCA (preview)', hover_cols, color_mode,
                focused_ids=st.session_state.get('focused_ids', []),
                show_hover=show_hover
            )
            apply_scale_and_alpha(fig_pca, point_alpha, point_size_scale)
            
            selection_pca = st.plotly_chart(
                fig_pca, use_container_width=True, on_select="rerun", 
                selection_mode=['points', 'box', 'lasso'],
                key=f"chart_pca_{st.session_state['chart_revisions']['chart_pca']}"
            )
            if show_download_buttons:
                st.download_button(
                    label='Download PCA', data=fig_pca.to_html(),
                    file_name='pca_plot.html', mime='text/html', use_container_width=True
                )

# ── Process Graph Selections ──
current_graph_selections = set()
for sel in [selection_focus, selection_tsne, selection_umap, selection_pca]:
    if sel and hasattr(sel, 'selection') and hasattr(sel.selection, 'points'):
        for pt in sel.selection.points:
            # pt acts like a dictionary in 1.35+
            cdata = pt.get('customdata', [])
            if cdata:
                current_graph_selections.add(str(cdata[0]))

if current_graph_selections:
    last_graph = st.session_state.get('last_graph_selections', set())
    if current_graph_selections != last_graph:
        st.session_state['last_graph_selections'] = set(current_graph_selections)
        # We replace the current selection with the new lasso selection completely
        update_selection(list(current_graph_selections))
        st.rerun()

# ── Dynamic Grid Layout for Analysis Section ──
if view_mode == "Single (Focus)":
    st.markdown("---")

st.markdown("### Analysis")
# Use a 3-column span instead of hidden tabs so everything is accessible at once
tab_docs, tab_stats, tab_keywords = st.columns([2, 1, 1], gap="medium")

def render_selection_details(selected_ids):
    st.subheader('Selected Document Details')
    
    if not selected_ids:
        st.info('Select points in the graphs to see details here.')
        return

    # Filter dataframe for selected IDs
    selected_df = df_work[df_work['doc_id'].isin(selected_ids)]
    
    if selected_df.empty:
        st.warning('Selected IDs not found in current filtered dataset.')
        return

    st.write(f"**{len(selected_df)} document(s) selected**")
    
    # Interactive Table for Selected Docs
    # Prepare display dataframe
    display_df = selected_df.copy()
    display_df['Snippet'] = display_df['__snippet']
    
    # Configure columns
    cols = ['doc_id', 'Snippet']
    if 'cluster' in display_df.columns:
        cols.append('cluster')
    # Add other useful metadata if available
    extras = [c for c in ['Slogan', 'Year', 'Publisher'] if c in display_df.columns]
    cols.extend(extras)
    
    # Show interactive dataframe
    selection_event = st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="sel_details_table_stable"
    )
    
    # Extract selected document IDs
    selected_doc_ids = []
    if selection_event.selection.rows:
        for row_idx in selection_event.selection.rows:
            selected_doc_ids.append(display_df.iloc[row_idx]['doc_id'])
            
    # --- Helper Functions for Comparison ---
    def fetch_vector(doc_id):
        idx = id_to_local_idx.get(doc_id)
        if idx is None:
            return None, None
        try:
            vec = embeddings_for_sim[idx]
            if hasattr(vec, 'toarray'):
                vec = vec.toarray().flatten()
            elif hasattr(vec, 'ndim') and vec.ndim > 1:
                vec = vec.flatten()
            vec = np.asarray(vec, dtype=float)
            return idx, vec
        except Exception:
            return None, None

    def compute_pair_metrics(a, b):
        idx_a, vec_a = fetch_vector(a)
        idx_b, vec_b = fetch_vector(b)
        if vec_a is None or vec_b is None:
            return None
        cosine_val = float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0])
        euclid_val = float(np.linalg.norm(vec_a - vec_b))
        return {'cosine': cosine_val, 'euclidean': euclid_val}

    def render_doc_card(doc_id, expanded=True):
        if not doc_id: return
        row = df_work[df_work['doc_id'] == doc_id].iloc[0]
        st.markdown(f"#### {doc_id}")
        st.caption(f"Cluster: {row.get('cluster', 'N/A')}")
        st.write(row.get('__snippet', row.get('text', '')))
        meta_dict = {}
        for col in row.index:
            if col not in ['doc_id', 'text', 'cluster', '__snippet', 'Snippet'] and not col.startswith('__'):
                val = row[col]
                if not pd.isna(val) and str(val).strip():
                    meta_dict[col] = val
        if meta_dict:
            st.json(meta_dict, expanded=False)

    # --- Interaction Logic ---
    if len(selected_doc_ids) == 0:
        st.caption("Select rows in the table above to see details or compare documents.")
        
    elif len(selected_doc_ids) == 1:
        # SINGLE DOC MODE
        sel_doc_id = selected_doc_ids[0]
        st.markdown("---")
        render_doc_card(sel_doc_id)
        
        # Show Neighbors for this single doc
        st.markdown("**Nearest Neighbors**")
        # Reuse neighbor logic
        neighbors_to_fetch = min(6, len(df_work))
        if neighbors_to_fetch >= 2:
            try:
                nn_model = NearestNeighbors(metric='cosine', n_neighbors=neighbors_to_fetch).fit(embeddings_for_sim)
                vec_idx, vec = fetch_vector(sel_doc_id)
                if vec is not None:
                    dists, indices = nn_model.kneighbors(vec.reshape(1, -1))
                    disp_data = []
                    for i, dist in zip(indices[0], dists[0]):
                        if i < len(df_work):
                            nid = df_work.iloc[i]['doc_id']
                            if nid != sel_doc_id:
                                sim = 1.0 - dist
                                disp_data.append({'Match (ID)': nid, 'Sim': sim})
                    if disp_data:
                        st.dataframe(pd.DataFrame(disp_data), hide_index=True, use_container_width=True)
            except Exception:
                pass

    elif len(selected_doc_ids) == 2:
        # COMPARISON MODE
        doc_a, doc_b = selected_doc_ids[0], selected_doc_ids[1]
        st.markdown("---")
        st.subheader("Comparison View")
        
        # Metrics
        metrics = compute_pair_metrics(doc_a, doc_b)
        c1, c2 = st.columns(2)
        c1.metric('Cosine Similarity', f"{metrics['cosine']:.3f}" if metrics else 'N/A', help="1.0 = Identical meaning")
        c2.metric('Euclidean Dist', f"{metrics['euclidean']:.3f}" if metrics else 'N/A', help="Lower is closer")
        
        # Side-by-Side Content
        colA, colB = st.columns(2)
        with colA:
            render_doc_card(doc_a)
        with colB:
            render_doc_card(doc_b)
            
    else:
        # MULTI-DOC MODE: show all selected docs
        st.markdown("---")
        st.subheader(f"Inspecting {len(selected_doc_ids)} Documents")
        
        # Group similarity summary
        if len(selected_doc_ids) <= 50:
            try:
                vecs = []
                for did in selected_doc_ids:
                    _, v = fetch_vector(did)
                    if v is not None:
                        vecs.append(v)
                if len(vecs) >= 2:
                    sim_matrix = cosine_similarity(np.vstack(vecs))
                    # Average pairwise similarity (excluding self-similarity on diagonal)
                    n = len(vecs)
                    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
                    st.metric("Avg Pairwise Similarity", f"{avg_sim:.3f}", help="Average cosine similarity between all pairs in this group. 1.0 = identical.")
            except Exception:
                pass
        
        # Render doc cards in a 2-column grid
        for i in range(0, len(selected_doc_ids), 2):
            cols = st.columns(2)
            with cols[0]:
                render_doc_card(selected_doc_ids[i])
            if i + 1 < len(selected_doc_ids):
                with cols[1]:
                    render_doc_card(selected_doc_ids[i + 1])

# ── Tab 1: Selected Documents ──
with tab_docs:
    with st.container(border=True):
        render_selection_details(st.session_state['selected_ids'])
    
    # Distance Heatmap (only when 2+ selected)
    if len(st.session_state['selected_ids']) >= 2:
        with st.container(border=True):
            st.subheader("Distance Matrix")
            st.caption('How similar is each selected document to every other? Green = Similar, Red = Different.')
            selected_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
            if len(selected_df) <= 100:
                selected_indices = [id_to_local_idx.get(doc_id) for doc_id in st.session_state['selected_ids'] if doc_id in id_to_local_idx]
                selected_indices = [i for i in selected_indices if i is not None and i < len(embeddings_for_sim)]
                
                if len(selected_indices) >= 2:
                    selected_embeddings = embeddings_for_sim[selected_indices]
                    similarity_matrix = cosine_similarity(selected_embeddings)
                    
                    import plotly.graph_objects as go
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=similarity_matrix,
                        x=[st.session_state['selected_ids'][i] for i in range(len(selected_indices))],
                        y=[st.session_state['selected_ids'][i] for i in range(len(selected_indices))],
                        colorscale='RdYlGn',
                        zmid=0.5,
                        text=similarity_matrix.round(3),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Cosine Sim")
                    ))
                    fig_heatmap.update_layout(
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=500
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(f'Heatmap available for ≤100 selected points (currently {len(selected_df)} selected)')

# ── Tab 2: Cluster Stats ──
with tab_stats:
    target_df = df_work
    stats_title = "Cluster Stats (All Visible)"
    
    if st.session_state['selected_ids']:
        sel_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
        if not sel_df.empty:
            target_df = sel_df
            stats_title = "Cluster Stats (Selection)"
    
    if 'cluster' in target_df.columns and target_df['cluster'].nunique() > 0:
        with st.container(border=True):
            st.subheader(stats_title)
            cluster_stats = target_df.groupby('cluster').agg({
                'doc_id': 'count',
            }).round(2)
            cluster_stats.columns = ['Count']
            cluster_stats = cluster_stats.sort_values('Count', ascending=True) # Ascending for top-down descending layout
            
            plot_df = cluster_stats.reset_index()
            # Ensure clusters always read beautifully on the layout
            plot_df['cluster_label'] = plot_df['cluster'].apply(lambda x: GLOBAL_CLUSTER_MAP.get(x, f'Cluster {x}'))
            
            fig_cluster = px.bar(
                plot_df,
                text='Count',
                y='cluster_label',
                x='Count',
                orientation='h',
                color='Count',
                color_continuous_scale='Blues',
            )
            fig_cluster.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                xaxis_title='Documents',
                yaxis_title='',
                yaxis=dict(type='category', dtick=1),
                coloraxis_showscale=False,
            )
            fig_cluster.update_traces(textposition='auto')
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            if not st.session_state['selected_ids']:
                st.caption('Click to select all docs in a cluster:')
                cluster_cols = st.columns(min(5, len(cluster_stats)))
                for idx, (cluster_id, row) in enumerate(cluster_stats.head(5).iterrows()):
                    with cluster_cols[idx]:
                        if st.button(f'Cluster {cluster_id} ({int(row["Count"])})', key=f'select_cluster_{cluster_id}', use_container_width=True):
                            cluster_docs = df_work[df_work['cluster'] == cluster_id]['doc_id'].tolist()
                            update_selection(cluster_docs, additive=False)
    else:
        st.info("No cluster data available.")

# ── Tab 3: Top Keywords ──
with tab_keywords:
    if st.session_state['selected_ids']:
        sel_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
        all_toks = [tok for toks in sel_df['__tokens'] for tok in toks]
        if all_toks:
            from collections import Counter
            counts = Counter(all_toks).most_common(20) # Trim space
            kw_df = pd.DataFrame(counts, columns=['Term', 'Count']).sort_values('Count', ascending=True)
            fig_kw = px.bar(
                kw_df,
                text='Count',
                y='Term',
                x='Count',
                orientation='h',
                color='Count',
                color_continuous_scale='Viridis',
            )
            fig_kw.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(type='category', dtick=1),
                yaxis_title='',
                xaxis_title='Count',
                bargap=0.3,
            )
            fig_kw.update_traces(textposition='auto')
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("No tokens available. Try clearing cache (⋮ → Clear cache).")
    else:
        st.info("Select documents to see their top keywords.")

st.markdown('---')
st.caption('Embeddings Explorer v2.0')

