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
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="Embeddings Explorer (Prototype)")

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
def try_load(path):
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
    roots.extend(['artifacts/preproc_default', 'artifacts/newsgroups', 'artifacts', ''])
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
    for col in ['Slogan', 'Data Domain', 'Tool Used/Mentioned', 'Year']:
        if col in df.columns:
            extras.append(col)
    return list(dict.fromkeys(available + extras + metadata_cols[:5]))


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
coords_tsne_path = find_file(bundle_candidates('coords_tsne.npy'))
coords_umap_path = find_file(bundle_candidates('coords_umap.npy'))
# Prefer preproc_default coords by default; bundle preference can override
coords_pca_path = find_file(bundle_candidates('coords.npy'))
tfidf_matrix_path = find_file(bundle_candidates('tfidf_matrix.npz'))
cluster_labels_path = find_file(bundle_candidates('cluster_labels.npy'))
processed_csv_path = find_file(bundle_candidates('processed_data_with_clusters.csv'))
doc_ids_path = find_file(bundle_candidates('doc_ids.txt'))


# Load or synthesize small dataset
if processed_csv_path:
    df = pd.read_csv(processed_csv_path)
else:
    # try to build df from doc_ids file
    if doc_ids_path:
        try:
            with open(doc_ids_path, 'r', encoding='utf-8') as f:
                ids = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({'doc_id': ids})
            df['text'] = df['doc_id'].apply(lambda x: f'Snippet for {x}')
            df['cluster'] = 0
        except Exception:
            df = pd.DataFrame({'doc_id': [], 'text': [], 'cluster': []})
    else:
        # synthetic fallback
        n = 200
        df = pd.DataFrame({
            'doc_id': [f'DOC{i:04d}' for i in range(n)],
            'text': [f'This is a sample snippet number {i}.' for i in range(n)],
            'cluster': np.random.randint(0, 8, size=n)
        })

if 'doc_id' not in df.columns:
    df['doc_id'] = df.index.astype(str)

df['doc_id'] = df['doc_id'].astype(str)
text_source = None
for candidate in ['text', 'cleaned_text', 'preprocessed_text']:
    if candidate in df.columns:
        text_source = candidate
        break
if text_source is None:
    df['text'] = df['doc_id'].apply(lambda x: f'Snippet for {x}')
    text_source = 'text'
df['__snippet'] = df[text_source].apply(build_snippet)
token_col, available_keywords = parse_keyword_space(df)
if '__tokens' not in df.columns:
    df['__tokens'] = [[] for _ in range(len(df))]

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
coords_tsne = try_load(coords_tsne_path) if coords_tsne_path else None
coords_umap = try_load(coords_umap_path) if coords_umap_path else None
coords_base = try_load(coords_pca_path) if coords_pca_path else None
tfidf_matrix = try_load(tfidf_matrix_path) if tfidf_matrix_path else None

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


def make_plot(df_plot, xcol, ycol, selected_ids, search_ids, title, hover_cols, color_mode='Selection'):
    df_plot = df_plot.copy()
    df_plot['__size'] = 8  # Default size
    
    if color_mode == 'Cluster':
        # Color by cluster
        if 'cluster' in df_plot.columns:
            # Convert cluster to string to ensure it's treated as categorical
            df_plot['cluster_str'] = df_plot['cluster'].astype(str)
            color_col = 'cluster_str'
        else:
            # Fallback to status if no cluster column
            df_plot['__status'] = 'Other'
            color_col = '__status'
        
        # Make selected points larger
        if selected_ids:
            df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
        if search_ids:
            df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
        
        fig = px.scatter(
            df_plot,
            x=xcol,
            y=ycol,
            color=color_col,
            hover_name='doc_id',
            hover_data=hover_cols,
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Use Plotly's default color palette
            # No fixed height - let it be responsive
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
        
        # Improved color scheme - more vibrant, less washed out
        color_map = {
            'Selected': '#E63946',      # Rich red
            'Search hit': '#F4A261',    # Warm orange
            'Other': '#457B9D'          # Deep blue-gray
        }
        
        fig = px.scatter(
            df_plot,
            x=xcol,
            y=ycol,
            color='__status',
            color_discrete_map=color_map,
            hover_name='doc_id',  # Show doc_id as the main title in hover
            hover_data=hover_cols,
            # No fixed height - let it be responsive
            category_orders={'__status': ['Selected', 'Search hit', 'Other']}  # Legend order
        )
    
    # Make hover text readable but not too large/intrusive
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.4)",  # More transparent to see dots
            font_size=11,  # Smaller font
            font_family="Arial, sans-serif",
            font_color="black",
            bordercolor="gray"
        )
    )
    
    # attach doc_id as customdata for robust mapping
    # We MUST ensure doc_id is available in the event data, regardless of trace count.
    # fig.update_traces(customdata=df_plot['doc_id'])  <-- REMOVED: This scramble IDs on multi-trace plots!
    # Instead, we rely on px.scatter adding 'doc_id' to customdata via hover_data (which it does if doc_id is in hover_cols)
    
    # attach doc_id as customdata for robust mapping
    # Keep it simple: don't modify traces, let plotly_events use point indices
    try:
        # Just set better marker styling
        for trace in fig.data:
            # Solid, vibrant markers with good visibility
            if color_mode == 'Selection':
                # In selection mode, size by status
                if trace.name == 'Selected':
                    trace.marker.size = 10
                    trace.marker.opacity = 0.95
                elif trace.name == 'Search hit':
                    trace.marker.size = 9
                    trace.marker.opacity = 0.9
                else:
                    trace.marker.size = 5
                    trace.marker.opacity = 0.75
            else:
                # In cluster mode, uniform sizing
                trace.marker.size = 6
                trace.marker.opacity = 0.85
            
            # Clean dark border for definition
            trace.marker.line = dict(width=0.8, color='rgba(0,0,0,0.4)')
    except Exception:
        pass
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family="Arial, sans-serif", color="#666666"),
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
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0'),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0'),
        autosize=True  # Enable responsive sizing
    )
    return fig


st.title('Embeddings Explorer — Prototype')

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
    st.session_state['chart_revisions'] = {'chart_tsne': 0, 'chart_umap': 0, 'chart_pca': 0}

# Status indicators at the top
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.metric("Total Documents", len(df))
with col_stat2:
    st.metric("Selected", len(st.session_state['selected_ids']))
with col_stat3:
    search_count = len(st.session_state.get('search_hits', []))
    st.metric("Search Hits", search_count if search_count > 0 else "—")
with col_stat4:
    cluster_count = df['cluster'].nunique() if 'cluster' in df.columns else 0
    st.metric("Clusters", cluster_count if cluster_count > 0 else "—")

if st.session_state.get('chunk_parent_map'):
    st.caption('Chunk ↔ parent linking is active: clicking either side will automatically highlight its counterpart(s).')

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
        # Probe known roots
        for root in ['artifacts/preproc_default', 'artifacts/newsgroups', 'artifacts', '']:
            label = root if root else '(repo root)'
            probe = f"{root}/processed_data_with_clusters.csv" if root else 'processed_data_with_clusters.csv'
            if os.path.exists(probe):
                bundle_options.append(label)
                bundle_map[label] = root
        if not bundle_options:
            # Fallback option when no artifacts found
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
        color_mode = st.radio(
            'Color points by:',
            options=['Selection', 'Cluster'],
            help='Switch between showing selection status (Red/Blue) or semantic clusters (Multi-color). Use "Cluster" to see topic distribution, "Selection" to track your active set.'
        )
        show_download_buttons = st.checkbox('Show plot download buttons', value=False, help='Enable this to download high-resolution PNGs of the current plots for reports or presentations.')
        detailed_hover = st.checkbox('Show detailed hover info', value=False, help='If checked, hovering over a point shows rich metadata. Uncheck for a cleaner, faster view with just IDs.')

    with st.expander('Search & Filter', expanded=False):
        search_query = st.text_input('Search doc_id / keyword / phrase', '')
        search_scopes = st.multiselect('Search scopes', options=['doc_id', 'keywords', 'phrase'], default=['doc_id', 'phrase'])
        clusters = sorted(df['cluster'].unique().tolist()) if 'cluster' in df.columns else []
        cluster_filter = st.multiselect('Clusters', options=clusters, default=clusters)
        keyword_filter = st.multiselect('Keyword tags', options=available_keywords, default=[], help='Filter the visible points to only those containing specific keywords. Useful for narrowing down to a theme.')
        
        st.markdown('**Metadata Filters**')
        cat_defaults = [c for c in ['Data Domain', 'Tool Used/Mentioned', 'Publisher'] if c in categorical_metadata]
        selected_cats = st.multiselect('Categorical fields', options=categorical_metadata, default=cat_defaults)
        for col in selected_cats:
            options = sorted({str(v) for v in df[col].dropna().unique() if str(v).strip()})
            sel = st.multiselect(f'{col}', options=options, key=f'cat_filter_{col}')
            if sel:
                metadata_filters[col] = sel
        num_defaults = [c for c in ['Year', 'Subjective Trust Score:'] if c in numeric_metadata]
        selected_nums = st.multiselect('Numeric fields', options=numeric_metadata, default=num_defaults)
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
        max_points = st.slider('Max points to display', min_value=50, max_value=3000, value=1200, step=50)

    with st.expander('Selection Tools', expanded=False):
        st.markdown('**Quick Jumps**')
        doc_options = df['doc_id'].tolist()
        default_doc = doc_options[0] if doc_options else ''
        select_doc = st.selectbox('Jump to doc_id', options=doc_options or [''], index=0, help='Instantly select and center on a specific document by ID.')
        if st.button('Highlight doc', key='btn_highlight_doc') and select_doc:
            update_selection([select_doc], additive=False)
        multi_select = st.multiselect('Pin doc_ids (max 15)', options=doc_options, default=st.session_state['selected_ids'][:5], max_selections=15)
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
            if st.button('Random', key='btn_random', use_container_width=True, help='Select 10 random points to explore diverse examples.'):
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
            import json
            export_data = {
                'selected_ids': st.session_state['selected_ids'],
                'count': len(st.session_state['selected_ids']),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            st.download_button(
                label='Export Selection (JSON)',
                data=json.dumps(export_data, indent=2),
                file_name='selected_docs.json',
                mime='application/json',
                use_container_width=True
            )

    with st.expander('Sessions & Advanced', expanded=False):
        st.markdown('**Sessions**')
        # Save session
        session_name = st.text_input('Session name', placeholder='my_analysis')
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
        chunk_choice = st.selectbox('Chunk column', options=['(none)'] + candidate_chunk_cols, index=1 if candidate_chunk_cols else 0)
        parent_choice = st.selectbox('Parent column', options=['(none)'] + candidate_parent_cols, index=1 if candidate_parent_cols else 0)
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
        doc_a = st.selectbox('Doc A', options=doc_options or [''], index=0, key='comp_a')
        doc_b = st.selectbox('Doc B', options=doc_options or [''], index=1 if len(doc_options) > 1 else 0, key='comp_b')

        st.markdown('---')
        st.markdown('**Embedding Parameters**')
        pca_comps = st.slider('PCA components (preview)', min_value=2, max_value=5, value=2)
        umap_neighbors = st.slider('UMAP neighbors', min_value=5, max_value=120, value=15)
        umap_min_dist = st.slider('UMAP min_dist', min_value=0.0, max_value=1.0, value=0.1)
        tsne_perplexity = st.slider('t-SNE perplexity', min_value=5, max_value=100, value=30)
        tsne_lr = st.slider('t-SNE learning rate', min_value=10, max_value=1000, value=200)
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

# Centralized Selection Logic: Process chart events before rendering new plots
# This ensures that a selection in any chart updates all charts immediately (fixing directional bugs)
# and handles empty selections properly (fixing deselection bugs).
chart_keys = ['chart_tsne', 'chart_umap', 'chart_pca']
for key in chart_keys:
    # Use the current revision to form the effective key
    rev = st.session_state['chart_revisions'][key]
    effective_key = f"{key}_{rev}"
    
    current_state = st.session_state.get(effective_key)
    last_state = st.session_state['last_chart_states'].get(key)
    
    # Detect if the chart state has changed since the last run
    if current_state != last_state:
        st.session_state['last_chart_states'][key] = current_state
        
        # Resolve the selection from the state object
        new_ids = []
        if current_state:
            # Handle both dict (older streamlit) and object (newer) styles
            selection_data = None
            if hasattr(current_state, 'selection'):
                selection_data = current_state.selection
            elif isinstance(current_state, dict):
                selection_data = current_state.get('selection', {})
            
            # Robust selection extraction using customdata (which we forced to be doc_id)
            if selection_data and 'points' in selection_data:
                points = selection_data['points']
                for p in points:
                    # Plotly puts customdata in 'customdata' list/array per point
                    # or sometimes directly in key 'customdata', or 'custom_data' (underscore)
                    cdata = p.get('customdata', p.get('custom_data'))
                    if cdata is not None:
                        # flatten if single-item list
                        if isinstance(cdata, list) and cdata:
                            new_ids.append(str(cdata[0]))
                        else:
                            new_ids.append(str(cdata))
                    # Fallback to point_index if customdata missing (shouldn't happen with our fix)
                    elif 'point_index' in p:
                        idx = p['point_index']
                        if 0 <= idx < len(df_work):
                            new_ids.append(df_work.iloc[idx]['doc_id'])

        # Update app selection state
        # If new_ids is empty, this clears the selection (handling deselection)
        # Unique the IDs
        selected_docs = list(dict.fromkeys(new_ids))
        
        is_additive = st.session_state.get('additive_mode', False)
        # print(f"DEBUG: Updating selection. Additive: {is_additive}")
        update_selection(selected_docs, additive=is_additive)
        
        if selected_docs:
            # If a selection was made (not just a clear), expire ALL charts to force their lassos to clear
            # This prevents the "ghost lasso" issue and ensures sequential selections work reliability
            # by forcing a fresh component mount for the next interaction.
            for k in chart_keys:
                st.session_state['chart_revisions'][k] += 1
                # Also clear their last state so the new empty state is accepted as "no change" or "clean slate"
                st.session_state['last_chart_states'][k] = None

        # Process only one interaction per rerun to avoid conflicts
        break

# Debug selection logic
with st.expander('Debug Selection Logic', expanded=False):
   st.write('Last Chart States:', {k: (v is not None) for k, v in st.session_state['last_chart_states'].items()})
   st.write('Additive Mode:', st.session_state.get('additive_mode'))
   st.write('Selected IDs count:', len(st.session_state.get('selected_ids', [])))
   st.write('Selection History len:', len(st.session_state.get('selection_history', [])))
   
   # Check actual chart keys in session state
   st.write('Session State Keys:', [k for k in st.session_state.keys() if k.startswith('chart_')])

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
    # (Removed dangerous fallback: local_idx in filtered df does not match global coords array)
    
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

# Diagnostic dump for running app (helps debug missing points in UI)
try:
    import json
    diag = {
        'n_points': len(df_work),
        'tsne_x_min': float(np.min(tsne_x)) if tsne_x else None,
        'tsne_x_max': float(np.max(tsne_x)) if tsne_x else None,
        'tsne_y_min': float(np.min(tsne_y)) if tsne_y else None,
        'tsne_y_max': float(np.max(tsne_y)) if tsne_y else None,
        'umap_x_min': float(np.min(umap_x)) if umap_x else None,
        'umap_x_max': float(np.max(umap_x)) if umap_x else None,
        'umap_y_min': float(np.min(umap_y)) if umap_y else None,
        'umap_y_max': float(np.max(umap_y)) if umap_y else None,
    }
    with open('artifacts/streamlit_diag.json', 'w', encoding='utf-8') as _f:
        json.dump(diag, _f)
except Exception:
    pass


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
        pca = PCA(n_components=pca_comps)
        pca_coords = pca.fit_transform(embeddings_for_sim)
        df_work['pca_x'] = pca_coords[:, 0]
        df_work['pca_y'] = pca_coords[:, 1] if pca_comps >= 2 else 0.0
    else:
        df_work['pca_x'] = np.zeros(len(df_work))
        df_work['pca_y'] = np.zeros(len(df_work))
except Exception:
    df_work['pca_x'] = np.zeros(len(df_work))
    df_work['pca_y'] = np.zeros(len(df_work))

hover_cols = build_hover_columns(df_work)
if not detailed_hover:
    # Minimal hover info as requested by user
    # Keep only the most critical identifier and grouping info
    hover_cols = ['doc_id']
    if 'cluster' in df_work.columns:
        hover_cols.append('cluster')
    # If a short snippet is very short, maybe include it?
    # But user asked to simplify "covering the screen", so stick to absolute minimum.

# Create index mapping for lookups (needed by heatmap and other features)
id_to_local_idx = {doc_id: idx for idx, doc_id in enumerate(df_work['doc_id'].tolist())}

# Build embeddings matrix for similarity calculations
# (Sim matrix calculation moved up)

if doc_a is None and not df_work.empty:
    doc_a = df_work.iloc[0]['doc_id']
if doc_b is None and len(df_work) > 1:
    doc_b = df_work.iloc[1]['doc_id']

# Display filter status
st.markdown('---')
filter_col1, filter_col2 = st.columns([3, 1])
with filter_col1:
    if len(df_work) < len(df):
        st.info(f'Showing **{len(df_work):,}** of **{len(df):,}** documents after filters')
    else:
        st.success(f'Showing all **{len(df):,}** documents')
with filter_col2:
    if len(df_work) < len(df):
        if st.button('Reset All Filters', use_container_width=True):
            st.rerun()

# Add helpful tooltips
with st.expander('How to use this tool', expanded=False):
    st.markdown('''
    **Lasso Selection**: Click and drag to draw a selection shape around points
    
    **Additive Selection**: Check the "Additive Selection" box to add multiple lasso selections together
    
    **t-SNE**: Preserves local structure, good for finding clusters
    
    **UMAP**: Balances local and global structure, faster than t-SNE
    
    **PCA**: Linear projection, shows main variance directions
    
    **Tips**:
    - Selected points appear in **red** and are larger
    - Search hits appear in **gold**
    - Use the sidebar to filter, search, and export selections
    - Click "Undo" to restore previous selection
    ''')


tab_explore, tab_analyze, tab_compare = st.tabs(["Explorer", "Analysis", "Comparison"])

with tab_explore:
    col1, col2, col3 = st.columns([1,1,1])

    # render plots with native Streamlit selection handling
    with col1:
        st.markdown("### t-SNE Projection")
        fig_tsne = make_plot(
            df_work,
            'tsne_x',
            'tsne_y',
            st.session_state['selected_ids'] or [],
            st.session_state.get('search_hits', []),
            't-SNE',
            hover_cols,
            color_mode
        )
        selection_tsne = st.plotly_chart(
            fig_tsne, 
            use_container_width=True,
            on_select="rerun",
            selection_mode=['points', 'box', 'lasso'],
            key=f"chart_tsne_{st.session_state['chart_revisions']['chart_tsne']}"
        )
        
        if show_download_buttons:
            st.download_button(
                label='Download t-SNE',
                data=fig_tsne.to_html(),
                file_name='tsne_plot.html',
                mime='text/html',
                use_container_width=True
            )

    with col2:
        st.markdown("### UMAP Projection")
        fig_umap = make_plot(
            df_work,
            'umap_x',
            'umap_y',
            st.session_state['selected_ids'] or [],
            st.session_state.get('search_hits', []),
            'UMAP',
            hover_cols,
            color_mode
        )
        selection_umap = st.plotly_chart(
            fig_umap,
            use_container_width=True,
            on_select="rerun",
            selection_mode=['points', 'box', 'lasso'],
            key=f"chart_umap_{st.session_state['chart_revisions']['chart_umap']}"
        )
        
        if show_download_buttons:
            st.download_button(
                label='Download UMAP',
                data=fig_umap.to_html(),
                file_name='umap_plot.html',
                mime='text/html',
                use_container_width=True
            )

    with col3:
        st.markdown("### PCA Projection")
        fig_pca = make_plot(
            df_work,
            'pca_x',
            'pca_y',
            st.session_state['selected_ids'] or [],
            st.session_state.get('search_hits', []),
            'PCA (preview)',
            hover_cols,
            color_mode
        )
        selection_pca = st.plotly_chart(
            fig_pca,
            use_container_width=True,
            on_select="rerun",
            selection_mode=['points', 'box', 'lasso'],
            key=f"chart_pca_{st.session_state['chart_revisions']['chart_pca']}"
        )
        
        if show_download_buttons:
            st.download_button(
                label='Download PCA',
                data=fig_pca.to_html(),
                file_name='pca_plot.html',
                mime='text/html',
                use_container_width=True
            )
        try:
            if st.sidebar.checkbox('Show raw PCA Plotly (debug)', value=False):
                st.plotly_chart(fig_pca, use_container_width=True)
        except Exception:
            pass

# Helper to render selection details
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

    # If too many, just show a table
    if len(selected_df) > 5:
        st.dataframe(selected_df.drop(columns=[c for c in selected_df.columns if c.startswith('__')]))
    else:
        # Show detailed cards for a few items
        cols = st.columns(len(selected_df))
        for idx, (_, row) in enumerate(selected_df.iterrows()):
            with cols[idx]:
                st.markdown(f"### {row['doc_id']}")
                st.caption(f"Cluster: {row.get('cluster', 'N/A')}")
                st.write(row.get('__snippet', row.get('text', '')))
                
                # Display other metadata
                meta_dict = {}
                for col in row.index:
                    if col not in ['doc_id', 'text', 'cluster', '__snippet'] and not col.startswith('__'):
                        val = row[col]
                        if not pd.isna(val) and str(val).strip():
                            meta_dict[col] = val
                
                if meta_dict:
                    st.json(meta_dict, expanded=False)

with tab_analyze:
    st.header('Analysis', help='Statistical breakdown and similarity analysis of the CURRENTLY SELECTED documents.')
    
    # Cluster Statistics
    if 'cluster' in df_work.columns and df_work['cluster'].nunique() > 0:
        with st.expander('Cluster Statistics', expanded=True):
            cluster_stats = df_work.groupby('cluster').agg({
                'doc_id': 'count',
                'tsne_x': ['mean', 'std'],
                'tsne_y': ['mean', 'std']
            }).round(2)
            cluster_stats.columns = ['Count', 't-SNE X Mean', 't-SNE X Std', 't-SNE Y Mean', 't-SNE Y Std']
            st.markdown('**Cluster Statistics**', help='Aggregated statistics for each cluster in the current filtered view.')
            cluster_stats = cluster_stats.sort_values('Count', ascending=False)
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Quick cluster selection buttons
            st.caption('Quick select cluster:')
            cluster_cols = st.columns(min(5, len(cluster_stats)))
            for idx, (cluster_id, row) in enumerate(cluster_stats.head(5).iterrows()):
                with cluster_cols[idx]:
                    if st.button(f'Cluster {cluster_id} ({int(row["Count"])})', key=f'select_cluster_{cluster_id}', use_container_width=True):
                        cluster_docs = df_work[df_work['cluster'] == cluster_id]['doc_id'].tolist()
                        update_selection(cluster_docs, additive=False)

    st.markdown('---')
    
    # Selection Details
    render_selection_details(st.session_state['selected_ids'])

    # Distance Heatmap for Selected Points
    if len(st.session_state['selected_ids']) >= 2:
        with st.expander('Distance Heatmap (Selected Points)', expanded=True):
            selected_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
            if len(selected_df) <= 100:  # Increased limit for larger heatmaps
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
                        colorbar=dict(title="Cosine Similarity")
                    ))
                    fig_heatmap.update_layout(
                        title='Pairwise Cosine Similarity',
                        xaxis_title='Document ID',
                        yaxis_title='Document ID',
                        height=400
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(f'Heatmap available for ≤100 selected points (currently {len(selected_df)} selected)')


with tab_compare:
    st.header('Document Comparison', help='Side-by-side comparison of any two documents from the filtered subset.')
    st.caption('Compare two documents to see their similarity scores and nearest neighbors. This tool operates on the *filtered* dataset, independent of the lasso selection (unless you filter by selection ids).')
    
    if len(st.session_state.get('selected_ids', [])) >= 2:
        if st.button('Use last 2 selected items for comparison', key='btn_use_selection_compare'):
            sel = st.session_state['selected_ids']
            st.session_state['comp_a'] = sel[-2] # second to last
            st.session_state['comp_b'] = sel[-1] # last
            st.rerun()
    
    def fetch_vector(doc_id):
        idx = id_to_local_idx.get(doc_id)
        if idx is None:
            return None, None
        # lookup_embedding expects a global index, but id_to_local_idx gives local idx in filtered view
        # We need the global index if we use lookup_embedding with base_embeddings. 
        # But embeddings_for_sim IS the filtered subset of base_embeddings.
        # So we should just replicate the densify logic here for embeddings_for_sim
        vec = embeddings_for_sim[idx]
        if scipy and scipy.sparse.issparse(vec):
            vec = vec.toarray().flatten()
        elif hasattr(vec, 'ndim') and vec.ndim > 1:
            vec = vec.flatten()
            
        # Ensure strict numeric array to avoid 'setting an array element with a sequence' errors
        try:
            vec = np.asarray(vec, dtype=float)
        except Exception:
            vec = None
            
        return idx, vec

    def compute_pair_metrics(a, b):
        idx_a, vec_a = fetch_vector(a)
        idx_b, vec_b = fetch_vector(b)
        if vec_a is None or vec_b is None:
            return None
        cosine_val = float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0])
        euclid_val = float(np.linalg.norm(vec_a - vec_b))
        return {'cosine': cosine_val, 'euclidean': euclid_val}

    metrics = compute_pair_metrics(doc_a, doc_b)
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric('Cosine similarity', f"{metrics['cosine']:.3f}" if metrics else 'N/A', help='Measures how similar the content/topic is (1.0 = identical; 0.0 = unrelated). Focuses on direction/angle, ignoring length.')
    metric_col2.metric('Euclidean distance', f"{metrics['euclidean']:.3f}" if metrics else 'N/A', help='Measures the literal distance between points (0.0 = identical). Closer points are more similar in all aspects (magnitude + direction).')

    def render_doc_panel(doc_id, column):
        column.subheader(doc_id or 'Select a document')
        if not doc_id or doc_id not in id_to_local_idx:
            column.info('Pick a document from the sidebar to populate this panel.')
            return
        row = df_work[df_work['doc_id'] == doc_id].iloc[0]
        column.write(row.get('__snippet', row.get('text', 'No snippet available.')))
        meta_lines = []
        for col in ['cluster', 'Slogan', 'Data Domain', 'Tool Used/Mentioned', 'Year', 'Publisher']:
            if col in row and not pd.isna(row[col]):
                meta_lines.append(f"**{col}:** {row[col]}")
        if meta_lines:
            column.caption(' | '.join(meta_lines))

    colA, colB = st.columns(2)
    render_doc_panel(doc_a, colA)
    render_doc_panel(doc_b, colB)

    st.markdown('---')
    st.subheader('Nearest neighbors (current subset)')
    # We want ~5 neighbors, but fetch +1 because the doc itself is usually included as distance 0
    neighbors_to_fetch = min(7, len(df_work))
    
    # Compute neighbors locally if needed (for display)
    if neighbors_to_fetch >= 2:
        try:
            nn_model = NearestNeighbors(metric='cosine', n_neighbors=neighbors_to_fetch).fit(embeddings_for_sim)
            cos_distances, cos_indices = nn_model.kneighbors(embeddings_for_sim)
        except Exception as exc:
            cos_distances, cos_indices = None, None
            st.warning(f'Neighbor computation failed: {exc}')
    else:
        cos_distances, cos_indices = None, None

    def get_neighbor_ids(doc_id):
        if cos_distances is None or cos_indices is None:
            return []
        idx = id_to_local_idx.get(doc_id)
        if idx is None:
            return []
        
        # indices in embeddings_for_sim
        neighbor_indices = cos_indices[idx]
        neighbor_dists = cos_distances[idx]
        
        # map back to global index then doc_id
        # embeddings_for_sim is 1:1 with df_work rows
        results = []
        for i, dist in zip(neighbor_indices, neighbor_dists):
            if i < len(df_work):
                n_id = df_work.iloc[i]['doc_id']
                if n_id != doc_id: # exclude self
                    results.append((n_id, dist))
        return results

    neighbors_a = get_neighbor_ids(doc_a)
    neighbors_b = get_neighbor_ids(doc_b)
    
    # Identify overlap
    ids_a = {x[0] for x in neighbors_a}
    ids_b = {x[0] for x in neighbors_b}
    common = ids_a.intersection(ids_b)

    def render_neighbors_table(neighbors, column, common_set):
        if not neighbors:
             column.info('No other neighbors found in subset.')
             return
             
        # Format for display
        disp_data = []
        for nid, dist in neighbors[:5]: # show top 5 unique
            sim = 1.0 - dist # convert distance to similarity roughly
            marker = "* " if nid in common_set else ""
            disp_data.append({
                'Match': f"{marker}{nid}",
                'Cosine Sim': f"{sim:.4f}"
            })
        column.table(pd.DataFrame(disp_data))

    colA_n, colB_n = st.columns(2)
    render_neighbors_table(neighbors_a, colA_n, common)
    render_neighbors_table(neighbors_b, colB_n, common)


st.markdown('---')
st.caption('Embeddings Explorer v2.0 | Optimized for Performance & Usability')

