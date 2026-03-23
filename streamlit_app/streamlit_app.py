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


def parse_variant_settings(variant_path):
    """
    Parse preprocessing settings from variant folder name and path.
    Returns dict with human-readable settings description.
    
    Examples:
      preproc_default → {"label": "Default", "settings": "Lemmatized, stopwords removed, lowercase"}
      preproc_no_stopwords → {"label": "No Stopwords", "settings": "Lemmatized, lowercase, NO stopword removal"}
      preproc_no_lemmatize → {"label": "No Lemmatization", "settings": "NO lemmatization, stopwords removed, lowercase"}
    """
    if not variant_path:
        return {"label": "Auto-detected", "settings": "Default settings", "folder": None, "config": {}}
    
    # Normalize and extract folder name
    norm_path = os.path.normpath(variant_path)
    folder_name = os.path.basename(norm_path)
    
    # Parse settings from folder name
    settings = {
        "lemmatize": True,
        "lowercase": True,
        "remove_stopwords": True,
        "min_length": 3
    }
    
    if "no_lemmatize" in folder_name:
        settings["lemmatize"] = False
    if "no_lowercase" in folder_name:
        settings["lowercase"] = False
    if "no_stopwords" in folder_name:
        settings["remove_stopwords"] = False
    if "minlen2" in folder_name:
        settings["min_length"] = 2
    
    # Generate human-readable label
    if folder_name == "preproc_default":
        label = "Default"
    elif "no_lemmatize" in folder_name and "no_stopwords" in folder_name:
        label = "No Lemmatization, No Stopwords"
    elif "no_lemmatize" in folder_name:
        label = "No Lemmatization"
    elif "no_stopwords" in folder_name:
        label = "No Stopwords"
    elif "no_lowercase" in folder_name:
        label = "Preserve Case"
    elif "minlen2" in folder_name:
        label = "Min Length 2"
    else:
        label = folder_name.replace("preproc_", "").replace("_", " ").title()
    
    # Generate settings description
    parts = []
    if settings["lemmatize"]:
        parts.append("Lemmatized")
    else:
        parts.append("NO lemmatization")
    
    if settings["remove_stopwords"]:
        parts.append("stopwords removed")
    else:
        parts.append("Keep stopwords")
    
    if settings["lowercase"]:
        parts.append("lowercase")
    else:
        parts.append("preserve case")
    
    parts.append(f"min {settings['min_length']}ch")
    
    return {
        "label": label,
        "settings": ", ".join(parts),
        "folder": folder_name,
        "config": settings
    }


def get_variant_comparison_info(current_variant_path, default_variant_path=None):
    """
    Generate comparison info between current variant and default.
    Returns dict with differences highlighted.
    """
    current = parse_variant_settings(current_variant_path)
    
    # Determine default path
    if not default_variant_path:
        if selected_bundle_root:
            default_variant_path = os.path.join(os.path.dirname(selected_bundle_root), 'preproc_default')
        else:
            default_variant_path = 'artifacts/preproc_default'
    
    default = parse_variant_settings(default_variant_path)
    
    diffs = []
    for key in current.get("config", {}):
        if current["config"].get(key) != default["config"].get(key):
            diffs.append(f"{key}: {current['config'].get(key)} (default: {default['config'].get(key)})")
    
    return {
        "current": current,
        "default": default,
        "differences": diffs
    }



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


def get_keywords_with_tfidf(selected_tokens, full_corpus_tokens, n=15, min_doc_freq=2):
    """
    Extract top keywords using TF-IDF weighting with strong stop word filtering.
    
    **CRITICAL**: This function requires PREPROCESSED token lists (lemmatized, stopword-filtered).
    These must come from the offline pipeline's 'tokens' column (created by parse.py::preprocess_texts).
    
    DO NOT pass raw text tokenized via regex — this causes keyword inconsistency between
    offline and online analysis.
    
    Args:
        selected_tokens: list of PREPROCESSED token lists from selected documents
                        (must be from the CSV 'tokens' column, not raw regex extraction)
        full_corpus_tokens: list of PREPROCESSED token lists from full corpus (for IDF computation)
        n: number of top keywords to return (default 15)
        min_doc_freq: minimum document frequency threshold (default 2)
    
    Returns:
        tuple: (list of (keyword, tfidf_score) tuples sorted by TF-IDF score descending,
                fallback_flag: bool indicating if fewer than 2 documents available)
    
    Raises:
        No exceptions, but returns empty results if input is empty or insufficient data
    """
    # Extended stop word list: common English filler words that survive TF-IDF anyway
    extended_stop_words = set([
        'also', 'make', 'use', 'using', 'used', 'one', 'way', 'can', 'will', 'may',
        'need', 'want', 'good', 'like', 'well', 'even', 'just', 'much', 'many', 'often',
        'best', 'help', 'get', 'give', 'keep', 'look', 'show', 'work', 'include',
        'new', 'other', 'time', 'user', 'data', 'system', 'may', 'would', 'should',
        'could', 'such', 'every', 'thing', 'set', 'case', 'part', 'group', 'high',
        'low', 'add', 'say', 'see', 'may', 'different', 'type', 'provide', 'create',
        'important', 'must', 'area', 'find', 'change', 'result', 'example', 'value',
        'state', 'contain', 'put', 'form', 'move', 'place', 'hold', 'take', 'allow',
        'apply', 'call', 'run', 'try', 'test', 'check', 'clear', 'close', 'come',
        'dashboard', 'interface', 'page', 'screen', 'click', 'view', 'display', 'button'
    ])
    
    # If fewer than 2 selected documents, fall back to raw frequency
    if not selected_tokens or len(selected_tokens) < 2:
        flat_tokens = [tok for toks in selected_tokens for tok in toks if isinstance(tok, str)]
        if not flat_tokens:
            return [], True
        # Use TF-IDF against full corpus instead of raw frequency
        selection_tf = pd.Series(flat_tokens).value_counts() / len(flat_tokens)
        keywords = []
        for tok, tf in selection_tf.items():
            if tok in extended_stop_words or len(tok) < 3 or tok.isdigit():
                continue
            docs_with_term = sum(1 for toks in full_corpus_tokens if tok in toks)
            if docs_with_term == 0:
                continue
            idf = np.log(len(full_corpus_tokens) / docs_with_term)
            keywords.append((tok, tf * idf))
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:n], True
    
    # Compute TF (term frequency in selected documents)
    flat_selected = [tok for toks in selected_tokens for tok in toks if isinstance(tok, str)]
    tf_counter = pd.Series(flat_selected).value_counts()
    
    # Compute document frequency in selected documents
    selected_doc_freq = {}
    for tokens_list in selected_tokens:
        unique_tokens = set(t for t in tokens_list if isinstance(t, str))
        for tok in unique_tokens:
            selected_doc_freq[tok] = selected_doc_freq.get(tok, 0) + 1
    
    # Apply minimum document frequency threshold (2 docs or 10% of selection, whichever is smaller)
    min_df_threshold = max(2, int(0.1 * len(selected_tokens)))
    min_df_threshold = min(min_df_threshold, len(selected_tokens))  # Ensure it doesn't exceed selection size
    
    viable_tokens = {
        tok: count for tok, count in tf_counter.items()
        if selected_doc_freq.get(tok, 0) >= min(min_df_threshold, 2)  # At least 2 docs minimum
    }
    
    # Compute IDF (inverse document frequency) from full corpus
    flat_corpus = [tok for toks in full_corpus_tokens for tok in toks if isinstance(tok, str)]
    corpus_doc_freq = {}
    for tokens_list in full_corpus_tokens:
        unique_tokens = set(t for t in tokens_list if isinstance(t, str))
        for tok in unique_tokens:
            corpus_doc_freq[tok] = corpus_doc_freq.get(tok, 0) + 1
    
    total_docs_in_corpus = max(len(full_corpus_tokens), 1)
    
    # Compute TF-IDF for each viable token
    tfidf_scores = {}
    for tok, tf_count in viable_tokens.items():
        if tok in extended_stop_words or len(tok) < 3 or tok.isdigit():
            continue  # Skip stop words, short tokens, and pure numbers
        
        # TF: normalized by total tokens in selection
        tf = tf_count / max(len(flat_selected), 1)
        
        # IDF: log(total_docs / docs_with_term)
        idf = np.log(total_docs_in_corpus / max(corpus_doc_freq.get(tok, 1), 1))
        
        # TF-IDF is product
        tfidf_scores[tok] = tf * idf
    
    # Sort by TF-IDF score descending and return top N
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:n], False  # False indicates normal TF-IDF mode


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


# ============ Embedding Trust / Sanity Check Utilities ============

def compute_keyword_overlap(doc1_tokens, doc2_tokens):
    """Compute Jaccard similarity between two token lists."""
    if not isinstance(doc1_tokens, list) or not isinstance(doc2_tokens, list):
        return 0.0
    set1 = set(str(t).lower() for t in doc1_tokens if t)
    set2 = set(str(t).lower() for t in doc2_tokens if t)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return float('inf')
    p1 = np.array(point1).flatten()
    p2 = np.array(point2).flatten()
    return float(np.linalg.norm(p1 - p2))


def get_colorblind_safe_palette():
    """Return a colorblind-safe categorical palette (Okabe-Ito + extended)."""
    # Designed for colorblind accessibility with up to 9 distinct colors
    return [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Red Orange
        '#CC79A7',  # Reddish Purple
        '#999999',  # Gray
        '#004225',  # Dark Green
    ]


def prepare_cohort_colors(saved_cohorts, df_work):
    """
    Prepare cohort color mapping with proper grouping for >6 cohorts.
    
    Returns:
        color_map: dict of cohort_name -> color
        cohort_names: list of cohort names in priority order
        cohort_sizes: dict of cohort_name -> count
    """
    if not saved_cohorts:
        return {}, [], {}
    
    palette = get_colorblind_safe_palette()
    
    # Calculate cohort sizes from actual presence in df_work
    cohort_sizes = {}
    for cohort_name, ids in saved_cohorts.items():
        count = len([did for did in ids if str(did) in df_work['doc_id'].astype(str).values])
        cohort_sizes[cohort_name] = count
    
    # Sort by size (descending) to prioritize larger cohorts for distinct colors
    sorted_cohorts = sorted(cohort_sizes.items(), key=lambda x: x[1], reverse=True)
    
    color_map = {}
    cohort_names = []
    
    # Assign colors to top 6 cohorts, group smaller ones into "Other"
    if len(sorted_cohorts) <= 6:
        # All cohorts get distinct colors
        for i, (cohort_name, size) in enumerate(sorted_cohorts):
            color_map[cohort_name] = palette[i % len(palette)]
            cohort_names.append(cohort_name)
    else:
        # Top 5 cohorts get distinct colors, remainder grouped
        for i in range(5):
            cohort_name, size = sorted_cohorts[i]
            color_map[cohort_name] = palette[i]
            cohort_names.append(cohort_name)
        
        # Remaining cohorts grouped into "Other"
        other_size = sum(size for _, size in sorted_cohorts[5:])
        if other_size > 0:
            color_map['Other'] = palette[7]  # Gray for "Other"
            cohort_names.append('Other')
    
    return color_map, cohort_names, cohort_sizes


def check_embedding_trust_for_pair(doc_id1, doc_id2, df, coords, keyword_overlap_threshold=0.15, 
                                   distance_threshold_pct=20):
    """
    Check if two documents have embedding-trust issues:
    - Are they visually close in projection?
    - Do they share enough keywords?
    Returns (is_suspicious, overlap, distance_pct, message)
    """
    try:
        # Get token data
        row1 = df[df['doc_id'].astype(str) == str(doc_id1)]
        row2 = df[df['doc_id'].astype(str) == str(doc_id2)]
        
        if row1.empty or row2.empty:
            return False, 0.0, 0.0, ""
        
        tokens1 = row1.iloc[0].get('__tokens', []) if '__tokens' in row1.columns else []
        tokens2 = row2.iloc[0].get('__tokens', []) if '__tokens' in row2.columns else []
        
        # Compute keyword overlap
        overlap = compute_keyword_overlap(tokens1, tokens2)
        
        # Get coordinates
        idx1 = df[df['doc_id'].astype(str) == str(doc_id1)].index[0]
        idx2 = df[df['doc_id'].astype(str) == str(doc_id2)].index[0]
        
        if coords is None or idx1 >= len(coords) or idx2 >= len(coords):
            return False, overlap, 0.0, ""
        
        point1 = coords[idx1]
        point2 = coords[idx2]
        distance = compute_euclidean_distance(point1, point2)
        
        # Normalize distance to percentile (rough approximation)
        # If distance < threshold, documents are "close"
        is_close = distance < distance_threshold_pct / 100.0
        has_low_overlap = overlap < keyword_overlap_threshold
        
        is_suspicious = is_close and has_low_overlap
        distance_pct = min(100.0, distance * 100)
        
        if is_suspicious:
            msg = f"⚠ Embedding trust: {doc_id1} and {doc_id2} are close in projection ({distance_pct:.1f}) but share few keywords ({overlap:.0%})"
            return True, overlap, distance_pct, msg
        
        return False, overlap, distance_pct, ""
    except Exception:
        return False, 0.0, 0.0, ""


def compute_global_embedding_health(df, coordinates, n_neighbors=5):
    """
    Compute average keyword overlap among nearest neighbors in embedding space.
    Returns health score (0-1) where 1 = perfect alignment, 0 = no alignment.
    """
    if coordinates is None or len(df) < 2 or '__tokens' not in df.columns:
        return None
    
    try:
        # Build nearest neighbors in embedding space
        n_neighbors = min(n_neighbors, len(df) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to include self
        nn.fit(coordinates)
        distances, indices = nn.kneighbors(coordinates)
        
        overlaps = []
        for i in range(len(df)):
            center_tokens = df.iloc[i].get('__tokens', [])
            if not isinstance(center_tokens, list) or not center_tokens:
                continue
            
            # Check overlap with neighbors (excluding self at index 0)
            for neighbor_idx in indices[i][1:]:
                neighbor_tokens = df.iloc[neighbor_idx].get('__tokens', [])
                overlap = compute_keyword_overlap(center_tokens, neighbor_tokens)
                overlaps.append(overlap)
        
        if overlaps:
            health_score = float(np.mean(overlaps))
            return health_score
        return None
    except Exception:
        return None


# ================================================================


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

@st.cache_data(show_spinner="Loading coordinates...")
def load_coordinate_files(_variant_key="default"):
    """Load all coordinate files for a specific variant, respecting bundle selection."""
    return {
        'tsne': find_file(aligned_candidates('coords_tsne.npy')),
        'umap': find_file(aligned_candidates('coords_umap.npy')),
        'pca': find_file(aligned_candidates('coords.npy')),
        'tfidf': find_file(aligned_candidates('tfidf_matrix.npz')),
        'cluster_labels': find_file(aligned_candidates('cluster_labels.npy')),
        'doc_ids': find_file(aligned_candidates('doc_ids.txt'))
    }


# Placeholder paths that will be updated dynamically based on variant
coords_tsne_path = None
coords_umap_path = None
coords_pca_path = None
tfidf_matrix_path = None
cluster_labels_path = None
doc_ids_path = None


@st.cache_data(show_spinner="Loading and processing dataset...")
def load_and_process_data(csv_path, doc_ids_path_arg, _variant_key="default", _version=2):
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
    
    # **CRITICAL**: Parse keywords from preprocessed token column ONLY.
    # The offline preprocessing pipeline (parse.py::preprocess_texts) creates the 'tokens' column.
    # This column contains lemmatized, stopword-filtered tokens.
    # We do NOT use regex fallback extraction because it causes keyword inconsistency.
    tok_col, av_kw = parse_keyword_space(df_local)
    
    if '__tokens' not in df_local.columns or all(len(t) == 0 for t in df_local['__tokens']):
        # Missing or empty token column indicates offline preprocessing was not run
        import warnings
        warnings.warn(
            "\n" + "="*70 + "\n"
            "CRITICAL: Token column missing or empty!\n"
            "="*70 + "\n\n"
            "The CSV file MUST have a 'tokens' column with preprocessed tokens from\n"
            "the offline pipeline (parse.py::preprocess_texts).\n\n"
            "This column contains:\n"
            "  - Lowercased text\n"
            "  - Lemmatized tokens (reduced to base form)\n"
            "  - Stopwords removed\n"
            "  - Tokens < 3 chars filtered\n\n"
            "Online keyword extraction REQUIRES this column for consistency with\n"
            "offline analysis. We do NOT use regex fallback extraction because:\n"
            "  - Regex returns raw text (no lemmatization)\n"
            "  - Results differ from offline pipeline\n"
            "  - Creates keyword inconsistency between offline & online\n\n"
            "TO FIX:\n"
            "  1. Run: python run_pipeline.py\n"
            "  2. Verify output CSV contains 'tokens' column\n"
            "  3. Restart Streamlit app\n"
            "="*70,
            UserWarning,
            stacklevel=3
        )
        # Initialize empty tokens to prevent app crash
        df_local['__tokens'] = df_local.apply(lambda row: [], axis=1)
        av_kw = []
        
    return df_local, tok_col, av_kw

# Execute cached loading - include variant key to make cache bundle-aware
variant_key = selected_bundle_root if selected_bundle_root else "default"
df, token_col, available_keywords = load_and_process_data(processed_csv_path, doc_ids_path, _variant_key=variant_key)

# Load coordinate files for the selected variant
coord_paths = load_coordinate_files(_variant_key=variant_key)
coords_tsne_path = coord_paths['tsne']
coords_umap_path = coord_paths['umap']
coords_pca_path = coord_paths['pca']
tfidf_matrix_path = coord_paths['tfidf']
cluster_labels_path = coord_paths['cluster_labels']
doc_ids_path = coord_paths['doc_ids']

def extract_cluster_topics(df_for_topics):
    """
    Extract cluster labels using inter-cluster TF-IDF.
    
    **QUALITY NOTE**: Cluster keywords are computed at runtime using inter-cluster TF-IDF,
    which is superior to offline frequency-based extraction. This ensures:
    - Keywords are distinctive to EACH cluster (not generic terms appearing everywhere)
    - Quality automatically improves when data changes or preprocessing varies
    - TF-IDF scores favor terms specific to individual clusters over globally common words
    
    **CRITICAL REQUIREMENT**: Input DataFrame MUST have a '__tokens' column containing
    PREPROCESSED token lists (lemmatized, stopword-filtered) from the offline pipeline.
    These come from the CSV 'tokens' column via parse.py::preprocess_texts.
    
    DO NOT use raw regex-extracted tokens — this breaks offline/online consistency.
    
    **Algorithm**:
    For each cluster, compute TF-IDF where:
    - TF = term frequency within the cluster
    - IDF = log(total_clusters / clusters_containing_term)
    - Top 3 terms with highest TF-IDF scores are selected
    - Extended stop words and short tokens (<3 chars) are filtered out
    
    This favors terms distinctive to ONE cluster, not generic terms appearing everywhere.
    Performance: ~5ms per cluster for typical datasets.
    
    Returns:
        dict mapping cluster_id -> "cluster_id: keyword1, keyword2, keyword3"
        Empty dict if __tokens column is missing or empty.
    """
    topic_map = {}
    if 'cluster' in df_for_topics.columns and '__tokens' in df_for_topics.columns:
        # Extended stop words
        extended_stop_words = set([
            'also', 'make', 'use', 'using', 'used', 'one', 'way', 'can', 'will', 'may',
            'need', 'want', 'good', 'like', 'well', 'even', 'just', 'much', 'many', 'often',
            'best', 'help', 'get', 'give', 'keep', 'look', 'show', 'work', 'include',
            'new', 'other', 'time', 'user', 'data', 'system', 'would', 'should',
            'could', 'such', 'every', 'thing', 'set', 'case', 'part', 'group', 'high',
            'low', 'add', 'say', 'see', 'may', 'different', 'type', 'provide', 'create',
            'important', 'must', 'area', 'find', 'change', 'result', 'example', 'value',
            'state', 'contain', 'put', 'form', 'move', 'place', 'hold', 'take', 'allow',
            'apply', 'call', 'run', 'try', 'test', 'check', 'clear', 'close', 'come',
            'dashboard', 'interface', 'page', 'screen', 'click', 'view', 'display', 'button'
        ])
        
        clusters = sorted(df_for_topics['cluster'].unique())
        
        for c in clusters:
            cluster_df = df_for_topics[df_for_topics['cluster'] == c]
            cluster_tokens = cluster_df['__tokens'].tolist()
            
            if not cluster_tokens:
                topic_map[c] = f"Cluster {c}"
                continue
            
            # Compute TF (term frequency within this cluster)
            flat_cluster_tokens = [tok for toks in cluster_tokens for tok in toks if isinstance(tok, str)]
            tf_counter = pd.Series(flat_cluster_tokens).value_counts()
            
            # Compute "inter-cluster IDF": in how many OTHER clusters does each term appear?
            # IDF = log(total_clusters / clusters_with_term)
            # High IDF = appears in few clusters (distinctive)
            inter_cluster_idf = {}
            for tok in tf_counter.index:
                if tok in extended_stop_words or len(tok) < 3 or tok.isdigit():
                    continue
                
                # Count how many clusters have this token
                clusters_with_tok = 0
                for other_c in clusters:
                    other_df = df_for_topics[df_for_topics['cluster'] == other_c]
                    other_tokens = [t for toks in other_df['__tokens'] for t in toks if isinstance(t, str)]
                    if tok in other_tokens:
                        clusters_with_tok += 1
                
                if clusters_with_tok > 0:
                    idf = np.log(len(clusters) / clusters_with_tok)
                    inter_cluster_idf[tok] = idf
            
            # Compute inter-cluster TF-IDF and sort
            tfidf_scores = {}
            for tok, tf_count in tf_counter.items():
                if tok not in inter_cluster_idf:
                    continue
                tf = tf_count / max(len(flat_cluster_tokens), 1)
                tfidf_scores[tok] = tf * inter_cluster_idf[tok]
            
            sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_keywords:
                top_words = [kw for kw, _ in sorted_keywords[:3]]
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

# Handle Year column specially: convert to numeric and add to numeric_metadata if present
if 'Year' in df.columns and 'Year' not in numeric_metadata:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    numeric_metadata.append('Year')

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

@st.cache_data
def load_doc_ids_and_index(path, _df):
    """Load doc_ids from file and build the doc_id to index mapping."""
    doc_ids_list = None
    if path:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                doc_ids_list = [line.strip() for line in f if line.strip()]
        except Exception:
            doc_ids_list = None
    
    if doc_ids_list:
        doc_id_to_global_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids_list)}
    else:
        doc_id_to_global_idx = {doc_id: idx for idx, doc_id in enumerate(_df['doc_id'].tolist())}
    
    return doc_ids_list, doc_id_to_global_idx

# read doc_ids list if available to align .npy indices
doc_ids_list, doc_id_to_global_idx = load_doc_ids_and_index(doc_ids_path, df)

if doc_ids_list and len(doc_ids_list) != len(df):
    # keep mismatch in warning list but ignore the file for alignment to avoid blank plots
    alignment_note = (
        f'doc_ids.txt length ({len(doc_ids_list)}) does not match processed CSV ({len(df)}). '
        'Falling back to dataframe order for alignment.'
    )
    alignment_notes.append(alignment_note)
    doc_ids_list = None
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

# Compute global embedding health metric (cached)
@st.cache_data
def _compute_embedding_health():
    # Use the 2D projection coordinates as proxy for visualization quality
    health = compute_global_embedding_health(df, coords_tsne, n_neighbors=5)
    return health

embedding_health_score = _compute_embedding_health()


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
    
    if color_mode == 'Year':
        # Color by temporal dimension (Year) using colorblind-safe sequential scale
        if 'Year' in df_plot.columns:
            # Convert Year to numeric, handling 'Unknown' and other non-numeric values
            try:
                year_numeric = pd.to_numeric(df_plot['Year'], errors='coerce')
                # Fill NaN/Unknown with median of numeric years, or default to 2023
                year_median = year_numeric.median()
                if pd.isna(year_median):
                    year_median = 2023  # Default if no numeric years found
                df_plot['__year'] = year_numeric.fillna(year_median)
            except Exception:
                # Fallback: assign a default year if conversion fails
                df_plot['__year'] = 2023
            
            fig = px.scatter(
                df_plot,
                x=xcol,
                y=ycol,
                color='__year',
                color_continuous_scale='Viridis',  # Colorblind-safe sequential scale
                hover_name='doc_id',
                hover_data=hover_dict,
                custom_data=['doc_id'],
            )
            
            # Update color bar label
            fig.update_coloraxes(colorbar_title="Year")
            
            # Apply selection/search highlighting on top
            if selected_ids or search_ids or focused_ids:
                df_plot['__status'] = 'Other'
                if search_ids:
                    df_plot.loc[df_plot['doc_id'].isin(search_ids), '__status'] = 'Search'
                    df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
                if selected_ids:
                    df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__status'] = 'Selected'
                    df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
                if focused_ids:
                    df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__status'] = 'Focused'
                    df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__size'] = 16
        else:
            # Fall back to selection mode if Year not available
            df_plot['__status'] = 'Other'
            if search_ids:
                df_plot.loc[df_plot['doc_id'].isin(search_ids), '__status'] = 'Search hit'
                df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
            if selected_ids:
                df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__status'] = 'Selected'
                df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
            if focused_ids:
                df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__status'] = 'Focused'
                df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__size'] = 16
            
            color_map = {
                'Focused': '#E63946',
                'Selected': '#E63946',
                'Search hit': '#F4A261',
                'Other': '#457B9D'
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
    
    elif color_mode == 'Cluster':
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
    
    elif color_mode == 'Saved Group':
        # Color by saved cohort membership with colorblind-safe palette
        saved_cohorts = {}  # Will be populated via __cohort column if it exists
        
        # Check if __cohort column has been prepared upstream
        if '__cohort' in df_plot.columns:
            # Extract cohort membership from the column
            for cohort_name in df_plot['__cohort'].unique():
                if cohort_name != 'Other' or True:  # Keep all cohorts including 'Other'
                    ids = df_plot[df_plot['__cohort'] == cohort_name]['doc_id'].tolist()
                    saved_cohorts[cohort_name] = ids
        
        if saved_cohorts:
            # Get colorblind-safe colors and cohort info
            color_map, cohort_names, cohort_sizes = prepare_cohort_colors(saved_cohorts, df_plot)
            
            # Add size highlighting for selected/search hits
            if selected_ids:
                df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
            if search_ids:
                df_plot.loc[df_plot['doc_id'].isin(search_ids), '__size'] = 12
            if focused_ids:
                df_plot.loc[df_plot['doc_id'].isin(focused_ids), '__size'] = 16
            
            # Map cohorts > 6 to "Other" in df_plot based on color_map
            if len(saved_cohorts) > 6:
                df_plot['__cohort_grouped'] = df_plot['__cohort']
                for cohort_name in saved_cohorts.keys():
                    if cohort_name not in color_map:
                        df_plot.loc[df_plot['__cohort'] == cohort_name, '__cohort_grouped'] = 'Other'
                cohort_col = '__cohort_grouped'
            else:
                cohort_col = '__cohort'
            
            fig = px.scatter(
                df_plot,
                x=xcol,
                y=ycol,
                color=cohort_col,
                color_discrete_map=color_map,
                hover_name='doc_id',
                hover_data=hover_dict,
                custom_data=['doc_id'],
                category_orders={cohort_col: cohort_names}
            )
            # Legend will be automatically generated by Plotly with cohort names and colors
            # and styled by the general layout settings below
        else:
            # Fallback to selection mode if no cohorts
            df_plot['__status'] = 'Other'
            if selected_ids:
                df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__status'] = 'Selected'
                df_plot.loc[df_plot['doc_id'].isin(selected_ids), '__size'] = 14
            
            color_map = {'Selected': '#E63946', 'Other': '#457B9D'}
            fig = px.scatter(
                df_plot,
                x=xcol,
                y=ycol,
                color='__status',
                color_discrete_map=color_map,
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
            elif color_mode == 'Year':
                # For Year mode, preserve year-based coloring but highlight selections
                if selected_ids or search_ids or focused_ids:
                    opacities = []
                    sizes = []
                    l_widths = []
                    l_colors = []
                    m_symbols = []
                    
                    for cdata in trace.customdata:
                        did = cdata[0]
                        if focused_ids and did in focused_ids:
                           opacities.append(1.0)
                           sizes.append(18)
                           l_widths.append(3)
                           l_colors.append('white')
                           m_symbols.append('x')
                        elif selected_ids and did in selected_ids:
                           opacities.append(1.0)
                           sizes.append(14)
                           l_widths.append(2)
                           l_colors.append('white')
                           m_symbols.append('diamond')
                        elif search_ids and did in search_ids:
                           opacities.append(1.0)
                           sizes.append(10)
                           l_widths.append(1.5)
                           l_colors.append('white')
                           m_symbols.append('star')
                        else:
                           opacities.append(0.4)
                           sizes.append(5)
                           l_widths.append(0.0)
                           l_colors.append('rgba(0,0,0,0.0)')
                           m_symbols.append('circle')
                    trace.marker.opacity = opacities
                    trace.marker.size = sizes
                    trace.marker.line = dict(width=l_widths, color=l_colors)
                    trace.marker.symbol = m_symbols
                else:
                    trace.marker.size = 7
                    trace.marker.opacity = 0.8
                    trace.marker.line = dict(width=0.5, color='rgba(0,0,0,0.2)')
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
                           opacities.append(0.05) # heavily faded
                           sizes.append(3)
                           l_widths.append(0.0)
                           l_colors.append('rgba(0,0,0,0.0)')
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
        xaxis=dict(title=f"{title} Dim 1", showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0', scaleanchor="y", scaleratio=1, range=[-1.1, 1.1]),
        yaxis=dict(title=f"{title} Dim 2", showgrid=True, gridwidth=0.5, gridcolor='#E0E0E0', scaleanchor="x", scaleratio=1, range=[-1.1, 1.1]),
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
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ''
if 'search_query_persist' not in st.session_state:
    st.session_state['search_query_persist'] = st.session_state.get('search_query', '')
if 'search_query_input' not in st.session_state:
    st.session_state['search_query_input'] = st.session_state.get('search_query_persist', '')
if 'visible_doc_ids' not in st.session_state:
    st.session_state['visible_doc_ids'] = df['doc_id'].astype(str).tolist()
if 'orient_me_dismissed' not in st.session_state:
    st.session_state['orient_me_dismissed'] = False

# ===== SESSION HISTORY TRACKING =====
if 'history_log' not in st.session_state:
    st.session_state['history_log'] = []
if 'history_expanded' not in st.session_state:
    st.session_state['history_expanded'] = False

# ===== PROJECTION EXPANSION STATE =====
if 'expanded_projection' not in st.session_state:
    st.session_state['expanded_projection'] = None  # None, 'tsne', 'umap', or 'pca'

# ===== PIPELINE CONFIGURATION & SNAPSHOTS =====
if 'pipeline_snapshots' not in st.session_state:
    st.session_state['pipeline_snapshots'] = {}  # {snapshot_name: {stage_configs + timestamp}}
if 'pipeline_locked_stages' not in st.session_state:
    st.session_state['pipeline_locked_stages'] = set()  # Locked stage names
if 'compare_snapshots' not in st.session_state:
    st.session_state['compare_snapshots'] = False  # Toggle for snapshot diff visualization

from datetime import datetime

def add_history_entry(action_type, description, state_snapshot=None):
    """Log a user action with timestamp and optional state snapshot for restoration."""
    entry = {
        'timestamp': datetime.now().strftime('%I:%M%p').lower(),  # "2:14pm" format
        'action_type': action_type,
        'description': description,
        'state_snapshot': state_snapshot or {},
        'full_timestamp': datetime.now()  # For export sorting
    }
    st.session_state['history_log'].append(entry)

def create_state_snapshot():
    """Capture current session state for restoration."""
    return {
        'selected_ids': st.session_state.get('selected_ids', []),
        'search_hits': st.session_state.get('search_hits', []),
        'search_query_persist': st.session_state.get('search_query_persist', ''),
        'visible_doc_ids': st.session_state.get('visible_doc_ids', []),
    }

def restore_state_from_entry(entry):
    """Restore session state from a history log entry."""
    if not entry.get('state_snapshot'):
        return
    
    snapshot = entry['state_snapshot']
    st.session_state['selected_ids'] = snapshot.get('selected_ids', [])
    st.session_state['search_hits'] = snapshot.get('search_hits', [])
    st.session_state['search_query_persist'] = snapshot.get('search_query_persist', '')
    st.session_state['visible_doc_ids'] = snapshot.get('visible_doc_ids', [])
    st.session_state['search_query'] = snapshot.get('search_query_persist', '')
    st.rerun()

# ===== PIPELINE CONFIGURATION & METRICS =====

def get_current_pipeline_config():
    """Extract current active pipeline configuration."""
    config = {
        'preprocessing': {
            'lemmatization': True,  # Default assumption based on parse.py
            'remove_stopwords': True,
            'min_word_length': 3,
            'lowercase': True,
        },
        'vectorization': {
            'method': 'tfidf',
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': (1, 2),
        },
        'embedding': {
            'method': 'pca/svd',
            'n_components': 50,  # High-dim intermediate
        },
        'projection': {
            'available_methods': ['pca', 'tsne', 'umap'],
            'current_method': 'pca',  # Will be enhanced with actual UI selection
        }
    }
    return config

def compute_stage_metrics():
    """Compute quality metrics for each pipeline stage."""
    metrics = {
        'preprocessing': {'metric': 'tokens', 'value': 'Extracted'},
        'vectorization': {'metric': 'vocab_size', 'value': '~5000'},
        'embedding': {'metric': 'variance', 'value': 'N/A'},
        'projection': {'metric': 'health', 'value': f'{embedding_health_score:.0%}' if embedding_health_score else 'N/A'},
    }
    
    # Try to compute embedding variance from PCA
    try:
        if len(base_embeddings) >= 2:
            from sklearn.decomposition import PCA
            pca_test = PCA(n_components=min(5, base_embeddings.shape[1]))
            pca_test.fit(base_embeddings)
            var_explained = sum(pca_test.explained_variance_ratio_) * 100
            metrics['embedding']['value'] = f'{var_explained:.1f}%'
    except Exception:
        pass
    
    return metrics

# Compact status bar (filled in later after filtering runs)
status_placeholder = st.empty()



# Guided Tour / Getting Started
with st.sidebar:
    with st.expander('Getting Started', expanded=True):
        # Orient Me: Smart onboarding subsection (will be populated after df_work is computed)
        orient_me_container = st.container(border=False)
        
        st.write("1. **Load Data**: The latest dataset is auto-loaded.")
        st.write("2. **Explore**: Use the 3 projections to see document relationships.")
        st.write("3. **Select**: Lasso-select points to highlight them across all views.")
        st.write("4. **Analyze**: Check the 'Analysis' tab for cluster stats and details.")
        
        # Store container reference for later population
        st.session_state['_orient_me_container'] = orient_me_container

# Add CSS to remove border from orient_me_container and fix button wrapping
st.markdown("""
<style>
    div[data-testid="stContainer"] > div:has(div) .stExpander > div:first-child {
        border: none !important;
    }
    button[key*="orient_select_cluster_"] {
        white-space: nowrap !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Cluster Keywords Quality Note
    with st.expander("ℹ️ About Cluster Keywords", expanded=False):
        st.markdown("""
**Cluster keywords are computed at runtime using Inter-Cluster TF-IDF**, which ranks terms by how distinctive they are to each cluster:

- **TF** = frequency of the term within the cluster
- **IDF** = log(total_clusters / clusters_containing_term)
- **Result** = top 3 distinctive terms per cluster

**Why this is better than offline frequency counting:**
- Keywords reflect what makes each cluster **different** (not just common globally)
- 🔄 Quality automatically improves when data/preprocessing changes
- Computed at load time (~5ms per cluster)
- Consistent with preprocessed tokens from the offline pipeline

**Quality indicators:**
- Green keywords = well-separated, distinctive cluster
- Vague keywords = clusters may overlap; plot projection for confirmation
        """)

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
        
        # Display variant information panel
        variant_info = get_variant_comparison_info(selected_bundle_root)
        current_variant = variant_info["current"]
        
        st.markdown("**Preprocessing Settings:**")
        variant_col1, variant_col2 = st.columns([1, 2])
        with variant_col1:
            st.metric(label="Variant", value=current_variant["label"])
        with variant_col2:
            st.write(f"_{current_variant['settings']}_")
        
        # Show differences from default if not default
        if variant_info["differences"]:
            with st.expander("🔄 Differences from Default"):
                for diff in variant_info["differences"]:
                    st.write(f"• {diff}")
        
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
            options=['Selection', 'Cluster', 'Saved Group', 'Year'],
            help='Switch between showing selection status (Red/Blue), semantic clusters (Multi-color), Saved Groups, or temporal distribution (Year).'
        )
        
        # Note about cluster keyword quality
        if color_mode == 'Cluster':
            st.info(
                "**Cluster Keywords** are computed at runtime using Inter-Cluster TF-IDF, which ranks terms by how distinctive they are to each cluster. "
                "This is more accurate than offline frequency-based extraction. "
                "Expand 'Clustering Settings' → 'About Cluster Keywords' for details.",
                icon="ℹ️"
            )
        
        st.markdown('**Visual Tweaks**')
        point_alpha = st.slider('Point Opacity', 0.1, 1.0, 0.7, 0.1)
        point_size_scale = st.slider('Point Size', 0.5, 2.0, 1.0, 0.1)

        show_download_buttons = st.checkbox('Show plot download buttons', value=False, help='Enable this to download high-resolution PNGs of the current plots for reports or presentations.')
        show_hover = st.checkbox('Show tooltip on hover', value=True, help='Uncheck to completely hide the hover tooltips for a cleaner view.')

    # ===== PIPELINE CONFIGURATION PANEL =====
    with st.expander('Pipeline', expanded=False):
        st.markdown('**Active Pipeline Stages**')
        
        # Help text explaining the panel layout
        with st.expander('Help: Understanding the Pipeline Panel', expanded=False):
            st.markdown("""
            **Left column (Lock toggle):**
            - LOCKED = Stage locked (method cannot be changed)
            - UNLOCKED = Stage unlocked (you can swap the method)
            
            **Middle column (Method dropdown):**
            - Choose which algorithm to use for this stage
            
            **Metric column:**
            - **Extracted/Tokens**: How many unique words in your vocabulary
            - **Vocab Size**: Total unique terms (e.g., ~5000)
            - **Variance %**: How much information the embedding preserves (higher is better)
            - **Health %**: Embedding coherence score (11% = scattered, neighbors are distant)
            
            **Right indicator (OK or LOCKED):**
            - OK = Stage completed successfully
            - LOCKED = Stage is locked, cannot be modified
            
            **Key Insight:** Low embedding health (11%) means embeddings are scattered. Try adjusting TF-IDF or using UMAP instead of PCA.
            """)
        
        # Get current config and metrics
        pipeline_cfg = get_current_pipeline_config()
        stage_metrics = compute_stage_metrics()
        
        # Define the 4 pipeline stages
        stages = [
            ('Preprocessing/Lemmatization', 'preprocessing', ['lemmatize', 'lowercase', 'stopword-removal']),
            ('Vectorization/TF-IDF', 'vectorization', ['tfidf']),
            ('Embedding', 'embedding', ['pca', 'svd']),
            ('Projection', 'projection', ['pca', 'tsne', 'umap']),
        ]
        
        pipeline_status = []
        
        for stage_label, stage_key, alternatives in stages:
            sc1, sc2, sc3, sc4 = st.columns([2.5, 1.5, 1.2, 0.6])
            
            # Stage name with lock toggle
            with sc1:
                is_locked = stage_key in st.session_state['pipeline_locked_stages']
                lock_label = "[LOCKED] " if is_locked else "[UNLOCKED] "
                toggle_help = "Click to lock/unlock this stage (prevents accidental method changes)"
                if st.checkbox(f"{lock_label}{stage_label}", 
                             value=not is_locked,
                             key=f"stage_active_{stage_key}",
                             label_visibility="collapsed",
                             help=toggle_help):
                    if is_locked:
                        st.session_state['pipeline_locked_stages'].discard(stage_key)
                    else:
                        st.session_state['pipeline_locked_stages'].add(stage_key)
            
            # Dropdown for alternatives
            with sc2:
                cfg = pipeline_cfg.get(stage_key, {})
                current_method = cfg.get('method', alternatives[0] if alternatives else 'default')
                selected = st.selectbox(
                    f"Method##_{stage_key}",
                    options=alternatives or ['default'],
                    index=0,
                    key=f"pipeline_method_{stage_key}",
                    label_visibility="collapsed",
                    help=f"Active algorithm for {stage_label}"
                )
            
            # Quality metric
            with sc3:
                metric_info = stage_metrics.get(stage_key, {})
                metric_label = metric_info.get('metric', '?')
                metric_value = metric_info.get('value', 'N/A')
                
                # Add tooltip based on metric type
                metric_tooltip = {
                    'Extracted': 'Unique tokens extracted (vocabulary size)',
                    'Vocab': 'Size of TF-IDF vocabulary',
                    'Variance': 'Information preserved by embedding (% of total)',
                    'Health': 'Embedding coherence score (higher is better, 11% = scattered neighbors)'
                }.get(metric_label, 'Quality metric for this stage')
                
                st.metric(metric_label, metric_value, label_visibility="collapsed", help=metric_tooltip)
            
            # Status indicator
            with sc4:
                status_text = "LOCKED" if is_locked else "OK"
                status_tooltip = "Stage locked (cannot modify)" if is_locked else "Stage completed"
                st.caption(status_text, help=status_tooltip)
            
            pipeline_status.append({
                'stage': stage_key,
                'method': selected,
                'locked': is_locked,
                'metric': metric_value
            })
        
        st.divider()
        
        # Save Snapshot UI
        st.markdown('**Pipeline Snapshots**')
        col_snap1, col_snap2 = st.columns([2, 1])
        
        with col_snap1:
            snapshot_name = st.text_input('Snapshot name', placeholder='e.g., "baseline", "high-variance"', key='snapshot_name_input', label_visibility="collapsed")
        
        with col_snap2:
            if st.button('Save Snapshot', key='save_pipeline_snapshot'):
                if snapshot_name and snapshot_name.strip():
                    snapshot_data = {
                        'timestamp': datetime.now().isoformat(),
                        'config': pipeline_cfg,
                        'metrics': stage_metrics,
                        'status': pipeline_status,
                    }
                    st.session_state['pipeline_snapshots'][snapshot_name.strip()] = snapshot_data
                    st.success(f'Saved snapshot: {snapshot_name.strip()}')
                    st.session_state['snapshot_name_input'] = ''
                else:
                    st.warning('Enter a snapshot name')
        
        # List existing snapshots
        if st.session_state['pipeline_snapshots']:
            st.markdown('**Saved Snapshots**')
            snapshot_names = list(st.session_state['pipeline_snapshots'].keys())
            
            for snap_name in snapshot_names:
                snap_data = st.session_state['pipeline_snapshots'][snap_name]
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.caption(f"📦 {snap_name}")
                    ts = snap_data.get('timestamp', '')
                    if ts:
                        st.caption(f"_{ts.split('T')[0]}_")
                
                with col2:
                    if st.button('View', key=f'view_snap_{snap_name}', use_container_width=True):
                        st.session_state['selected_snapshot_view'] = snap_name
                
                with col3:
                    if st.button('Delete', key=f'del_snap_{snap_name}', use_container_width=True):
                        del st.session_state['pipeline_snapshots'][snap_name]
                        st.rerun()
        
        # Diff toggle when 2+ snapshots exist
        if len(st.session_state['pipeline_snapshots']) >= 2:
            st.divider()
            st.markdown('**Snapshot Comparison**')
            compare_enabled = st.checkbox(
                'Show projection diff when comparing snapshots',
                value=st.session_state.get('compare_snapshots', False),
                key='compare_snapshots_toggle',
                help='Enable side-by-side visualization diff across saved pipeline configurations'
            )
            if compare_enabled != st.session_state.get('compare_snapshots', False):
                st.session_state['compare_snapshots'] = compare_enabled

    with st.expander('Search & Filter', expanded=False):
        # Improved Search
        search_query = st.text_input('Search doc_id / keyword / phrase / regex', key='search_query_input')
        st.session_state['search_query'] = search_query
        st.session_state['search_query_persist'] = search_query
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
        
        # Temporal Analysis Guidance
        if 'Year' in numeric_metadata:
            st.caption(
                "ℹ️ Tip: Switch color to **Year** (Display Mode) to see corpus evolution. Use the Year range slider to filter time windows and watch topics shift across the projection."
            )
        
        max_points = st.slider('Max points to display', min_value=50, max_value=3000, value=1200, step=50, help="Reduce this number if the app feels slow. Limits how many dots are drawn.")

    # Saved Group Management (Persistent Selection)
    with st.expander('Saved Groups', expanded=True):
        if 'saved_cohorts' not in st.session_state:
            st.session_state['saved_cohorts'] = {}
        
        # Save current selection
        current_sel_len = len(st.session_state['selected_ids'])
        new_cohort_name = st.text_input('New saved group name', placeholder='e.g., Outliers, Group A')
        if st.button('Save current selection') and new_cohort_name and current_sel_len > 0:
            st.session_state['saved_cohorts'][new_cohort_name] = list(st.session_state['selected_ids'])
            add_history_entry(
                'cohort_saved',
                f"Saved group '{new_cohort_name}' with {current_sel_len} document{'s' if current_sel_len != 1 else ''}",
                create_state_snapshot()
            )
            st.success(f"Saved {current_sel_len} docs to '{new_cohort_name}'")
            st.rerun()

        # Display saved cohorts
        if st.session_state['saved_cohorts']:
            st.markdown('**Saved Groups**')
            cohorts_to_delete = []
            for name, ids in st.session_state['saved_cohorts'].items():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    if st.button(f"{name} ({len(ids)})", key=f"load_{name}", width="stretch"):
                        update_selection(ids, additive=st.session_state.get('additive_mode', False))
                        add_history_entry(
                            'cohort_loaded',
                            f"Loaded group '{name}' with {len(ids)} document{'s' if len(ids) != 1 else ''}",
                            create_state_snapshot()
                        )
                with c2:
                    st.write("") # Spacer
                with c3:
                    if st.button('Delete', key=f"del_{name}"):
                        cohorts_to_delete.append(name)
            
            if cohorts_to_delete:
                for name in cohorts_to_delete:
                    del st.session_state['saved_cohorts'][name]
                st.rerun()
        else:
            st.caption("No saved groups yet.")

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
            if st.button('Clear', key='btn_clear_selection', width="stretch"):
                update_selection([], additive=False)
        with qcol2:
            if st.button('Undo', key='btn_undo', width="stretch", disabled=len(st.session_state.get('selection_history', [])) == 0):
                history = st.session_state.get('selection_history', [])
                if history:
                    st.session_state['selected_ids'] = history.pop()
                    st.session_state['selection_history'] = history
                    st.rerun()
        with qcol3:
            if st.button('Random', key='btn_random', width="stretch", help='Select 10 random documents to explore the dataset.'):
                import random
                random_ids = random.sample(doc_options, min(10, len(doc_options)))
                update_selection(random_ids, additive=False)
        
        # Brushing modes
        qcol4, qcol5 = st.columns(2)
        with qcol4:
            if st.button('Invert', key='btn_invert', width="stretch", help='Select everything that is NOT currently selected.'):
                all_ids = set(st.session_state.get('visible_doc_ids', df['doc_id'].astype(str).tolist()))
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
                width="stretch"
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
                   width="stretch"
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
        if st.button('Save Session', width="stretch"):
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
                if st.button('Load Session', width="stretch") and load_session:
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
            if st.button('Build chunk â†” parent map', key='btn_build_chunk_map'):
                try:
                    chunk_parent_map, parent_chunk_map = apply_chunk_mapping(df, chunk_col, parent_col)
                    st.success(f'Built mapping with {len(chunk_parent_map)} chunk entries')
                    st.session_state['chunk_parent_map'] = chunk_parent_map
                    st.session_state['parent_chunk_map'] = parent_chunk_map
                except Exception as exc:
                    st.error('Failed to build mapping: ' + str(exc))
        else:
            st.caption('Provide both chunk and parent columns to enable chunkâ†”parent linking')

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
    with st.spinner('Recomputing embeddings preview â€¦'):
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

# Log search action if query was performed
if search_query and search_query != st.session_state.get('last_logged_search', ''):
    query_preview = search_query[:40] + ('...' if len(search_query) > 40 else '')
    add_history_entry(
        'search',
        f"Searched '{query_preview}', {len(run_search(df.copy(), search_query, search_scopes))} results visible",
        create_state_snapshot()
    )
    st.session_state['last_logged_search'] = search_query

df_work = run_search(df_work, search_query, search_scopes)
df_work = df_work.reset_index(drop=True)
if len(df_work) > max_points:
    df_work = df_work.iloc[:max_points]

st.session_state['visible_doc_ids'] = df_work['doc_id'].astype(str).tolist()

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
        
        # Log selection action
        num_selected = len(st.session_state.get('selected_ids', []))
        top_keyword = ''
        if selected_docs and '__tokens' in df_work.columns:
            from collections import Counter
            all_tokens = []
            for doc_id in selected_docs[:5]:  # Check first 5 selected docs
                token_list = df_work[df_work['doc_id'] == doc_id].iloc[0].get('__tokens', [])
                if isinstance(token_list, list):
                    all_tokens.extend(token_list)
            if all_tokens:
                top_keyword = f", top keyword: {Counter(all_tokens).most_common(1)[0][0]}"
        
        add_history_entry(
            'selection',
            f"Selected {num_selected} document{'s' if num_selected != 1 else ''}{top_keyword}",
            create_state_snapshot()
        )
        
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

def generate_findings_summary(avg_similarity=None):
    """Generate a markdown summary of current session state for export.
    
    Args:
        avg_similarity: Optional float of average pairwise similarity (or None if not computed)
    """
    
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    selected_ids = st.session_state.get('selected_ids', [])
    selected_count = len(selected_ids)
    search_q = st.session_state.get('search_query_persist', '')
    saved_cohorts = st.session_state.get('saved_cohorts', {})
    pipeline_snapshots = st.session_state.get('pipeline_snapshots', {})
    
    # Build findings text
    findings = []
    findings.append("# Session Findings Export")
    findings.append("")
    findings.append(f"Exported: {timestamp}")
    findings.append("")
    
    # Active Filters
    findings.append("## Active Filters")
    active_filters = []
    if search_q:
        active_filters.append(f"- Search Query: {search_q}")
    if 'cluster' in df.columns:
        unique_clusters = df['cluster'].unique()
        if len(unique_clusters) > 0:
            active_filters.append(f"- Cluster Filter: Active")
    if active_filters:
        findings.extend(active_filters)
    else:
        findings.append("(No filters applied; all documents visible)")
    findings.append("")
    
    # Selection Summary
    findings.append("## Selection Summary")
    total_docs = len(df)
    findings.append(f"- Selected Documents: {selected_count} / {total_docs}")
    findings.append("")
    
    # Top Keywords in Selection
    if selected_count > 0:
        findings.append("## Top Keywords in Selection")
        try:
            selected_df = df_work[df_work['doc_id'].isin(selected_ids)]
            if '__tokens' in selected_df.columns:
                all_tokens = []
                for tokens in selected_df['__tokens']:
                    all_tokens.extend(tokens)
                if all_tokens:
                    token_counts = pd.Series(all_tokens).value_counts()
                    top_10 = token_counts.head(10)
                    for keyword, count in top_10.items():
                        findings.append(f"- {keyword}: {count}")
                else:
                    findings.append("(No tokens found in selected documents)")
            else:
                findings.append("(Token data not available)")
        except Exception:
            findings.append("(Unable to extract keywords)")
        findings.append("")
    
    # Average Pairwise Similarity (passed in)
    findings.append("## Semantic Cohesion")
    if avg_similarity is not None:
        if avg_similarity > 0.6:
            interp = "High (similar topics)"
        elif avg_similarity > 0.3:
            interp = "Moderate (mixed)"
        else:
            interp = "Low (diverse)"
        findings.append(f"- Average Pairwise Similarity: {avg_similarity:.3f} ({interp})")
    else:
        if selected_count < 2:
            findings.append("(Need at least 2 documents for similarity)")
        elif selected_count > 50:
            findings.append(f"(Too many docs selected: {selected_count}; limit is 50)")
        else:
            findings.append("(Similarity not computed)")
    findings.append("")
    
    # Saved Cohorts
    if saved_cohorts:
        findings.append("## Saved Cohorts")
        for cohort_name, cohort_ids in saved_cohorts.items():
            findings.append(f"- {cohort_name}: {len(cohort_ids)} documents")
        findings.append("")
    
    # Current Pipeline Configuration
    findings.append("## Pipeline Configuration")
    try:
        pipeline_cfg = get_current_pipeline_config()
        if pipeline_cfg:
            findings.append("- Preprocessing: " + pipeline_cfg.get('preprocessing', {}).get('method', 'unknown'))
            findings.append("- Vectorization: " + pipeline_cfg.get('vectorization', {}).get('method', 'unknown'))
            findings.append("- Embedding: " + pipeline_cfg.get('embedding', {}).get('method', 'unknown'))
            findings.append("- Projection: " + pipeline_cfg.get('projection', {}).get('method', 'unknown'))
        else:
            findings.append("(Default pipeline)")
    except Exception:
        findings.append("(Unable to retrieve pipeline config)")
    findings.append("")
    
    # Latest Pipeline Snapshot
    if pipeline_snapshots:
        findings.append("## Latest Pipeline Snapshot")
        latest_snap_name = list(pipeline_snapshots.keys())[-1]
        latest_snap = pipeline_snapshots[latest_snap_name]
        findings.append(f"- Snapshot Name: {latest_snap_name}")
        findings.append(f"- Timestamp: {latest_snap.get('timestamp', 'N/A')}")
    
    findings.append("")
    findings.append("---")
    findings.append("*End of Export*")
    
    return "\n".join(findings)

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

# Populate the Orient Me sidebar container (computed after active_filters and df_work are ready)
if '_orient_me_container' in st.session_state:
    with st.session_state['_orient_me_container']:
        if not st.session_state.get('orient_me_dismissed', False):
            sel_count_for_orient = len(st.session_state.get('selected_ids', []))
            should_expand_orient = (sel_count_for_orient == 0 and len(active_filters) == 0 and 'cluster' in df_work.columns)
            
            with st.expander("Orient Me", expanded=should_expand_orient):
                # Dismiss button inline with content
                if st.button("×", key="orient_me_dismiss", help="Dismiss for this session"):
                    st.session_state['orient_me_dismissed'] = True
                    st.rerun()
                
                if 'cluster' in df_work.columns:
                    # Get top 5 clusters, explicitly sorted by document count descending
                    cluster_counts = df_work['cluster'].value_counts().sort_values(ascending=False).head(5)
                    
                    for cluster_id, cluster_size in cluster_counts.items():
                        cluster_slice = df_work[df_work['cluster'] == cluster_id]
                        
                        # Use pre-computed GLOBAL_CLUSTER_MAP (inter-cluster TF-IDF based labels)
                        cluster_label = GLOBAL_CLUSTER_MAP.get(cluster_id, f"Cluster {cluster_id}")
                        # Extract just the descriptor part after the colon (e.g., "neural • network • deep")
                        if ': ' in cluster_label:
                            keywords_html = cluster_label.split(': ', 1)[1]
                        else:
                            keywords_html = cluster_label
                        
                        # Layout: cluster_label + keywords + select button
                        c1, c2, c3 = st.columns([1.2, 2, 0.8])
                        
                        with c1:
                            density_pct = (cluster_size / max(1, len(df_work))) * 100
                            st.write(f"**Cluster {cluster_id}**")
                            st.caption(f"{cluster_size} docs ({density_pct:.0f}%)")
                        
                        with c2:
                            st.caption(keywords_html)
                        
                        with c3:
                            if st.button("Select", key=f"orient_select_cluster_{cluster_id}"):
                                ids = cluster_slice['doc_id'].astype(str).tolist()
                                update_selection(ids, additive=False)
                                st.rerun()

with status_placeholder.container():

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Total", f"{total_n:,}")
    sc2.metric("Visible", f"{filtered_n:,}")
    sc3.metric("Selected", f"{sel_n:,}")
    sc4.metric("Filters", len(active_filters))
    
    # Embedding health indicator with info popover — PROMINENT DISPLAY
    if embedding_health_score is not None:
        # Use more expressive visual hierarchy
        health_pct = embedding_health_score * 100
        if embedding_health_score >= 0.5:
            health_icon = "[OK]"
            health_label = "GOOD"
            health_visual = f"{health_icon} {health_label}"
        elif embedding_health_score >= 0.3:
            health_icon = "[WARNING]"
            health_label = "FAIR"
            health_visual = f"{health_icon} {health_label}"
        else:
            health_icon = "[POOR]"
            health_label = "POOR"
            health_visual = f"{health_icon} {health_label}"
        
        with sc5:
            hc_col, hc_help = st.columns([3, 1])
            with hc_col:
                # Use st.metric for consistent styling, but add visual emphasis
                st.metric(
                    "Embedding Quality",
                    f"{health_visual} ({health_pct:.0f}%)",
                    delta=None
                )
            with hc_help:
                with st.popover("ℹ️", use_container_width=False):
                    st.markdown(f"""
**Score: {health_pct:.1f}%**

Measures semantic coherence in the 2D projection.

**How computed:**
- 5-nearest neighbors for each doc in 2D space
- Jaccard similarity of keywords between doc and neighbors
- Average across all docs (0-100%)

**Thresholds:**
- ≥50%: Good coherence
- 30-50%: Fair (caution advised)
- 🚨 <30%: Poor (misleading)

**Interpretation:**
If low, nearby visual distance ≠ semantic similarity.
                    """)
    else:
        sc5.metric("Embedding Quality", "N/A")
    
    if active_filters:
        st.caption(" Â· ".join(active_filters))

if len(df_work) < len(df):
    rc1, rc2 = st.columns([4, 1])
    with rc2:
        if st.button('Reset Filters', width="stretch"):
            st.rerun()

tab_explore = st.container()

with tab_explore:
    # === EMBEDDING QUALITY BANNER ===
    # Surface embedding health prominently at the TOP of the page
    if embedding_health_score is not None:
        # Determine severity: alert at 0.5, warning at 0.3, critical below 0.2
        if embedding_health_score >= 0.5:
            # Good health - show subtle info
            health_icon = "[OK]"
            health_status = "GOOD"
            health_color = "green"
            health_message = f"Embeddings are healthy ({embedding_health_score:.0%} — nearby docs are semantically similar)"
            banner_type = "success"
        elif embedding_health_score >= 0.3:
            # Fair health - show warning
            health_icon = "[WARNING]"
            health_status = "FAIR"
            health_color = "orange"
            health_message = (f"**Moderate embedding quality ({embedding_health_score:.0%})** — "
                            "Some nearby docs may not be semantically similar. "
                            "Results should be interpreted with caution.")
            banner_type = "warning"
        else:
            # Poor health - show alert
            health_icon = "[ALERT]"
            health_status = "POOR"
            health_color = "red"
            health_message = (f"**POOR EMBEDDING QUALITY ({embedding_health_score:.0%})** — "
                            "Nearby documents are semantically distant. "
                            "This projection may be misleading. Consider regenerating with different settings (TF-IDF, UMAP, etc.)")
            banner_type = "error"
        
        # Display banner using the appropriate Streamlit alert method
        if banner_type == "error":
            st.error(f"{health_icon} {health_message}")
        elif banner_type == "warning":
            st.warning(f"{health_icon} {health_message}")
        else:
            st.success(f"{health_icon} {health_message}")
        
        # Add compact explanation in expander
        with st.expander("What is embedding quality? Why does it matter?", expanded=False):
            st.markdown(f"""
**Embedding Quality Score: {embedding_health_score:.1%}**

Measures semantic coherence of the projection by checking if nearby documents in 2D space have similar keywords.

**How it works:**
- Finds 5 nearest neighbors for each document in the 2D projection
- Computes keyword overlap (Jaccard similarity) between each document and its neighbors
- Averages across all documents (0-1 scale, higher = better coherence)

**What the score means:**
- **≥50%** Good: Nearby docs consistently share keywords → projection is trustworthy
- **30-50%** Fair: Some nearby docs are unrelated → use caution when interpreting spatial proximity
- **<30%** Poor: Scattered topics in local neighborhoods → projection may be misleading

**Why it matters:**
Visual distance in the plot is supposed to represent semantic similarity. If embedding quality is poor:
- Documents that look close together might not actually be related
- Clusters might be artifacts of the dimensionality reduction rather than real topics
- Filtering/exploring "nearby" documents may give you false positives

**To improve:**
If quality is poor, try:
1. Adjusting preprocessing (lemmatization, stopwords)
2. Switching dimensionality reduction: UMAP instead of PCA/t-SNE
3. Adjusting TF-IDF parameters
4. Using different projection coordinates
            """)
    
    st.divider()
    
    # --- PERSISTENT SELECTION CONTROL BAR ---
    sel_count = len(st.session_state.get('selected_ids', []))

    if sel_count > 0:
        st.markdown(f"**Selection:** `{sel_count}` docs selected")
        bc1, bc2, bc3, bc4, bc5 = st.columns([1, 1, 1, 1.5, 1])
        with bc1:
            if st.button('Clear Selection', width="stretch", key='top_clear'):
                update_selection([], additive=False)
                st.rerun()
        with bc2:
            with st.popover("Save Group", width="stretch"):
                if 'saved_cohorts' not in st.session_state:
                    st.session_state['saved_cohorts'] = {}
                sg_name = st.text_input("Group Name", key='top_sg_name')
                if st.button("Confirm Save", key='top_sg_save'):
                    if sg_name:
                        st.session_state['saved_cohorts'][sg_name] = list(st.session_state['selected_ids'])
                        st.success("Saved!")
        with bc3:
            if st.button("Undo", width="stretch", key='top_undo', disabled=len(st.session_state.get('selection_history', [])) <= 0):
                history = st.session_state.get('selection_history', [])
                if history:
                    st.session_state['selected_ids'] = history.pop()
                    st.session_state['selection_history'] = history
                    st.rerun()
        with bc4:
            if st.session_state.get('selection_history'):
                with st.popover("Selection History", width="stretch"):
                    for i, past_sel in enumerate(reversed(st.session_state['selection_history'][-5:])):
                        if st.button(f"Step -{i+1}: {len(past_sel)} docs", key=f"top_hist_{i}"):
                            st.session_state['selected_ids'] = past_sel
                            st.rerun()
        with bc5:
            # Compute average similarity if applicable
            avg_sim_for_export = None
            selected_ids_export = st.session_state.get('selected_ids', [])
            if 2 <= len(selected_ids_export) <= 50 and len(embeddings_for_sim) > 0:
                try:
                    vecs = []
                    for did in selected_ids_export:
                        idx = id_to_local_idx.get(did)
                        if idx is not None and idx < len(embeddings_for_sim):
                            v = embeddings_for_sim[idx]
                            if hasattr(v, 'toarray'):
                                v = v.toarray().flatten()
                            elif hasattr(v, 'ndim') and v.ndim > 1:
                                v = v.flatten()
                            v = np.asarray(v, dtype=float)
                            vecs.append(v)
                    if len(vecs) >= 2:
                        sim_matrix = cosine_similarity(np.vstack(vecs))
                        n = len(vecs)
                        avg_sim_for_export = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
                except Exception:
                    pass
            
            findings_text = generate_findings_summary(avg_similarity=avg_sim_for_export)
            st.download_button(
                label="Export Findings",
                data=findings_text,
                file_name="findings.md",
                mime="text/markdown",
                width="stretch",
                key="export_findings_btn",
                help="Download a markdown summary of current session state"
            )
        st.markdown("---")

    # --- EMBEDDING TRUST VALIDATION ---
    # Show inline warnings if 2+ docs selected and there are embedding-trust issues
    if sel_count >= 2:
        selected_ids = st.session_state.get('selected_ids', [])
        trust_warnings = []
        
        # Check all pairs of selected documents
        for i in range(len(selected_ids)):
            for j in range(i + 1, min(i + 3, len(selected_ids))):  # Limit to 3 pairs to avoid clutter
                doc_id1, doc_id2 = selected_ids[i], selected_ids[j]
                is_suspicious, overlap, distance_pct, msg = check_embedding_trust_for_pair(
                    doc_id1, doc_id2, df, coords_tsne, 
                    keyword_overlap_threshold=0.15, 
                    distance_threshold_pct=0.25
                )
                if is_suspicious and msg:
                    trust_warnings.append(msg)
        
        # Display warnings if any
        if trust_warnings:
            with st.expander("⚠ Embedding Alerts", expanded=True):
                for warning in trust_warnings[:5]:  # Show max 5 warnings
                    st.caption(warning)

    # Color mapping and preparation for Saved Groups mode
    if color_mode == 'Saved Group':
        # Initialize cohort mapping with colorblind-safe colors
        df_work['__cohort'] = 'Other'
        cohort_map = st.session_state.get('saved_cohorts', {})
        
        if cohort_map:
            # Get colorblind-safe palette and handle >6 cohorts
            color_map, cohort_names, cohort_sizes = prepare_cohort_colors(cohort_map, df_work)
            
            # Apply cohort labels to df_work
            for cohort_name, ids in cohort_map.items():
                # Handle >6 cohorts: map to 'Other' if not in color_map
                if cohort_name in color_map:
                    df_work.loc[df_work['doc_id'].isin(ids), '__cohort'] = cohort_name
                else:
                    # This cohort is grouped into 'Other'
                    df_work.loc[df_work['doc_id'].isin(ids), '__cohort'] = 'Other'
        else:
            # No cohorts saved, default to 'Other'
            color_map = {'Other': '#CCCCCC'}
    
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
                width="stretch", 
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
        
        # Snapshot Comparison UI
        if st.session_state.get('compare_snapshots', False) and len(st.session_state['pipeline_snapshots']) >= 2:
            st.markdown('**Snapshot Diff Mode**')
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            snapshot_names = list(st.session_state['pipeline_snapshots'].keys())
            
            with comp_col1:
                snap1 = st.selectbox('Compare Snapshot 1', snapshot_names, key='snap_compare_1', label_visibility="collapsed")
            with comp_col2:
                st.caption('vs')
            with comp_col3:
                snap2 = st.selectbox('Compare Snapshot 2', snapshot_names, index=min(1, len(snapshot_names)-1), key='snap_compare_2', label_visibility="collapsed")
            
            if snap1 != snap2:
                st.info(f"Comparing **{snap1}** vs **{snap2}** — Projections show data from current active config")
        
        # Define projection metadata
        projections_meta = {
            'tsne': {'title': 't-SNE', 'subtitle': 'Preserves local clusters', 'x_col': 'tsne_x', 'y_col': 'tsne_y', 'download_name': 'tsne_plot.html'},
            'umap': {'title': 'UMAP', 'subtitle': 'Balances local and global structure', 'x_col': 'umap_x', 'y_col': 'umap_y', 'download_name': 'umap_plot.html'},
            'pca': {'title': 'PCA (preview)', 'subtitle': 'Preserves global variance', 'x_col': 'pca_x', 'y_col': 'pca_y', 'download_name': 'pca_plot.html'}
        }
        
        expanded_proj = st.session_state.get('expanded_projection')
        
        # Render projections
        if expanded_proj:
            # Full-width mode for expanded projection
            proj_key = expanded_proj
            proj_meta = projections_meta[proj_key]
            
            st.markdown(f"### {proj_meta['title']}", help="Click collapse button to return to 3-column view")
            st.caption(proj_meta['subtitle'])
            
            # Collapse button
            col_expand, col_spacer = st.columns([1, 10])
            with col_expand:
                if st.button('Collapse', key=f'collapse_{proj_key}', help='Return to 3-column layout'):
                    st.session_state['expanded_projection'] = None
                    st.rerun()
            
            # Create and display the figure
            if proj_key == 'tsne':
                fig = make_plot(
                    df_work, 'tsne_x', 'tsne_y',
                    st.session_state['selected_ids'] or [],
                    st.session_state.get('search_hits', []),
                    proj_meta['title'], hover_cols, color_mode,
                    focused_ids=st.session_state.get('focused_ids', []),
                    show_hover=show_hover
                )
                apply_scale_and_alpha(fig, point_alpha, point_size_scale)
                selection_tsne = st.plotly_chart(
                    fig, use_container_width=True, on_select="rerun",
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_tsne_{st.session_state['chart_revisions']['chart_tsne']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download t-SNE', data=fig.to_html(),
                        file_name=proj_meta['download_name'], mime='text/html', use_container_width=True
                    )
            elif proj_key == 'umap':
                fig = make_plot(
                    df_work, 'umap_x', 'umap_y',
                    st.session_state['selected_ids'] or [],
                    st.session_state.get('search_hits', []),
                    proj_meta['title'], hover_cols, color_mode,
                    focused_ids=st.session_state.get('focused_ids', []),
                    show_hover=show_hover
                )
                apply_scale_and_alpha(fig, point_alpha, point_size_scale)
                selection_umap = st.plotly_chart(
                    fig, use_container_width=True, on_select="rerun",
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_umap_{st.session_state['chart_revisions']['chart_umap']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download UMAP', data=fig.to_html(),
                        file_name=proj_meta['download_name'], mime='text/html', use_container_width=True
                    )
            else:  # pca
                fig = make_plot(
                    df_work, 'pca_x', 'pca_y',
                    st.session_state['selected_ids'] or [],
                    st.session_state.get('search_hits', []),
                    proj_meta['title'], hover_cols, color_mode,
                    focused_ids=st.session_state.get('focused_ids', []),
                    show_hover=show_hover
                )
                apply_scale_and_alpha(fig, point_alpha, point_size_scale)
                selection_pca = st.plotly_chart(
                    fig, use_container_width=True, on_select="rerun",
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_pca_{st.session_state['chart_revisions']['chart_pca']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download PCA', data=fig.to_html(),
                        file_name=proj_meta['download_name'], mime='text/html', use_container_width=True
                    )
        else:
            # Three-column mode
            plot_col1, plot_col2, plot_col3 = st.columns(3, gap="small")
            
            with plot_col1:
                st.markdown('**t-SNE**')
                st.caption('Preserves local clusters')
                if st.button('Expand', key='expand_tsne', help='Expand to full width'):
                    st.session_state['expanded_projection'] = 'tsne'
                    st.rerun()
                
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
                    fig_tsne, width="stretch", on_select="rerun", 
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_tsne_{st.session_state['chart_revisions']['chart_tsne']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download t-SNE', data=fig_tsne.to_html(),
                        file_name='tsne_plot.html', mime='text/html', width="stretch"
                    )
            
            with plot_col2:
                st.markdown('**UMAP**')
                st.caption('Balances local and global structure')
                if st.button('Expand', key='expand_umap', help='Expand to full width'):
                    st.session_state['expanded_projection'] = 'umap'
                    st.rerun()
                
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
                    fig_umap, width="stretch", on_select="rerun", 
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_umap_{st.session_state['chart_revisions']['chart_umap']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download UMAP', data=fig_umap.to_html(),
                        file_name='umap_plot.html', mime='text/html', width="stretch"
                    )
                    
            with plot_col3:
                st.markdown('**PCA (preview)**')
                st.caption('Preserves global variance')
                if st.button('Expand', key='expand_pca', help='Expand to full width'):
                    st.session_state['expanded_projection'] = 'pca'
                    st.rerun()
                
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
                    fig_pca, width="stretch", on_select="rerun", 
                    selection_mode=['points', 'box', 'lasso'],
                    key=f"chart_pca_{st.session_state['chart_revisions']['chart_pca']}"
                )
                if show_download_buttons:
                    st.download_button(
                        label='Download PCA', data=fig_pca.to_html(),
                        file_name='pca_plot.html', mime='text/html', width="stretch"
                    )

# â”€â”€ Process Graph Selections â”€â”€
current_graph_selections = set()

# Ensure selection variables are initialized (some may be in expanded mode)
selection_focus = locals().get('selection_focus', None)
selection_tsne = locals().get('selection_tsne', None)
selection_umap = locals().get('selection_umap', None)
selection_pca = locals().get('selection_pca', None)

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

# â”€â”€ Dynamic Grid Layout for Analysis Section â”€â”€
if view_mode == "Single (Focus)":
    st.markdown("---")

st.markdown("### Analysis")
# Give more width to keywords chart (1/5 for keywords, 2/5 for docs, 2/5 for distance)
tab_docs, tab_stats, tab_keywords = st.columns([2, 1.5, 1.5], gap="medium")

def render_selection_details(selected_ids):
    st.subheader('Selected Document Details')
    
    LARGE_SELECTION_THRESHOLD = 20  # Switch to large-selection mode at this count
    
    if not selected_ids:
        st.info('Select points in the graphs to see details here.')
        return

    # Filter dataframe for selected IDs
    selected_df = df_work[df_work['doc_id'].isin(selected_ids)]
    
    if selected_df.empty:
        st.warning('Selected IDs not found in current filtered dataset.')
        return

    st.write(f"**{len(selected_df)} document(s) selected**")
    
    # Large selection mode: paginate the results
    if len(selected_df) > LARGE_SELECTION_THRESHOLD:
        page_size = 10
        total_pages = (len(selected_df) + page_size - 1) // page_size
        
        if 'sel_page_num' not in st.session_state:
            st.session_state['sel_page_num'] = 0
        
        # Pagination controls
        pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
        with pcol1:
            if st.button("← Prev", disabled=st.session_state['sel_page_num'] == 0, key='sel_prev_page'):
                st.session_state['sel_page_num'] -= 1
                st.rerun()
        with pcol2:
            st.write(f"Page {st.session_state['sel_page_num'] + 1} / {total_pages}")
        with pcol3:
            if st.button("Next →", disabled=st.session_state['sel_page_num'] >= total_pages - 1, key='sel_next_page'):
                st.session_state['sel_page_num'] += 1
                st.rerun()
        
        # Show current page
        start_idx = st.session_state['sel_page_num'] * page_size
        end_idx = min(start_idx + page_size, len(selected_df))
        page_df = selected_df.iloc[start_idx:end_idx]
        
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(selected_df)}")
    else:
        page_df = selected_df
    
    # Interactive Table for Selected Docs
    # Prepare display dataframe
    display_df = page_df.copy()
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
        width="stretch",
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
        
        # Create display version: replace underscores with spaces
        display_id = doc_id.replace('_', ' ')
        # Truncate long display IDs with ellipsis
        if len(display_id) > 50:
            display_id_short = display_id[:50] + '…'
        else:
            display_id_short = display_id
        # Show with tooltip using HTML title attribute for full ID on hover
        safe_id = doc_id.replace("'", "\\'").replace('"', '\\"')
        st.markdown(f"#### <span title='{safe_id}' style='cursor: help;'>{display_id_short}</span>", unsafe_allow_html=True)
        
        # PRIMARY CONTENT: Document text/snippet
        st.write(row.get('__snippet', row.get('text', '')))
        
        # SECONDARY BLOCK: Metadata card
        meta_dict = {}
        for col in row.index:
            if col not in ['doc_id', 'text', 'cluster', '__snippet', 'Snippet'] and not col.startswith('__'):
                val = row[col]
                if not pd.isna(val) and str(val).strip():
                    meta_dict[col] = val
        
        if meta_dict:
            st.markdown("---")
            st.markdown("**Metadata**")
            
            # Create metadata card with clean formatting
            meta_cols = list(meta_dict.items())
            # Display in 2-column grid for readability
            for i in range(0, len(meta_cols), 2):
                col1, col2 = st.columns(2)
                
                # First column
                field_name, field_val = meta_cols[i]
                with col1:
                    # Format field name as label
                    st.caption(field_name)
                    # Display value with proper formatting
                    if isinstance(field_val, (int, float)):
                        st.write(f"**{field_val}**")
                    else:
                        st.write(str(field_val)[:200] + ('...' if len(str(field_val)) > 200 else ''))
                
                # Second column (if exists)
                if i + 1 < len(meta_cols):
                    field_name, field_val = meta_cols[i + 1]
                    with col2:
                        st.caption(field_name)
                        if isinstance(field_val, (int, float)):
                            st.write(f"**{field_val}**")
                        else:
                            st.write(str(field_val)[:200] + ('...' if len(str(field_val)) > 200 else ''))

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
                        st.dataframe(pd.DataFrame(disp_data), hide_index=True, width="stretch")
            except Exception:
                pass

    elif len(selected_doc_ids) == 2:
        # COMPARISON MODE
        doc_a, doc_b = selected_doc_ids[0], selected_doc_ids[1]
        st.markdown("---")
        st.subheader("Comparison View")
        
        # Metrics
        metrics = compute_pair_metrics(doc_a, doc_b)
        c1, c2, c3 = st.columns(3)
        c1.metric('Cosine Similarity', f"{metrics['cosine']:.3f}" if metrics else 'N/A', help="1.0 = Identical meaning")
        c2.metric('Euclidean Dist', f"{metrics['euclidean']:.3f}" if metrics else 'N/A', help="Lower is closer")

        # Projectual Distance (Diagnose projection vs original space)
        if metrics:
            row_a, row_b = df_work[df_work['doc_id'] == doc_a], df_work[df_work['doc_id'] == doc_b]
            if not row_a.empty and not row_b.empty:
                def p_dist(col_x, col_y):
                    return np.sqrt((row_a.iloc[0][col_x]-row_b.iloc[0][col_x])**2 + (row_a.iloc[0][col_y]-row_b.iloc[0][col_y])**2)
                p_dist_str = f"t-SNE: {p_dist('tsne_x', 'tsne_y'):.2f} | UMAP: {p_dist('umap_x', 'umap_y'):.2f} | PCA: {p_dist('pca_x', 'pca_y'):.2f}"
                c3.caption("Projection Dists:")
                c3.write(p_dist_str)
        
        # Side-by-Side Content
        colA, colB = st.columns(2)
        with colA:
            render_doc_card(doc_a)
        with colB:
            render_doc_card(doc_b)
            
        if 'row_a' in locals() and 'row_b' in locals() and not row_a.empty and not row_b.empty:
            toks_a = set(row_a.iloc[0].get('__tokens', []))
            toks_b = set(row_b.iloc[0].get('__tokens', []))
            shared = toks_a.intersection(toks_b)
            if shared:
                st.markdown(f"**Shared Terms**: `{'`, `'.join(sorted(shared)[:15])}`")
            else:
                st.caption("No shared terms found.")
            
        st.markdown("---")
        st.markdown("#### Documents Between A and B")
        st.caption("Finding the semantic gradient between these two documents...")
        vec_a_idx, vec_a = fetch_vector(doc_a)
        vec_b_idx, vec_b = fetch_vector(doc_b)
        if vec_a is not None and vec_b is not None:
             midpoint = (vec_a + vec_b) / 2.0
             num_neighbors = min(8, len(df_work))
             if num_neighbors > 0:
                 nn_model = NearestNeighbors(metric='cosine', n_neighbors=num_neighbors).fit(embeddings_for_sim)
                 dists, indices = nn_model.kneighbors(midpoint.reshape(1, -1))
                 
                 between_data = []
                 for dist, i in zip(dists[0], indices[0]):
                     if i < len(df_work):
                         nid = df_work.iloc[i]['doc_id']
                         if nid not in (doc_a, doc_b):
                             sim = 1.0 - dist
                             cluster = df_work[df_work['doc_id']==nid]['cluster'].iloc[0] if 'cluster' in df_work else 'N/A'
                             between_data.append({'Doc ID': nid, 'Similarity to Midpoint': f"{sim:.3f}", 'Cluster': str(cluster)})
                 
                 if between_data:
                     st.dataframe(pd.DataFrame(between_data), hide_index=True, width="stretch")
                 else:
                     st.write("No distinct documents found strictly between A and B.")
            
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
                    # Interpret similarity level
                    if avg_sim > 0.6:
                        interp = "High (similar topics)"
                    elif avg_sim > 0.3:
                        interp = "Moderate (mixed)"
                    else:
                        interp = "Low (diverse)"
                    st.metric("Group Similarity", f"{avg_sim:.3f} — {interp}", help="Average cosine similarity between all document pairs. 1.0 = identical, 0.0 = unrelated.")
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

# â”€â”€ Tab 1: Selected Documents â”€â”€
with tab_docs:
    with st.container(border=True):
        render_selection_details(st.session_state['selected_ids'])
    
# â”€â”€ Tab 2: Cluster Stats â”€â”€
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
            
            try:
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
                st.plotly_chart(fig_cluster, width="stretch")
            except Exception as e:
                st.error(f"Chart rendering error: {e}")
                st.dataframe(plot_df)
            
            if not st.session_state['selected_ids']:
                st.caption('Click to select all docs in a cluster:')
                cluster_cols = st.columns(min(5, len(cluster_stats)))
                for idx, (cluster_id, row) in enumerate(cluster_stats.head(5).iterrows()):
                    with cluster_cols[idx]:
                        if st.button(f'Cluster {cluster_id} ({int(row["Count"])})', key=f'select_cluster_{cluster_id}', width="stretch"):
                            cluster_docs = df_work[df_work['cluster'] == cluster_id]['doc_id'].tolist()
                            update_selection(cluster_docs, additive=False)
            st.markdown("---")
            # Distance Heatmap (only when 2+ selected)
            if len(st.session_state['selected_ids']) >= 2:
                with st.container(border=True):
                    st.subheader("Distance Analysis")
        
                    selected_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
                    n_selected = len(selected_df)
        
                    # Large selection mode: show centroid summary instead of full matrix
                    if n_selected > 20:
                        st.caption(f'Showing summary for {n_selected} selected documents (full matrix hidden for performance)')
            
                        selected_indices = [id_to_local_idx.get(doc_id) for doc_id in st.session_state['selected_ids'] if doc_id in id_to_local_idx]
                        selected_indices = [i for i in selected_indices if i is not None and i < len(embeddings_for_sim)]
            
                        if len(selected_indices) >= 2:
                            selected_embeddings = embeddings_for_sim[selected_indices]
                            centroid = selected_embeddings.mean(axis=0)
                
                            # Compute distance from each doc to centroid
                            distances_to_centroid = []
                            for idx in selected_indices:
                                vec = embeddings_for_sim[idx]
                                if hasattr(vec, 'toarray'):
                                    vec = vec.toarray().flatten()
                                elif hasattr(vec, 'ndim') and vec.ndim > 1:
                                    vec = vec.flatten()
                                dist = float(cosine_similarity(vec.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
                                distances_to_centroid.append(dist)
                
                            distances_to_centroid = np.array(distances_to_centroid)
                
                            # Show stats
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Avg Distance to Centroid", f"{distances_to_centroid.mean():.3f}")
                            col2.metric("Min", f"{distances_to_centroid.min():.3f}")
                            col3.metric("Max", f"{distances_to_centroid.max():.3f}")
                            col4.metric("Std Dev", f"{distances_to_centroid.std():.3f}")
                
                            st.caption("Interpretation: Higher average = selection is more spread out in embedding space")
        
                    # Show full heatmap only for reasonable sizes
                    elif n_selected <= 100:
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
                            st.plotly_chart(fig_heatmap, width="stretch")
                    else:
                        st.info(f'Distance matrix not shown for {n_selected} documents (too large). Use centroid summary above.')

    else:
        if 'cluster' not in target_df.columns:
            st.error("No cluster column found in dataframe. Check that cluster_labels.npy exists in the artifacts folder and that the pipeline has been run.")
        elif target_df['cluster'].nunique() == 0:
            st.error("Cluster column exists but has no values. The dataframe may be empty after filtering.")
        elif not GLOBAL_CLUSTER_MAP:
            st.warning("Cluster labels not loaded — GLOBAL_CLUSTER_MAP is empty. Cluster stats will show numeric IDs only.")
            st.dataframe(target_df.groupby('cluster')['doc_id'].count().reset_index())
        else:
            st.info("No cluster data available.")

# â”€â”€ Tab 3: Top Keywords â”€â”€
with tab_keywords:
    st.subheader("Top Keywords (TF-IDF Ranked)")
    
    # Explain TF-IDF scoring and why it matters
    with st.expander("Why TF-IDF? How does ranking work?", expanded=False):
        st.markdown("""
**TF-IDF (Term Frequency-Inverse Document Frequency)** ranks keywords by how specific and meaningful they are to YOUR selection:

**The Problem with Raw Frequency**: When you count word occurrences, common words dominate the results.
- Example: Word frequency ranking gives you "use", "make", "also", "data" — generic words that appear everywhere
- These don't tell you anything unique about YOUR selection

**The TF-IDF Solution**: Score each word by how often it appears in your selection vs. the entire corpus.
- **HIGH score** = Appears frequently in YOUR docs, but rarely in the corpus overall = **DISTINCTIVE to your selection**
- **LOW score** = Appears everywhere (in most documents) = generic filler words filtered out

**Practical Example**: Dashboard Design Corpus
- Raw frequency ranking: "use" (150 times), "make" (120), "also" (100), "data" (95)
- TF-IDF ranking: "dashboard" (high relevance), "design" (high relevance), "ui" (high relevance)

**Why This Matters**: TF-IDF reveals what actually distinguishes your group of documents from the rest of the corpus, helping you understand the unique themes and topics in your selection.
        """)
    
    # **CRITICAL CHECK**: Verify that token column is available and populated
    # If empty or missing, it means the offline pipeline was not run properly
    if '__tokens' not in df.columns or all(len(t) == 0 for t in df['__tokens']):
        st.error(
            "**OFFLINE PREPROCESSING REQUIRED**\n\n"
            "The 'tokens' column is missing or empty. This is created by the offline preprocessing "
            "pipeline (parse.py::preprocess_texts) which lemmatizes and filters tokens.\n\n"
            "**Keyword extraction requires preprocessed tokens for consistency between offline and online analysis.**\n\n"
            "**Actions:**\n"
            "1. Run the offline pipeline: `python run_pipeline.py`\n"
            "2. Verify the output CSV has a 'tokens' column\n"
            "3. Restart the Streamlit app\n\n"
            "We do NOT use regex fallback extraction because it produces different results than "
            "the offline pipeline (no lemmatization, different stopword filtering)."
        )
    
    if st.session_state['selected_ids']:
        sel_df = df_work[df_work['doc_id'].isin(st.session_state['selected_ids'])]
        all_toks = [tok for toks in sel_df['__tokens'] for tok in toks]
        if all_toks:
            # Use TF-IDF weighted keyword extraction with full corpus for IDF
            full_corpus_toks = df['__tokens'].tolist()  # Full corpus for IDF
            keywords_with_scores, fallback_mode = get_keywords_with_tfidf(
                selected_tokens=sel_df['__tokens'].tolist(),
                full_corpus_tokens=full_corpus_toks,
                n=15,
                min_doc_freq=2
            )
            
            # Show fallback note if in fallback mode
            if fallback_mode:
                st.warning("Limited Selection: Showing frequency-ranked keywords (need 2+ documents for accurate TF-IDF)")
            
            # Build dataframe from TF-IDF results
            if keywords_with_scores:
                kw_df = pd.DataFrame(keywords_with_scores, columns=['Term', 'Score'])
                kw_df['Score'] = kw_df['Score'].round(4)
                kw_df = kw_df.sort_values('Score', ascending=True)
            else:
                kw_df = pd.DataFrame(columns=['Term', 'Score'])
            
            # Show selection context
            st.caption(f"Analyzing {len(sel_df)} selected document(s) — Extracting top 15 distinct keywords")
            
            # Show TF-IDF visualization with labeled explanation
            st.markdown("**Ranked by TF-IDF Score** — Higher = more specific to your selection, lower = more generic")
            fig_kw = px.bar(
                kw_df,
                text='Score',
                y='Term',
                x='Score',
                orientation='h',
                hover_data={'Score': ':.4f'},
                labels={'Score': 'TF-IDF Score'}
            )
            fig_kw.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                yaxis=dict(type='category', dtick=1),
                yaxis_title='',
                xaxis_title='TF-IDF Score (Higher = More Distinctive)',
                bargap=0.3
            )
            fig_kw.update_xaxes(title_text="TF-IDF Score (Higher = More Distinctive to Your Selection)")
            fig_kw.update_traces(marker_color='steelblue', textposition='auto')
            st.plotly_chart(fig_kw, width="stretch")
            
            # Show table for reference with explanation
            with st.expander("Keyword Details Table — See exact TF-IDF scores"):
                # Display table first (primary content)
                display_df = kw_df.sort_values('Score', ascending=False).copy()
                display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.6f}")
                st.caption("Term | Score (higher = more distinctive)")
                st.dataframe(display_df, width="stretch", hide_index=True)
                
                # Filter explanations behind an info box below the table
                st.markdown("---")
                with st.expander("ℹ️ How does filtering work?", expanded=False):
                    st.markdown("""
**Column Explanation:**
- **Term**: The keyword or phrase extracted from your selection
- **Score**: TF-IDF score (0.0 = generic filler, higher values = unique to your selection)

**Quality Filters Applied:**
- Minimum 2 documents must contain term (eliminates one-off artifacts)
- Excluded 100+ generic filler words ("use", "make", "also", "data", "design", "good", etc.)
- Only 3+ character terms (removes abbreviations and single letters)
- No pure numbers
                    """)
        else:
            st.info("No tokens available. Try clearing cache (â‹® â†’ Clear cache).")
    else:
        st.info("Select documents to see their top keywords (ranked by TF-IDF specificity).")

# ===== SESSION HISTORY LOG PANEL =====
st.markdown("---")
with st.expander("Session Reasoning Trail", expanded=False):
    """
    **Your session history** — All significant actions timestamped. Click any entry to restore that state.
    """
    
    history_log = st.session_state.get('history_log', [])
    
    if history_log:
        # Create columns for display and actions
        col1_exp, col2_exp = st.columns([4, 1])
        
        with col2_exp:
            if st.button("Export as TXT", key="export_history"):
                # Format history for export
                export_lines = ['=== Session Reasoning Trail ===\n']
                export_lines.append(f'Session generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                export_lines.append(f'Total actions: {len(history_log)}\n\n')
                
                for i, entry in enumerate(history_log, 1):
                    export_lines.append(f"{i}. [{entry['timestamp']}] {entry['action_type'].upper()}")
                    export_lines.append(f"   {entry['description']}\n")
                
                export_text = '\n'.join(export_lines)
                st.download_button(
                    label="Download TXT",
                    data=export_text,
                    file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_history"
                )
        
        # Display history entries
        for i, entry in enumerate(reversed(history_log)):
            cols = st.columns([0.5, 3, 1.5])
            with cols[0]:
                st.caption(f"#{len(history_log) - i}")
            with cols[1]:
                st.write(f"**{entry['timestamp']}** — *{entry['action_type']}*")
                st.caption(entry['description'])
            with cols[2]:
                if st.button("Restore", key=f"restore_{i}", help="Restore session state to this point"):
                    restore_state_from_entry(entry)
                    st.toast(f"Restored state from {entry['timestamp']}")
    else:
        st.caption("No actions logged yet. Start searching, selecting, or saving groups to build your reasoning trail.")

st.markdown('---')
st.caption('Embeddings Explorer v2.0')



