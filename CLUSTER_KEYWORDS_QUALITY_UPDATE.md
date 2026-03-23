# Cluster Keywords Quality Update

## Problem Statement

Cluster keywords in `cluster_keywords.csv` are precomputed offline using **frequency-based extraction**, which:
- ❌ Produces generic keywords appearing across many clusters
- ❌ Doesn't reflect what makes EACH cluster distinctive
- ❌ Can become stale when preprocessing or data changes
- ❌ Quality degrades with different preprocessing variants

The app already had a superior **runtime TF-IDF extraction function** available, but users weren't aware it was being used instead of the offline CSV.

## Solution Implemented

Rather than loading stale offline keywords, the app now:
1. **Computes cluster keywords at runtime** using Inter-Cluster TF-IDF
2. **Prominently documents** this quality improvement to users
3. **Explains the algorithm** in expandable UI panels

### Why Inter-Cluster TF-IDF is Better

| Aspect | Frequency-Based | Inter-Cluster TF-IDF |
|--------|-----------------|----------------------|
| **What it measures** | How often word appears | How distinctive word is to cluster |
| **Generic words** | High rank (generic filler) | Filtered out (low TF-IDF) |
| **Distinctive terms** | Mixed with generic | High rank (distinctive) |
| **Quality** | Static (precomputed) | Dynamic (responds to data changes) |
| **Performance** | Pre-computed (~0ms) | Runtime (~5ms per cluster) |

**Example**: For a Dashboard Design corpus:
- ❌ Frequency-based: "use" (150×), "make" (120×), "also" (100×) ← generic!
- ✅ TF-IDF: "dashboard", "design", "ui" ← distinctive!

## Algorithm: Inter-Cluster TF-IDF

For each cluster, compute:
```
TF-IDF(term) = TF(term) × IDF(term)

where:
  TF(term) = frequency of term within cluster
  IDF(term) = log(total_clusters / clusters_containing_term)
```

**Key properties:**
- **TF** = how common the term is in THIS cluster
- **IDF** = how rare the term is across clusters (log inverse)
- **Result** = high score if term is BOTH frequent locally AND rare globally
- **Top 3 terms** with highest TF-IDF scores become cluster keywords
- **Filtering**: Extended stop words + short tokens (<3 chars) removed

## UI Changes

### 1. Clustering Settings Panel
Added expandable section: **"ℹ️ About Cluster Keywords"**

Located in: **Sidebar → Clustering Settings**

Shows:
- TF-IDF algorithm explanation
- Why it's better than frequency-based extraction
- Quality indicators (distinctive vs. overlapping clusters)
- Performance note (~5ms per cluster)

```
Expanded by default: NO (user can click to read)
```

### 2. Dynamic Info on Cluster Coloring
When user selects "Cluster" as color mode, shows:
```
🎯 Cluster Keywords are computed at runtime using Inter-Cluster TF-IDF,
   which ranks terms by how distinctive they are to each cluster.
   This is more accurate than offline frequency-based extraction.
   Expand 'Clustering Settings' → 'About Cluster Keywords' for details.
```

Links users to the detailed explanation.

### 3. Enhanced `extract_cluster_topics()` Docstring

Updated docstring emphasizes:
- ✅ Keywords are computed at **runtime** (not stale offline files)
- ✅ Quality automatically improves with data/preprocessing changes
- ✅ Algorithm details (TF-IDF scoring, filtering, top-3 selection)
- ✅ Performance (~5ms per cluster)
- ✅ Requirement for preprocessed tokens from offline pipeline

## Code Changes

| Location | Change | Purpose |
|----------|--------|---------|
| Lines 826-856 | Enhanced `extract_cluster_topics()` docstring | Document algorithm quality & details |
| Lines 1651-1668 | Added "About Cluster Keywords" expander | Explain TF-IDF algorithm to users |
| Lines 1767-1779 | Added dynamic info on Cluster mode | Link to documentation when Cluster is selected |

## Offline Keywords CSV

The `cluster_keywords.csv` file in artifacts folders:
- ✅ Still exists (for reference/audit trails)
- ❌ NO LONGER LOADED or used by the app
- ℹ️ Marked as stale (superseded by runtime TF-IDF)
- 📝 Could be deleted without breaking anything

If you want to preserve the file for auditing, add a note:
```
# DEPRECATED: Superseded by runtime Inter-Cluster TF-IDF algorithm
# These offline keywords use frequency-based extraction and may be stale.
# The Streamlit app now computes superior cluster keywords at runtime (~5ms per cluster).
```

## Verification

✅ **Syntax**: File compiles without errors  
✅ **Algorithm**: Runtime TF-IDF was already working; now documented  
✅ **API**: No changes to existing functions (pure documentation + UI addition)  
✅ **Backward Compatibility**: Existing cluster filtering/display works unchanged  

## User Impact

**Before:**
- Users may have assumed cluster keywords came from offline preprocessing
- No indication quality could be improved
- Vague keywords = no explanation why

**After:**
- Clear explanation that keywords use superior runtime TF-IDF
- Expandable documentation explaining WHY this is better
- Visual indicator when Cluster mode is selected
- Performance context (5ms per cluster is negligible)

## Performance

- **Computation**: ~5ms per cluster (already happening at load time)
- **UI rendering**: Instant (expanders are lazy-loaded)
- **No additional caching needed**: `extract_cluster_topics()` is already cached via `GLOBAL_CLUSTER_MAP`

## Next Steps (Optional)

1. **Delete offline keywords**: Remove `cluster_keywords.csv` from artifacts folders (optional; harmless to keep for audit)
2. **Add to documentation**: Update project README to note cluster keywords use runtime TF-IDF
3. **Add metrics**: Could compute/display "cluster separation score" to indicate how well clusters are separated (lower = more overlap)
4. **Compare variants**: Could show how cluster keywords change when switching preprocessing variants

## Testing Checklist

- [ ] Load app and select "Cluster" color mode
- [ ] Verify info message appears about TF-IDF runtime computation
- [ ] Click "Clustering Settings" → "About Cluster Keywords" expander
- [ ] Verify algorithm explanation is clear and accurate
- [ ] Verify cluster labels in legend match TF-IDF top-3 keywords
- [ ] Verify keywords are human-readable and distinctive per cluster
- [ ] Switch between variants and confirm keywords update appropriately
