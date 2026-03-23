# Offline/Online Keyword Extraction Consistency Fix

## Summary
Fixed critical inconsistency between offline and online keyword extraction. The online Streamlit app **no longer falls back to regex extraction** when the token column is missing. This ensures all keyword results are consistent with the offline pipeline's lemmatized tokens.

---

## The Problem ❌

**Before this fix:**

| Stage | Token Source | Processing |
|-------|--------------|-----------|
| **Offline** | CSV 'tokens' column | Lemmatized input by parse.py::preprocess_texts |
| **Online** | Regex fallback (if tokens missing) | Raw regex extraction - NO lemmatization |
| **Result** | Different keywords depending on which analysis ran | ❌ Inconsistent |

**Example mismatch:**
- Offline: "running", "ran", "runs" → lemmatized to "run" → **1 keyword**
- Online (regex): "running", "ran", "runs" → kept as-is → **3 keywords**
- TF-IDF scores would differ significantly

---

## The Solution ✅

**After this fix:**

| Stage | Token Source | Processing |
|-------|--------------|-----------|
| **Offline** | CSV 'tokens' column created by parse.py | Lemmatized, stopword-filtered tokens |
| **Online - required** | CSV 'tokens' column (same source) | Uses exact same preprocessed tokens |
| **Online - missing** | Shows error → forces user to run pipeline | Prevents silent inconsistency |
| **Result** | All keyword extraction uses identical tokens | ✅ Consistent |

---

## What Changed

### 1. **Removed Regex Fallback** (~33 lines deleted)
- **Location**: `streamlit_app/streamlit_app.py` lines 680-714 (before fix)
- **What was removed**: 
  ```python
  # REMOVED: Fallback when token column missing
  words = re.findall(r'[a-zA-Z]{3,}', text.lower())  # ❌ Raw extraction
  ```
- **Reason**: This created raw tokens with no lemmatization

### 2. **Added Token Column Validation**
- **Location**: `load_and_process_data()` function (lines 645-685)
- **New behavior**: 
  - If 'tokens' column missing/empty → show **critical warning**
  - Never silently extract tokens via regex
  - Force user to run offline pipeline

### 3. **Updated Documentation**
Files updated with "CRITICAL" notices:
- `get_keywords_with_tfidf()` docstring: Explains tokens must be preprocessed
- `extract_cluster_topics()` docstring: Notes dependency on offline preprocessing
- Top Keywords tab UI: Shows error if tokens unavailable

### 4. **Added User-Facing Warnings**
Three levels of warning:
1. **Console warning** when app loads and tokens are missing
2. **Top Keywords tab error** explaining what happened and how to fix
3. **Docstring notes** in functions explaining the requirement

---

## User Impact

### If CSV has 'tokens' column (expected case) ✅
- **No change**: Everything works as before
- Keyword extraction is now guaranteed correct and consistent

### If CSV missing 'tokens' column ❌
**Before fix:** Got results silently (but wrong - from regex)
**After fix:** Gets clear error message explaining:
- Why tokens are required
- What the 'tokens' column contains
- How to get the column (run offline pipeline)

---

## How to Fix Missing Tokens

If user sees error "CRITICAL: Token column missing or empty!":

```bash
# 1. Run offline pipeline
python run_pipeline.py

# 2. Verify output CSV has 'tokens' column
# Look in artifacts/processed_data.csv or artifacts/processed_data_with_clusters.csv

# 3. Restart Streamlit app
streamlit run streamlit_app/streamlit_app.py
```

---

## Technical Details

### Token Column Properties
The 'tokens' column created by offline pipeline contains:
- ✅ Lowercased text
- ✅ Lemmatized tokens (e.g., "running" → "run")
- ✅ Stopwords removed (e.g., "the", "is", "with")
- ✅ Short tokens filtered (only 3+ characters)
- ✅ Numbers removed

### Functions Updated
1. `parse_keyword_space()` - Now validates tokens present
2. `get_keywords_with_tfidf()` - Docstring clarifies preprocessing requirement
3. `extract_cluster_topics()` - Added preprocessing requirement notice
4. Top Keywords Streamlit UI - Added validation warning

### Consistency Guarantee
All three keyword extraction points in Streamlit now use identical pipeline:
- **Top Keywords tab** ← same 'tokens' column
- **Orient Me cluster labels** ← same 'tokens' column
- **Cluster Topic extraction** ← same 'tokens' column

---

## Files Modified
- **streamlit_app/streamlit_app.py** (4 locations, net removal of ~33 lines)
  - Lines 645-685: Regex fallback removal + validation
  - Lines 168-191: `get_keywords_with_tfidf()` docstring
  - Lines 803-813: `extract_cluster_topics()` docstring
  - Lines ~3458-3475: UI error message

---

## Testing
- ✅ Python syntax validation passed
- ✅ Warning messages display correctly
- ✅ Docstrings updated
- ✅ Error handling prevents silent inconsistency

---

## Recommendation
After deploying, verify by:
1. Opening Streamlit app
2. Checking Top Keywords tab for any warning messages
3. If warning appears, run: `python run_pipeline.py`
4. Verify CSV contains 'tokens' column: `head -1 artifacts/processed_data.csv | grep -i token`
