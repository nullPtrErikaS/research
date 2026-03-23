# Embedding Quality Score - Visual Guide

## What Users Will See

### Scenario 1: GOOD Embedding Quality (≥50%)

```
┌───────────────────────────────────────────────────────────────────┐
│ ✅ Embeddings are healthy (78% — nearby docs are semantically     │
│    similar)                                                        │
│                                                                    │
│ ▼ What is embedding quality? Why does it matter?                  │
└───────────────────────────────────────────────────────────────────┘
────────────────────────────────────────────────────────────────────

**Selection:** 0 docs selected
```

**Location**: Top of page, green success box
**Appearance**: Green banner with checkmark
**Implies**: User can trust the spatial relationships in the plot

---

### Scenario 2: FAIR Embedding Quality (30-50%)

```
┌───────────────────────────────────────────────────────────────────┐
│ ⚠️ Moderate embedding quality (42%) — Some nearby docs may not     │
│    be semantically similar. Results should be interpreted with     │
│    caution.                                                        │
│                                                                    │
│ ▼ What is embedding quality? Why does it matter?                  │
└───────────────────────────────────────────────────────────────────┘
────────────────────────────────────────────────────────────────────

**Selection:** 0 docs selected
```

**Location**: Top of page, orange warning box
**Appearance**: Orange banner with warning triangle
**Implies**: Proximity in the plot is approximate; verify manually before trusting

---

### Scenario 3: POOR Embedding Quality (<30%)

```
┌───────────────────────────────────────────────────────────────────┐
│ 🚨 POOR EMBEDDING QUALITY (18%) — Nearby documents are             │
│    semantically distant. This projection may be misleading.        │
│    Consider regenerating with different settings (TF-IDF, UMAP,    │
│    etc.)                                                           │
│                                                                    │
│ ▼ What is embedding quality? Why does it matter?                  │
└───────────────────────────────────────────────────────────────────┘
────────────────────────────────────────────────────────────────────

**Selection:** 0 docs selected
```

**Location**: Top of page, red error box
**Appearance**: Red banner with alert emoji
**Implies**: Be very cautious; spatial proximity may not reflect actual similarity

---

## Expandable Details Section

When user clicks "What is embedding quality? Why does it matter?", they see:

```
**Embedding Quality Score: 42.3%**

Measures semantic coherence of the projection by checking if nearby 
documents in 2D space have similar keywords.

**How it works:**
- Finds 5 nearest neighbors for each document in the 2D projection
- Computes keyword overlap (Jaccard similarity) between each 
  document and its neighbors
- Averages across all documents (0-1 scale, higher = better coherence)

**What the score means:**
- **≥50%** ✅ Good: Nearby docs consistently share keywords → projection 
  is trustworthy
- **30-50%** ⚠️ Fair: Some nearby docs are unrelated → use caution when 
  interpreting spatial proximity
- **<30%** 🔴 Poor: Scattered topics in local neighborhoods → projection 
  may be misleading

**Why it matters:**
Visual distance in the plot is supposed to represent semantic similarity. 
If embedding quality is poor:
- Documents that look close together might not actually be related
- Clusters might be artifacts of the dimensionality reduction rather than 
  real topics
- Filtering/exploring "nearby" documents may give you false positives

**To improve:**
If quality is poor, try:
1. Adjusting preprocessing (lemmatization, stopwords)
2. Switching dimensionality reduction: UMAP instead of PCA/t-SNE
3. Adjusting TF-IDF parameters
4. Using different projection coordinates
```

---

## Sidebar Changes

### Before
```
┌─────────────────────┐
│ Orient Me           │
├─────────────────────┤
│ Embedding Health    │
│ 🟢 78%              │
├─────────────────────┤
```

### After
```
┌─────────────────────────────────┐
│ Orient Me                       │
├─────────────────────────────────┤
│ Embedding Quality         ℹ️     │
│ ✅ GOOD 78%               [?]   │
├─────────────────────────────────┤
```

**Sidebar improvements**:
- Clearer label: "Embedding Quality" vs "Embedding Health"
- Status badge: ✅ GOOD / ⚠️ FAIR / 🚨 POOR
- Percentage shown directly
- Info popover (ℹ️) for quick reference

---

## User Decision Tree

```
User opens app
    ↓
[Sees embedding quality banner at TOP of page]
    ↓
Is quality GOOD (≥50%)?
├─ YES → ✅ Explore with confidence; spatial proximity is reliable
├─ MAYBE (30-50%) → ⚠️ Browse cautiously; verify keyword overlap manually
└─ NO (<30%) → 🚨 Consider regenerating with different settings
                before trusting spatial relationships
    ↓
User clicks expander for details
    ↓
Learns:
- How score is computed (5NN Jaccard similarity)
- Why it matters (visual distance ≠ semantic similarity if poor)
- How to improve (preprocessing, UMAP, TF-IDF tuning)
```

---

## Key Design Decisions

### 1. **Why at the TOP of the page?**
- Users see it immediately upon loading
- No scrolling required → ensures they're aware
- Prominent alerts (red/orange/green) draw attention
- Can expand details without leaving main view

### 2. **Why color-coded alerts?**
- **Green** (good): Reassuring, signals confidence
- **Orange** (fair): Cautions, signals uncertainty
- **Red** (poor): Alerts, signals action needed
- Uses standard UI patterns users expect

### 3. **Why the expander pattern?**
- Defaults to collapsed: doesn't add visual clutter
- Available immediately: users can learn without searching
- Optional engagement: only interested users expand

### 4. **Why these thresholds?**
- ≥50%: "Good" indicates ≥50% of neighbors share keywords
- 30-50% "Fair": Some coherence, but mixed
- <30%: "Poor": Most neighbors are semantically different
- Aligned with user requirement: "warn if below 0.5"

### 5. **Why keep sidebar metric?**
- Provides consistent reference point in Orient Me section
- Users who want quick status can check sidebar
- Doesn't duplicate banner; sidebar shows status, banner explains
- Familiar location for power users

---

## Testing Checklist

- [ ] Load app with different datasets
- [ ] Verify banner appears at TOP of main page
- [ ] Check color changes at thresholds (50%, 30%)
- [ ] Expand details section; verify readability
- [ ] Check sidebar metric shows status badges correctly
- [ ] Test with embedding_health_score = None (should show "N/A")
- [ ] Verify metrics persist when filtering/selecting docs
- [ ] Check that threshold messages are clear and actionable
