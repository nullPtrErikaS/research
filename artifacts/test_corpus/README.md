# Test Corpus — Expected Behaviors

Use this small corpus (20 docs) to verify the pipeline and explorer work correctly.

## Document Groups

| ID(s) | Group | What to Check |
|---|---|---|
| `DUPE_A1`, `DUPE_A2`, `DUPE_A3` | **Exact duplicates** | Must have cosine similarity = 1.000. Must overlap perfectly on all projections. |
| `PARA_A1`, `PARA_A2` | **Paraphrases** | Should have high similarity (>0.5) and sit near the DUPE cluster but not overlap exactly. |
| `SPORT_01`–`SPORT_05` | **Sports cluster** | Should form a tight cluster, clearly separated from Cooking. |
| `COOK_01`–`COOK_05` | **Cooking cluster** | Should form a tight cluster, clearly separated from Sports. |
| `OUTLIER_01` | **Quantum physics** | Should be far from everything else — an isolated point. |
| `OUTLIER_02` | **Crypto/blockchain** | Should also be isolated, and dissimilar to OUTLIER_01. |
| `MIXED_01`–`MIXED_03` | **Cross-domain** | Intentionally bridge Sports and Cooking. Should appear between the two clusters. |

## Verification Checklist

### Pipeline (`run_pipeline.py`)
- [ ] Pipeline runs without errors
- [ ] `artifacts/test_corpus/processed_data_with_clusters.csv` is generated
- [ ] `artifacts/test_corpus/coords_tsne.npy` shape is (20, 2)
- [ ] `artifacts/test_corpus/coords_umap.npy` shape is (20, 2)
- [ ] `artifacts/test_corpus/coords.npy` shape is (20, 2)

### Explorer Projections
- [ ] DUPE_A1/A2/A3 overlap on all 3 projections
- [ ] Sports docs group together
- [ ] Cooking docs group together
- [ ] OUTLIER_01 and OUTLIER_02 are far from main clusters

### Selection & Focus
- [ ] Lasso around Sports cluster → only SPORT_ IDs selected
- [ ] Click DUPE_A1 within the group → green diamond, group preserved
- [ ] Select DUPE_A1 + DUPE_A2 from table → distance matrix shows 1.000

### Nearest Neighbors
- [ ] Selecting DUPE_A1 → nearest neighbors are DUPE_A2 and DUPE_A3 (sim ≈ 1.0)
- [ ] Selecting OUTLIER_01 → nearest neighbor is NOT OUTLIER_02 (different topics)

### Top Keywords
- [ ] Sports selection → keywords include: team, ball, player, match, etc.
- [ ] Cooking selection → keywords include: sauce, chicken, oven, batter, etc.

### Clustering
- [ ] With k=4: Sports, Cooking, Duplicates/Paraphrases, and Outliers/Mixed should separate
- [ ] With k=2: Sports+Mixed vs Cooking+Dupes should roughly split
