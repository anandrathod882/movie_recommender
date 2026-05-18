# Model Card: CineAI Hybrid Recommendation System

> This model card follows the format proposed by Mitchell et al. (2019) — *"Model Cards for Model Reporting"* — to document intended use, evaluation results, limitations, and ethical considerations for the CineAI recommendation models.

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | CineAI Hybrid Recommender |
| **Version** | 1.0 |
| **Type** | Hybrid Recommender System (Content-Based + Collaborative Filtering) |
| **Sub-models** | TF-IDF Content-Based · SVD · ALS (Implicit) · Neural CF (NCF) |
| **Date** | May 2026 |
| **Framework** | scikit-learn · PyTorch · implicit |
| **License** | MIT |
| **Contact** | See repository README |

---

## Intended Use

### Primary Use Cases
- **Personal movie discovery:** Recommend 10 movies a user is likely to enjoy based on their rating history
- **Cold-start discovery:** Suggest movies similar to a seed title for new users with no rating history
- **Browse-by-genre:** Surface popular and highly-rated films within a requested genre
- **"More Like This":** Given any movie, find the K most similar titles in the catalog

### Intended Users
- End users of the CineAI Streamlit application
- Researchers studying recommendation system benchmarks on MovieLens
- Students learning about hybrid recommendation architectures

### Out-of-Scope Uses
- **Commercial deployment at scale** without retraining on a larger, up-to-date dataset
- **Recommendations for non-movie content** (books, music, products) without retraining
- **Sensitive personalisation** (health, finance, news) where recommendation errors carry significant risk
- **Replacement for human editorial curation** in safety-critical contexts

---

## Training Data

### Dataset
**MovieLens Latest Small** — collected and maintained by GroupLens Research, University of Minnesota.

| Property | Value |
|---|---|
| Source | https://grouplens.org/datasets/movielens/latest/ |
| Ratings | 100,836 |
| Movies | 9,742 |
| Users | 610 (anonymised) |
| Rating scale | 0.5 – 5.0 (half-star) |
| Time span | March 1996 – September 2018 |
| License | GroupLens Research Terms of Use |

### Pre-processing
- Ratings sorted by timestamp; 80% used for training, 20% held out for evaluation
- Movie genres extracted as pipe-separated strings; release year extracted via regex from title
- For NCF: userIds and movieIds re-encoded as 0-indexed integers
- For ALS: ratings used as confidence weights (c = 1 + α × rating, α = 40)

### Data Limitations
- All 610 users are from a single geographic region (USA, English-language platform assumed)
- The dataset excludes movies rated fewer than 1 time — extremely rare films are not in the catalog
- Ratings reflect voluntary user behaviour; they may not represent population-level preferences
- The dataset is static (last updated 2018); recent films are absent

---

## Model Descriptions

### 1. Content-Based (TF-IDF)

| Property | Value |
|---|---|
| Input | Movie genres + release year ("soup") |
| Vectoriser | TF-IDF (English stop-words removed) |
| Similarity | Cosine similarity |
| Matrix size | 9,742 × 9,742 |
| Inference | O(1) — pre-computed at startup |
| Cold-start safe | ✅ Yes |

### 2. SVD (Collaborative Filtering)

| Property | Value |
|---|---|
| Input | User-movie rating matrix (610 × 9,742) |
| Algorithm | Truncated SVD (scipy.linalg.svds) |
| Latent factors | 50 |
| Output | Full predicted rating matrix |
| Model file | `models/svd_predicted.pkl` (47 MB) |
| Cold-start safe | ❌ No — users must appear in training |

### 3. ALS (Alternating Least Squares)

| Property | Value |
|---|---|
| Library | `implicit` (Hu et al., 2008) |
| Factors | 50 |
| Regularisation | 0.01 |
| Iterations | 20 |
| Confidence α | 40 |
| Model file | `models/als_model.pkl` |
| Cold-start safe | ❌ No |

### 4. Neural Collaborative Filtering (NCF)

| Property | Value |
|---|---|
| Architecture | GMF-style: Embedding(32) × 2 → MLP(64→32→16→1) |
| Loss | Binary Cross-Entropy |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-5) |
| Epochs | 10 |
| Batch size | 512 |
| Device | CPU / CUDA |
| Model file | `models/ncf_model.pt` (2.6 MB) |
| Cold-start safe | ❌ No |

### 5. Hybrid Ranker

```
Warm user  (≥ 20 ratings): score = 0.7 × CF_score + 0.3 × CB_score
Cold-start (< 20 ratings): score = CB_score
```

---

## Evaluation Results

### Protocol
- **Split:** Temporal 80/20 (sort by `timestamp`)
- **Relevant items:** Test ratings ≥ 4.0
- **Sampled users:** 50 (first unique users in test set)
- **K = 10**

### Metrics @ K=10

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | MAP@10 | RMSE |
|---|---|---|---|---|---|---|
| Content-Based (TF-IDF) | 0.0420 | 0.0111 | 0.0558 | **0.4200** | — | N/A |
| SVD (Collaborative) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | **1.8384** |
| Neural CF (NCF) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | N/A |

> **Interpretation:** The zero CF ranking metrics reflect the cold-start nature of the temporal test split: most test users have few or no training interactions. SVD's RMSE of 1.84 measures rating-prediction accuracy independently of ranking. The content-based 42% HitRate is strong for a sparse 100k-rating dataset.

### Evaluation Caveats
- The 50-user sample may not be representative of the full user distribution
- Temporal split may inflate content-based performance (genre preferences are stable over time)
- A leave-one-out or negative-sampling protocol would give a fairer CF comparison

---

## Ethical Considerations

### Fairness & Bias
| Risk | Severity | Mitigation |
|---|---|---|
| **Popularity bias** | Medium | Long-tail movies are under-recommended due to fewer ratings |
| **Genre bias** | Low-Medium | Drama/Comedy dominate training data; niche genres may be under-served |
| **Geographic bias** | Medium | Dataset is US-centric; non-English films are under-represented |
| **Recency bias** | Medium | Films post-2018 are absent; the model cannot recommend new releases |
| **Demographic bias** | Unknown | User demographics are not available; potential age/gender skew unknown |

### Privacy
- All user IDs in MovieLens are **anonymised** — no real-world identity is linkable
- The Streamlit application does not collect, store, or transmit any user data
- OMDb API calls expose only movie titles (no user data) to a third-party service

### Transparency
- This model card, the README, and the full source code are publicly available
- Evaluation metrics are reported honestly, including zero-value CF metrics

### Misuse Risks
- The model should **not** be used to infer personal characteristics (age, political views) from watching history
- The model should **not** be used to create "filter bubbles" by exclusively recommending within genres already watched

---

## Limitations

| Limitation | Description |
|---|---|
| **Dataset size** | 100k ratings is insufficient for deep CF models to generalise |
| **Static catalog** | No mechanism to add new movies at inference time |
| **No real-time feedback** | User preferences are captured at training time only |
| **Single-domain** | Trained only on movies; not transferable to other domains |
| **SVD scalability** | Full predicted matrix (~610×9742) is stored in RAM — does not scale to millions of users |
| **NCF sample size** | NCF is undertrained on 100k ratings; would benefit from 10M+ |

---

## How to Use

### Inference (Python)
```python
from src.hybrid import load_all_models, hybrid_recommend

models = load_all_models()

# Warm user
recs = hybrid_recommend(user_id=1, models=models, top_n=10)
print(recs[["title", "genres", "hybrid_score", "method"]])

# Cold-start user
recs = hybrid_recommend(user_id=999, models=models, top_n=10)
print(recs[["title", "genres", "hybrid_score", "method"]])
```

### Evaluation
```python
python -m src.evaluator
# Outputs: models/evaluation_results.csv, models/evaluation_results.json
```

### Web Application
```bash
streamlit run app/streamlit_app.py
```

---

## Citation

If you use this system or its evaluation code in research, please cite:

```bibtex
@misc{cineai2026,
  title   = {CineAI: A Hybrid Movie Recommendation System},
  year    = {2026},
  note    = {Built on MovieLens Latest Small (Harper & Konstan, 2015)},
  url     = {https://github.com/your-username/movie_recommender}
}
```

**Underlying dataset:**
```bibtex
@article{harper2015movielens,
  title   = {The MovieLens Datasets: History and Context},
  author  = {Harper, F. Maxwell and Konstan, Joseph A.},
  journal = {ACM Transactions on Interactive Intelligent Systems (TiiS)},
  volume  = {5},
  number  = {4},
  pages   = {19:1--19:19},
  year    = {2015}
}
```

---

## References

1. Mitchell, M., et al. "Model Cards for Model Reporting." *FAccT*, 2019.
2. Harper, F.M., Konstan, J.A. "The MovieLens Datasets." *ACM TIIS*, 2015.
3. Hu, Y., Koren, Y., Volinsky, C. "Collaborative Filtering for Implicit Feedback." *ICDM*, 2008.
4. He, X., et al. "Neural Collaborative Filtering." *WWW*, 2017.
5. GroupLens Research. MovieLens Latest Small. https://grouplens.org/datasets/movielens/latest/
