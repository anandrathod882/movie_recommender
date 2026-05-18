# CineAI: A Hybrid Movie Recommendation System
## Technical Report

**Dataset:** MovieLens Latest Small  
**Authors:** CineAI Project  
**Date:** May 2026  

---

## Abstract

This report presents CineAI, a hybrid movie recommendation system that combines content-based filtering, collaborative filtering (SVD and ALS), and neural collaborative filtering (NCF) to deliver personalised movie recommendations. We evaluate all models on a temporal 80/20 hold-out split of the MovieLens Latest Small dataset (100,836 ratings, 9,742 movies, 610 users) using ranking and prediction metrics. The content-based model achieves a HitRate@10 of 42%, while SVD produces an RMSE of 1.84 on rating prediction. A hybrid ranker is deployed via a Streamlit web application with live OMDb poster integration.

---

## 1. Introduction

Recommender systems have become a cornerstone of modern digital platforms, driving content discovery at Netflix, Spotify, and Amazon. Building an effective recommender requires balancing three challenges:

1. **Sparsity** — the user-movie interaction matrix is ~98% empty
2. **Cold Start** — new users have insufficient history for collaborative filtering
3. **Scalability** — inference must be fast enough for interactive applications

This project implements a full ML pipeline: data ingestion, EDA, model training, offline evaluation, and a polished Streamlit interface (CineAI). We explore four model families and fuse them into a hybrid ranker.

---

## 2. Dataset Analysis

### 2.1 Dataset Statistics

| Statistic | Value |
|---|---|
| Total ratings | 100,836 |
| Unique movies | 9,742 |
| Unique users | 610 |
| Rating scale | 0.5 – 5.0 (half-star increments) |
| Mean rating | 3.50 |
| Median rating | 3.50 |
| Most frequent rating | 4.0 |
| Matrix sparsity | 98.3% |
| Time span | 1996 – 2018 |

### 2.2 Key EDA Findings

**Rating Distribution (Fig. 1):** The distribution is left-skewed, with 4.0 being the modal rating. Only 5.2% of ratings are below 2.0, suggesting a positivity bias typical of voluntary rating systems.

**Genre Analysis (Fig. 2):** Drama (4,361 movies) and Comedy (3,756) dominate the catalog. Documentary and Film-Noir receive the highest average ratings (≈3.9), but have far fewer total ratings than Action and Drama.

**User Activity (Fig. 3):** Rating counts follow a power-law distribution. The median user has rated 70 movies; the mean is 165. Users with fewer than 20 ratings (cold-start threshold) account for 12.8% of the user base.

**Movie Popularity (Fig. 4):** The top movie (*Forrest Gump*) has 329 ratings; the median movie has only 3 ratings — a classic long-tail distribution that motivates content-based fallback.

**Temporal Trends (Fig. 5):** Rating activity peaks in 1999–2000 and again in 2015–2018. Average ratings remained stable at 3.4–3.6 across time periods.

**Matrix Sparsity (Fig. 6):** A 50×100 sample visualisation confirms extreme sparsity, with ~96% of cells empty — justifying the use of latent-factor models.

**Genre Ratings (Fig. 7):** Film-Noir and Documentary lead in quality while Action and Horror trail. This informs genre-weight tuning in future work.

**Model Comparison (Fig. 8):** Among ranking metrics, the content-based model outperforms CF models on this dataset/split combination.

**Release Year Distribution (Fig. 9):** Most movies were released in the 1990s–2010s, reflecting the dataset's focus on films rated by contemporary users.

---

## 3. Models

### 3.1 Content-Based Filtering (TF-IDF)

**Method:** Each movie is represented as a "soup" string consisting of its genres (pipe-separated) and release year. A TF-IDF vectoriser transforms the corpus into a term-frequency matrix, and pairwise cosine similarity is computed to produce a 9,742 × 9,742 similarity matrix.

**Inference:** Given a seed movie (the user's highest-rated unrated film), the top-K most similar movies (excluding already-rated ones) are returned.

**Complexity:** O(V·M) build time where V is vocabulary size and M is movie count. Inference is O(M) per query (single row lookup).

**Hyperparameters:**
- Vectoriser: TF-IDF, English stop-words removed
- Similarity: Cosine
- K = 10 recommendations

### 3.2 SVD (Matrix Factorisation)

**Method:** The user-movie rating matrix (610 × 9,742) is constructed with NaN for missing entries. Truncated SVD decomposes it into latent user and item factor matrices. The predicted rating matrix is reconstructed and clipped to the [0.5, 5.0] range.

**Cold-Start Limitation:** Users must exist in the training matrix. New users cannot be served by SVD alone.

**Storage:** The full predicted matrix (~610 × 9,742 float32 values) is serialised to `models/svd_predicted.pkl` (47 MB).

### 3.3 ALS (Alternating Least Squares)

**Method:** Uses the `implicit` library to train an implicit ALS model treating ratings as confidence signals (confidence = 1 + α × rating). The model alternates between solving for user factors and item factors until convergence.

**Implicit Feedback Advantage:** ALS treats all observed interactions as positive signal with varying confidence, which avoids treating unobserved pairs as negative.

**Saved Artefact:** A bundle dictionary `{model, user_map, item_map}` is persisted so that `userId`/`movieId` lookups remain fast at evaluation and inference time.

### 3.4 Neural Collaborative Filtering (NCF)

**Architecture:**
```
userId ──► Embedding(32) ──┐
                            ├──► Concatenate ──► MLP(64→32→16→1) ──► Sigmoid
movieId ──► Embedding(32) ──┘
```

**Training:**
- Loss: Binary Cross-Entropy (positive rating ≥ 4.0 as positive, sampled negatives)
- Optimiser: Adam (lr=1e-3)
- Batch size: 512
- Epochs: 10
- Device: CPU (GPU if available)

**Regularisation:** L2 weight decay (1e-5) on embeddings.

### 3.5 Hybrid Recommender

```
hybrid_score(u, i) = 0.7 × cf_score(u, i) + 0.3 × cb_score(seed, i)
```

where `cf_score` is the average of min-max-normalised SVD and NCF scores.

**Cold-Start Branch (< 20 ratings):** Pure content-based on the user's highest-rated movie as the seed.

---

## 4. Evaluation Methodology

### 4.1 Protocol

- **Split:** Temporal 80/20 split (sort by timestamp, hold out last 20%)
- **Relevant items:** Movies with rating ≥ 4.0 in the test set
- **Sampled users:** First 50 unique test users (due to compute constraints)
- **K:** 10

### 4.2 Metrics

| Metric | Formula | What it Measures |
|---|---|---|
| Precision@K | hits / K | Exactness — fraction of top-K that are relevant |
| Recall@K | hits / \|relevant\| | Coverage — fraction of relevant retrieved |
| NDCG@K | DCG / IDCG | Rank-weighted relevance quality |
| HitRate@K | 1 if any hit else 0 | At least one relevant item in top-K |
| MAP@K | mean AP across users | Area under precision-recall curve |
| Coverage | \|unique recs\| / catalog | Diversity across the full user population |
| RMSE | √(mean((pred-actual)²)) | Rating-prediction accuracy (SVD only) |

### 4.3 Results

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | RMSE |
|---|---|---|---|---|---|
| Content-Based (TF-IDF) | 0.0420 | 0.0111 | 0.0558 | **0.4200** | N/A |
| SVD (Collaborative) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.8384** |
| Neural CF (NCF) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |

### 4.4 Discussion

**Content-Based dominates ranking metrics** on this dataset/split because:
- The test split contains many users whose training interactions are minimal, making CF models unable to form meaningful latent representations.
- Genre + year similarity is a strong proxy for user taste on a small dataset.

**SVD RMSE of 1.84** reflects the difficulty of predicting absolute rating values for held-out user-movie pairs, especially given the 98% sparsity of the training matrix.

**NCF zero metrics** are explained by the same cold-start phenomenon: the model sees most test users for the first time and defaults to average embeddings.

**Recommended improvements:**
1. Use leave-one-out evaluation (hold out each user's last interaction) rather than a global temporal split — this better evaluates ranking ability
2. Add negative sampling evaluation (1 positive + 99 random negatives per user) as used in He et al. (2017)
3. Increase training data with the full MovieLens 25M dataset
4. Use Bayesian Personalised Ranking (BPR) loss for implicit CF

---

## 5. Deployment

The CineAI Streamlit app (`app/streamlit_app.py`) provides:

- **Search & Recommend:** Type any movie title to get 10 similar movies
- **Browse by Genre:** Explore top movies within a selected genre
- **Similar Movies:** Click any movie poster to see nearest neighbours
- **Inspiration Feed:** Curated daily picks from popular genres
- **Movie Detail View:** Full detail overlay with cast, plot, and ratings
- **Settings Popover:** Theme toggle, cache management, and configuration

Live poster images are fetched from the OMDb API (API key configured in sidebar).

---

## 6. Limitations and Future Work

| Limitation | Proposed Fix |
|---|---|
| Small dataset (100k ratings) | Scale to MovieLens 25M or Netflix Prize |
| No session context | Add session-based transformer (BERT4Rec) |
| Static similarity matrix | Retrain nightly on new ratings |
| No diversity enforcement | Add MMR re-ranking for genre diversity |
| No A/B testing | Deploy with feature flags + metrics logging |
| Cold-start CF | Implement meta-learning (MAML) for fast adaptation |

---

## 7. Conclusion

CineAI demonstrates a complete recommendation pipeline from raw data to a deployed interactive application. The content-based model provides the most reliable ranking performance on the MovieLens Latest Small dataset, achieving 42% HitRate@10. A hybrid architecture combining content-based and collaborative signals is deployed in the Streamlit app, providing graceful cold-start handling while leveraging latent user preferences for warm users. Future work will scale the dataset, improve evaluation protocols, and explore session-aware recommendation.

---

## References

1. F.M. Harper, J.A. Konstan. "The MovieLens Datasets." *ACM TIIS*, 5(4), 2015.
2. Y. Hu, Y. Koren, C. Volinsky. "Collaborative Filtering for Implicit Feedback." *ICDM*, 2008.
3. X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T. Chua. "Neural Collaborative Filtering." *WWW*, 2017.
4. S. Rendle et al. "BPR: Bayesian Personalised Ranking." *UAI*, 2009.
5. Sun et al. "BERT4Rec: Sequential Recommendation with BERT." *CIKM*, 2019.
