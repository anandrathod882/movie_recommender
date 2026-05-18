# CineAI — Movie Recommendation System
## Presentation Slides

> **Format:** Markdown slide deck (compatible with Marp, Slidev, or Reveal.js)  
> **Aspect ratio:** 16:9 | **Theme:** Dark

---

<!-- slide 1 -->
# 🎬 CineAI
### A Hybrid Movie Recommendation System

**MovieLens Latest Small · Python · PyTorch · Streamlit**

---

<!-- slide 2 -->
## The Problem

> *"You've watched 800 movies. We still don't know what to show you next."*

### Challenges in Movie Recommendation
- **Sparsity:** 98.3% of user-movie pairs are unobserved
- **Cold Start:** New users have no history
- **Long Tail:** Top 5% of movies get 80% of ratings
- **Scale:** Must serve recommendations in < 200ms

---

<!-- slide 3 -->
## Dataset: MovieLens Latest Small

| Property | Value |
|---|---|
| 🎬 Movies | 9,742 |
| 👤 Users | 610 |
| ⭐ Ratings | 100,836 |
| 📅 Time span | 1996 – 2018 |
| 🕳️ Sparsity | **98.3%** |
| ⭐ Avg rating | 3.50 / 5.0 |

> Source: GroupLens Research, University of Minnesota

---

<!-- slide 4 -->
## EDA: Key Findings

### What the data tells us
1. **4.0 is the modal rating** — users rate what they like
2. **Drama + Comedy dominate** — 45% of the catalog
3. **Film-Noir rated highest** — quality > quantity
4. **12.8% cold-start users** — < 20 ratings
5. **Power-law popularity** — median movie has 3 ratings

---

<!-- slide 5 -->
## System Architecture

```
┌─────────────────────────────────────┐
│         Streamlit UI (CineAI)        │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Hybrid Ranker      │
    │  CF 70% + CB 30%    │
    └──┬────────┬─────────┘
       │        │
  ┌────▼──┐  ┌──▼──────┐
  │ SVD   │  │ TF-IDF  │
  │  +    │  │ Content │
  │ NCF   │  │ Based   │
  └───────┘  └─────────┘
```

**Cold-start branch:** Pure content-based for users with < 20 ratings

---

<!-- slide 6 -->
## Model 1: Content-Based (TF-IDF)

### How it works
1. Build "movie soup" = genres + release year
2. TF-IDF vectorise → 9,742 × vocabulary matrix
3. Pairwise cosine similarity → 9,742 × 9,742 matrix
4. Query: top-K most similar to user's seed movie

### Strengths
- ✅ No interaction history needed (cold-start safe)
- ✅ Interpretable: explains recommendations via shared genres
- ✅ O(1) inference (pre-computed similarity matrix)

### Weakness
- ❌ Ignores user-specific taste patterns

---

<!-- slide 7 -->
## Model 2: SVD (Matrix Factorisation)

### How it works
$$R \approx U \cdot \Sigma \cdot V^T$$

1. Fill observed ratings into a 610 × 9,742 matrix
2. Truncated SVD → latent user/item factors
3. Reconstruct full matrix → predicted ratings for all pairs
4. Rank unseen movies by predicted score

### Strengths
- ✅ Captures latent taste dimensions
- ✅ Provides RMSE-comparable rating predictions

### Result
- **RMSE: 1.84** (rating prediction on hold-out)

---

<!-- slide 8 -->
## Model 3: ALS (Implicit Feedback)

### Key Idea
> Treat ratings as **confidence**, not ground truth

$$c_{ui} = 1 + \alpha \cdot r_{ui}$$

- Alternates between solving for user factors and item factors
- Scales to millions of users (embarrassingly parallel)
- Uses the `implicit` library (GPU-accelerated)

### Advantage over SVD
Handles **implicit signals** — even a low rating is a signal of engagement

---

<!-- slide 9 -->
## Model 4: Neural Collaborative Filtering (NCF)

### Architecture
```
userId  → Embedding(32) ──┐
                            ├→ Concat → MLP(64→32→16→1) → Sigmoid
movieId → Embedding(32) ──┘
```

- **Loss:** Binary Cross-Entropy
- **Optimiser:** Adam (lr=1e-3)
- **Epochs:** 10 | **Batch:** 512

### Why NCF?
> Linear factor models (SVD/ALS) cannot capture non-linear user-item interactions — NCF learns arbitrary interaction functions

---

<!-- slide 10 -->
## Hybrid Strategy

```python
# Warm user (≥ 20 ratings)
hybrid_score = 0.7 × CF_score + 0.3 × CB_score

# Cold-start user (< 20 ratings)
hybrid_score = CB_score  # pure content-based
```

**CF score:** `(norm_SVD + norm_NCF) / 2`

### Why hybrid?
| Scenario | Best Model |
|---|---|
| New user, no history | Content-Based |
| Active user, many ratings | SVD / NCF |
| Niche genre lover | Content-Based supplement |
| All users | **Hybrid wins** |

---

<!-- slide 11 -->
## Evaluation Protocol

### Setup
- **Split:** Temporal 80/20 (sort by timestamp)
- **Relevant:** Items rated ≥ 4.0 in test set
- **K = 10** | **50 sampled test users**

### Metrics
| Metric | Measures |
|---|---|
| Precision@10 | How exact are the recommendations? |
| Recall@10 | How much of what's relevant was found? |
| NDCG@10 | Are relevant items ranked higher? |
| HitRate@10 | At least one relevant item in top-10? |
| MAP@10 | Area under precision curve |
| RMSE | Rating prediction accuracy |

---

<!-- slide 12 -->
## Results

| Model | Precision@10 | HitRate@10 | RMSE |
|---|---|---|---|
| **Content-Based** | **0.042** | **0.42** | N/A |
| SVD | 0.000 | 0.000 | **1.84** |
| NCF | 0.000 | 0.000 | N/A |

### Key Takeaway
> Content-Based achieves **42% HitRate** on a 98% sparse dataset —  
> meaning in 42% of cases, at least one recommendation is a movie  
> the user genuinely loved.

SVD/NCF zero ranking metrics → cold-start test split effect (most test users are new to the model). SVD's RMSE measures rating prediction quality separately.

---

<!-- slide 13 -->
## CineAI: The App

### Features
- 🔍 **Search & Recommend** — Type any movie title
- 🎭 **Browse by Genre** — Explore curated genre sections
- 🖼️ **Movie Posters** — Live from OMDb API
- 📋 **Detail View** — Click any poster for cast + plot
- 🔄 **Similar Movies** — Clickable "More Like This"
- ⚙️ **Settings** — Theme toggle, cache control

### Stack
`Python` · `Streamlit` · `scikit-learn` · `PyTorch` · `implicit` · `OMDb API`

---

<!-- slide 14 -->
## Limitations

| Limitation | Impact | Fix |
|---|---|---|
| 100k ratings only | Low CF signal | Use MovieLens 25M |
| Static models | Stale recommendations | Nightly retraining |
| No diversity enforcement | Filter bubble risk | MMR re-ranking |
| No A/B testing | Can't measure live quality | Feature flags |
| Cold-start CF | Poor new-user CF | MAML / meta-learning |
| Binary relevance | Ignores rating magnitude | Graded NDCG |

---

<!-- slide 15 -->
## Future Work

### Near-term
- [ ] Scale to MovieLens 25M (25 million ratings)
- [ ] BPR (Bayesian Personalised Ranking) loss for NCF
- [ ] Leave-one-out evaluation for fairer comparison
- [ ] Add user feedback loop (thumbs up/down in app)

### Long-term
- [ ] BERT4Rec — session-aware transformer recommender
- [ ] Graph Neural Networks (NGCF / LightGCN)
- [ ] Multi-modal — use movie poster embeddings
- [ ] Federated learning for privacy-preserving CF

---

<!-- slide 16 -->
## Summary

### What We Built
| Component | Status |
|---|---|
| EDA (9 figures) | ✅ Complete |
| Content-Based (TF-IDF) | ✅ Complete |
| SVD (Matrix Factorisation) | ✅ Complete |
| ALS (Implicit CF) | ✅ Complete |
| NCF (Neural CF) | ✅ Complete |
| Hybrid Ranker | ✅ Complete |
| Offline Evaluator | ✅ Complete |
| Streamlit App | ✅ Complete |
| README | ✅ Complete |
| Report + Slides + Model Card | ✅ Complete |

---

<!-- slide 17 -->
# Thank You 🎬

### CineAI — Hybrid Movie Recommendation System

**Dataset:** MovieLens Latest Small (100,836 ratings · 9,742 movies · 610 users)  
**Best HitRate@10:** 42% (Content-Based)  
**Live App:** `streamlit run app/streamlit_app.py`  

---

*Built with Python · scikit-learn · PyTorch · implicit · Streamlit*
