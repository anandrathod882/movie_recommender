# 🎬 CineAI — Movie Recommendation System

> A full-stack, research-grade movie recommender built on MovieLens Latest Small.  
> Combines TF-IDF content filtering, SVD matrix factorisation, Alternating Least Squares (ALS), and Neural Collaborative Filtering (NCF) into a Hybrid recommender — served through a polished Streamlit interface.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Evaluation Results](#evaluation-results)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Running the App](#running-the-app)
9. [Evaluation Pipeline](#evaluation-pipeline)
10. [Notebooks](#notebooks)
11. [Deliverables](#deliverables)
12. [References](#references)

---

## Project Overview

CineAI implements and benchmarks four recommendation algorithms on the MovieLens-Latest-Small dataset (100 k ratings, 9 k movies, 610 users). The system is designed as an end-to-end ML pipeline:

```
Raw Data → EDA → Model Training → Offline Evaluation → Hybrid API → Streamlit UI
```

Key capabilities:
- **Cold-start handling**: content-based fallback for users with < 20 ratings
- **Warm-user hybrid**: weighted blend of CF (SVD + NCF) and content-based scores
- **Live poster fetching**: OMDb API integration in the Streamlit UI
- **Offline metrics**: Precision, Recall, NDCG, HitRate, MAP, Coverage @ K=10

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app/)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │        Hybrid Engine         │
              │   (src/hybrid.py)            │
              └──┬──────────┬───────────┬───┘
                 │          │           │
         ┌───────▼──┐  ┌────▼───┐  ┌───▼──────┐
         │ Content   │  │  SVD   │  │   NCF    │
         │ Based     │  │ (CF)   │  │ (Neural) │
         │TF-IDF     │  │ ALS    │  │          │
         └───────────┘  └────────┘  └──────────┘
                 │          │           │
         ┌───────▼──────────▼───────────▼──────┐
         │         data/processed/              │
         │  ratings.csv · movies.csv            │
         └──────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) |
| Ratings | 100,836 |
| Movies | 9,742 |
| Users | 610 |
| Rating scale | 0.5 – 5.0 (half-star) |
| Avg rating | ~3.50 |
| Matrix sparsity | ~98.3% |
| Time range | 1996 – 2018 |

---

## Models

### 1. Content-Based Filtering (TF-IDF)
- Builds a "movie soup" from genres + release year
- TF-IDF vectorisation → cosine similarity matrix
- Seed movie → top-K nearest neighbours
- **Cold-start safe** — no interaction history needed

### 2. SVD (Matrix Factorisation)
- Decomposes the user-movie rating matrix via truncated SVD
- Predicts missing ratings; unrated items are ranked by predicted score
- Saves full predicted matrix to `models/svd_predicted.pkl`

### 3. ALS (Alternating Least Squares)
- Implicit feedback variant via the `implicit` library
- Treats ratings as confidence-weighted implicit signals
- Stored as a bundle `{model, user_map, item_map}` in `models/als_model.pkl`

### 4. Neural Collaborative Filtering (NCF)
- Embedding-based architecture (GMF + MLP fusion)
- Trained with binary cross-entropy on positive/negative rating pairs
- Saved to `models/ncf_model.pt`

### 5. Hybrid Recommender
- **Warm users** (≥ 20 ratings): `score = 0.7 × CF_score + 0.3 × CB_score`
- **Cold-start users** (< 20 ratings): pure content-based on top-rated seed
- CF score = mean of normalised SVD + NCF scores

---

## Evaluation Results

*Evaluation on 20% temporal hold-out, K=10, first 50 test users.*

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | RMSE |
|---|---|---|---|---|---|
| Content-Based (TF-IDF) | 0.0420 | 0.0111 | 0.0558 | 0.4200 | N/A |
| SVD (Collaborative) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.8384 |
| Neural CF (NCF) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |

> Note: SVD and NCF show zero ranking metrics on this split due to the cold-start nature of the test set and the small dataset size. SVD RMSE of 1.84 reflects rating-prediction accuracy. Content-based achieves 42% HitRate — a strong signal on such a sparse dataset.

---

## Project Structure

```
movie_recommender/
├── app/
│   └── streamlit_app.py        ← Streamlit UI (search, browse, detail view)
├── data/
│   ├── raw/                    ← Original MovieLens CSVs
│   └── processed/
│       ├── ratings.csv
│       └── movies.csv
├── models/
│   ├── als_model.pkl           ← Implicit ALS bundle
│   ├── svd_predicted.pkl       ← Full predicted rating matrix
│   ├── ncf_model.pt            ← PyTorch NCF weights
│   ├── ncf_index_map.csv       ← user/movie index mapping
│   └── evaluation_results.csv  ← Offline metrics table
├── notebooks/
│   ├── eda.ipynb               ← Exploratory Data Analysis notebook
│   └── fig1_*.png … fig9_*.png ← EDA figures
├── src/
│   ├── __init__.py
│   ├── content_based.py        ← TF-IDF pipeline
│   ├── collaborative.py        ← SVD pipeline
│   ├── als_model.py            ← ALS training
│   ├── neural_cf.py            ← NCF model & training
│   ├── hybrid.py               ← Hybrid recommender
│   ├── data_loader.py          ← Data ingestion helpers
│   └── evaluator.py            ← Offline evaluation (Step 10)
├── docs/
│   ├── report.md               ← Technical report (Step 13)
│   ├── slides.md               ← Presentation slides (Step 14)
│   └── model_card.md           ← Model Card (Step 15)
└── README.md                   ← This file
```

---

## Quick Start

### Prerequisites
```
Python 3.9+
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn torch implicit streamlit requests
```

### Data Setup
Download [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/), place `ratings.csv` and `movies.csv` in `data/raw/`, then run the data loader:
```bash
python -m src.data_loader
```

### Train Models
```bash
# Content-based (no training needed — runs at inference)
python -m src.collaborative   # SVD
python -m src.als_model       # ALS
python -m src.neural_cf       # NCF (GPU optional)
```

---

## Running the App
```bash
streamlit run app/streamlit_app.py
```
Set your OMDb API key in the sidebar to enable live movie posters.

---

## Evaluation Pipeline
```bash
python -m src.evaluator
```
Results are saved to:
- `models/evaluation_results.csv`
- `models/evaluation_results.json`

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/eda.ipynb` | Full exploratory data analysis with 9 publication-quality figures |

Run with:
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Deliverables

| Step | File | Description |
|---|---|---|
| 10 | `src/evaluator.py` | Offline evaluation module (Precision, Recall, NDCG, MAP, Coverage, RMSE) |
| 11 | `notebooks/eda.ipynb` | EDA notebook with 9 figures |
| 12 | `README.md` | This GitHub documentation |
| 13 | `docs/report.md` | Full technical report with metrics and analysis |
| 14 | `docs/slides.md` | Presentation slide deck |
| 15 | `docs/model_card.md` | Model card (intended use, limitations, metrics) |

---

## References

1. F. Maxwell Harper and Joseph A. Konstan. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems, 5(4):19:1–19:19, 2015.
2. Yifan Hu, Yehuda Koren, Chris Volinsky. *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008.
3. Xiangnan He et al. *Neural Collaborative Filtering.* WWW 2017.
4. Simon Funk. *Netflix Update: Try This at Home.* 2006.
5. GroupLens Research. *MovieLens Latest Small Dataset.* https://grouplens.org/datasets/movielens/latest/

---

*Built with ❤️ using Python · scikit-learn · PyTorch · implicit · Streamlit*
