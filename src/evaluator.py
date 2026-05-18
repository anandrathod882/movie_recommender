"""
evaluator.py — Offline Evaluation Module
Metrics: Precision@K, Recall@K, NDCG@K, HitRate@K, MAP@K, Coverage, RMSE
Models:  Content-Based (TF-IDF), SVD, NCF, ALS
Usage:   python -m src.evaluator
"""
import pandas as pd
import numpy as np
import os, json


# ── 1. Data Loading ──────────────────────────────────────────────────────────
def load_data(processed_path="data/processed"):
    ratings = pd.read_csv(os.path.join(processed_path, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(processed_path, "movies.csv"))
    return ratings, movies


# ── 2. Train/Test Split (temporal) ───────────────────────────────────────────
def train_test_split(ratings, test_ratio=0.2):
    ratings = ratings.sort_values("timestamp")
    split   = int(len(ratings) * (1 - test_ratio))
    train   = ratings.iloc[:split]
    test    = ratings.iloc[split:]
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


# ── 3. Point-wise Metric ─────────────────────────────────────────────────────
def compute_rmse(predicted_df, test_df):
    errors = []
    for _, row in test_df.iterrows():
        uid, mid = row["userId"], row["movieId"]
        if uid in predicted_df.index and mid in predicted_df.columns:
            errors.append((predicted_df.loc[uid, mid] - row["rating"]) ** 2)
    return round(float(np.sqrt(np.mean(errors))), 4) if errors else None


# ── 4. Ranking Metrics ───────────────────────────────────────────────────────
def precision_at_k(recommended, relevant, k=10):
    hits = len(set(recommended[:k]) & set(relevant))
    return round(hits / k, 4) if k > 0 else 0.0

def recall_at_k(recommended, relevant, k=10):
    hits = len(set(recommended[:k]) & set(relevant))
    return round(hits / len(relevant), 4) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k=10):
    dcg  = sum(1/np.log2(i+2) for i, x in enumerate(recommended[:k]) if x in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0

def hit_rate_at_k(recommended, relevant, k=10):
    return 1.0 if set(recommended[:k]) & set(relevant) else 0.0

def ap_at_k(recommended, relevant, k=10):
    hits, score = 0, 0.0
    for i, x in enumerate(recommended[:k], 1):
        if x in relevant:
            hits += 1
            score += hits / i
    return round(score / min(len(relevant), k), 4) if relevant else 0.0

def catalog_coverage(all_rec_ids, total_movies):
    return round(len(set(all_rec_ids)) / total_movies, 4) if total_movies > 0 else 0.0


# ── 5. Internal helper ───────────────────────────────────────────────────────
def _agg(precisions, recalls, ndcgs, hits, aps, all_recs, total_movies, k):
    avg = lambda lst: round(float(np.mean(lst)), 4) if lst else 0.0
    return {
        f"Precision@{k}": avg(precisions),
        f"Recall@{k}"   : avg(recalls),
        f"NDCG@{k}"     : avg(ndcgs),
        f"HitRate@{k}"  : avg(hits),
        f"MAP@{k}"      : avg(aps),
        "Coverage"      : catalog_coverage(all_recs, total_movies),
    }


# ── 6a. Content-Based ────────────────────────────────────────────────────────
def evaluate_content_based(movies_cb, cosine_sim, test_df, k=10, n_users=50):
    from src.content_based import get_recommendations
    precisions, recalls, ndcgs, hits, aps, all_recs = [], [], [], [], [], []
    for uid in test_df["userId"].unique()[:n_users]:
        ut = test_df[test_df["userId"] == uid]
        relevant = ut[ut["rating"] >= 4.0]["movieId"].tolist()
        if not relevant:
            continue
        seed_row = movies_cb[movies_cb["movieId"] == ut["movieId"].iloc[0]]
        if len(seed_row) == 0:
            continue
        recs = get_recommendations(seed_row.iloc[0]["title"], movies_cb, cosine_sim, top_n=k)
        if isinstance(recs, list) or len(recs) == 0:
            continue
        rec_ids = recs["movieId"].tolist()
        all_recs.extend(rec_ids)
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        hits.append(hit_rate_at_k(rec_ids, relevant, k))
        aps.append(ap_at_k(rec_ids, relevant, k))
    return {"Model": "Content-Based (TF-IDF)",
            **_agg(precisions, recalls, ndcgs, hits, aps, all_recs, len(movies_cb), k),
            "RMSE": "N/A"}


# ── 6b. SVD ─────────────────────────────────────────────────────────────────
def evaluate_svd(predicted_df, user_movie_matrix, test_df, movies, k=10, n_users=50):
    from src.collaborative import get_svd_recommendations
    precisions, recalls, ndcgs, hits, aps, all_recs = [], [], [], [], [], []
    for uid in test_df["userId"].unique()[:n_users]:
        if uid not in predicted_df.index:
            continue
        ut = test_df[test_df["userId"] == uid]
        relevant = ut[ut["rating"] >= 4.0]["movieId"].tolist()
        if not relevant:
            continue
        recs = get_svd_recommendations(uid, user_movie_matrix, predicted_df, movies, top_n=k)
        if len(recs) == 0:
            continue
        rec_ids = recs["movieId"].tolist()
        all_recs.extend(rec_ids)
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        hits.append(hit_rate_at_k(rec_ids, relevant, k))
        aps.append(ap_at_k(rec_ids, relevant, k))
    rmse = compute_rmse(predicted_df, test_df)
    return {"Model": "SVD (Collaborative)",
            **_agg(precisions, recalls, ndcgs, hits, aps, all_recs, len(movies), k),
            "RMSE": rmse if rmse else "N/A"}


# ── 6c. NCF ─────────────────────────────────────────────────────────────────
def evaluate_ncf(model, ratings_df, movies, device, test_df, k=10, n_users=50):
    from src.neural_cf import get_ncf_recommendations
    precisions, recalls, ndcgs, hits, aps, all_recs = [], [], [], [], [], []
    for uid in test_df["userId"].unique()[:n_users]:
        ut = test_df[test_df["userId"] == uid]
        relevant = ut[ut["rating"] >= 4.0]["movieId"].tolist()
        if not relevant:
            continue
        recs = get_ncf_recommendations(uid, model, ratings_df, movies, device, top_n=k)
        if isinstance(recs, list) or len(recs) == 0:
            continue
        rec_ids = recs["movieId"].tolist()
        all_recs.extend(rec_ids)
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        hits.append(hit_rate_at_k(rec_ids, relevant, k))
        aps.append(ap_at_k(rec_ids, relevant, k))
    return {"Model": "Neural CF (NCF)",
            **_agg(precisions, recalls, ndcgs, hits, aps, all_recs, len(movies), k),
            "RMSE": "N/A"}


# ── 6d. ALS ─────────────────────────────────────────────────────────────────
def evaluate_als(als_path, ratings, movies, test_df, k=10, n_users=50):
    """
    Evaluate implicit ALS model. The pickle may contain:
      - raw model (implicit.als.AlternatingLeastSquares), OR
      - dict with keys: 'model', 'user_map' (userId→rowIdx), 'item_map' (movieId→colIdx)
    """
    import pickle
    if not os.path.exists(als_path):
        print(f"⚠️  ALS model not found: {als_path}")
        return None
    with open(als_path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        als_obj      = bundle.get("model")
        user_map     = bundle.get("user_map", {})
        item_map     = bundle.get("item_map", {})
        item_map_inv = {v: k for k, v in item_map.items()}
    else:
        print("⚠️  ALS pickle is a raw model without user/item maps — skipping ranking eval.")
        return None

    precisions, recalls, ndcgs, hits, aps, all_recs = [], [], [], [], [], []
    for uid in test_df["userId"].unique()[:n_users]:
        if uid not in user_map:
            continue
        ut = test_df[test_df["userId"] == uid]
        relevant = ut[ut["rating"] >= 4.0]["movieId"].tolist()
        if not relevant:
            continue
        try:
            raw_recs = als_obj.recommend(user_map[uid], None, N=k,
                                         filter_already_liked_items=True)
            rec_ids = [item_map_inv[idx] for idx, _ in raw_recs if idx in item_map_inv]
        except Exception:
            continue
        if not rec_ids:
            continue
        all_recs.extend(rec_ids)
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        hits.append(hit_rate_at_k(rec_ids, relevant, k))
        aps.append(ap_at_k(rec_ids, relevant, k))
    if not precisions:
        print("⚠️  ALS: no valid users evaluated.")
        return None
    return {"Model": "ALS (Implicit CF)",
            **_agg(precisions, recalls, ndcgs, hits, aps, all_recs, len(movies), k),
            "RMSE": "N/A"}


# ── 7. Print & Save ──────────────────────────────────────────────────────────
def print_results(results_list, k=10):
    print("\n" + "="*80)
    print(f"  EVALUATION RESULTS @ K={k}")
    print("="*80)
    df = pd.DataFrame(results_list).set_index("Model")
    print(df.to_string())
    print("="*80)
    return df


def save_results(result_df,
                 csv_path="models/evaluation_results.csv",
                 json_path="models/evaluation_results.json"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    result_df.to_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(result_df.reset_index().to_dict(orient="records"), f, indent=2)
    print(f"Saved → {csv_path}\nSaved → {json_path}")


# ── 8. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle, torch
    from src.content_based import load_movies, build_movie_soup, build_tfidf_matrix, build_cosine_sim
    from src.collaborative  import load_ratings, build_user_movie_matrix
    from src.neural_cf      import NCF

    K = 10
    ratings, movies = load_data()
    train_df, test_df = train_test_split(ratings)

    # Content-Based
    movies_cb  = build_movie_soup(movies.copy())
    _, tfidf_m = build_tfidf_matrix(movies_cb)
    cosine_sim = build_cosine_sim(tfidf_m)
    cb_result  = evaluate_content_based(movies_cb, cosine_sim, test_df, k=K)

    # SVD
    svd_result = None
    if os.path.exists("models/svd_predicted.pkl"):
        predicted_df      = pd.read_pickle("models/svd_predicted.pkl")
        user_movie_matrix = build_user_movie_matrix(ratings)
        svd_result        = evaluate_svd(predicted_df, user_movie_matrix, test_df, movies, k=K)

    # NCF
    ncf_result = None
    if os.path.exists("models/ncf_model.pt"):
        r = ratings.copy()
        r["user_idx"]  = r["userId"].astype("category").cat.codes
        r["movie_idx"] = r["movieId"].astype("category").cat.codes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = NCF(r["user_idx"].nunique(), r["movie_idx"].nunique(), embed_dim=32)
        m.load_state_dict(torch.load("models/ncf_model.pt", map_location=device, weights_only=True))
        m.to(device).eval()
        ncf_result = evaluate_ncf(m, r, movies, device, test_df, k=K)

    # ALS
    als_result = evaluate_als("models/als_model.pkl", ratings, movies, test_df, k=K)

    results   = [r for r in [cb_result, svd_result, ncf_result, als_result] if r]
    result_df = print_results(results, k=K)
    save_results(result_df)