import pandas as pd
import numpy as np
import torch
import os

from src.content_based import (
    load_movies, build_movie_soup,
    build_tfidf_matrix, build_cosine_sim,
    get_recommendations as get_cb_recommendations
)
from src.collaborative import (
    load_ratings, build_user_movie_matrix,
    get_svd_recommendations
)
from src.neural_cf import NCF, get_ncf_recommendations


# ── 1. Load all models ───────────────────────────────────────
def load_all_models(processed_path="data/processed", models_path="models"):
    print("📦 Loading all models...\n")

    ratings = load_ratings(processed_path)
    movies  = load_movies(processed_path)

    # Content-Based
    print("⏳ Building Content-Based model...")
    movies_cb            = build_movie_soup(movies.copy())
    tfidf, tfidf_matrix  = build_tfidf_matrix(movies_cb)
    cosine_sim           = build_cosine_sim(tfidf_matrix)
    print("✅ Content-Based ready\n")

    # SVD
    print("⏳ Loading SVD model...")
    svd_path          = os.path.join(models_path, "svd_predicted.pkl")
    predicted_df      = pd.read_pickle(svd_path)
    user_movie_matrix = build_user_movie_matrix(ratings)
    print("✅ SVD ready\n")

    # NCF
    print("⏳ Loading NCF model...")
    ratings["user_idx"]  = ratings["userId"].astype("category").cat.codes
    ratings["movie_idx"] = ratings["movieId"].astype("category").cat.codes
    num_users  = ratings["user_idx"].nunique()
    num_movies = ratings["movie_idx"].nunique()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncf_model = NCF(num_users, num_movies, embed_dim=32)
    ncf_model.load_state_dict(
        torch.load(os.path.join(models_path, "ncf_model.pt"), map_location=device)
    )
    ncf_model.to(device)
    ncf_model.eval()
    print("✅ NCF ready\n")

    return {
        "ratings"          : ratings,
        "movies"           : movies,
        "movies_cb"        : movies_cb,
        "cosine_sim"       : cosine_sim,
        "predicted_df"     : predicted_df,
        "user_movie_matrix": user_movie_matrix,
        "ncf_model"        : ncf_model,
        "device"           : device,
    }


# ── 2. Count user ratings ────────────────────────────────────
def get_user_rating_count(user_id, ratings):
    return len(ratings[ratings["userId"] == user_id])


# ── 3. Hybrid Recommender ────────────────────────────────────
def hybrid_recommend(user_id, models, top_n=10, cf_weight=0.7, cb_weight=0.3):
    ratings           = models["ratings"]
    movies            = models["movies"]
    movies_cb         = models["movies_cb"]
    cosine_sim        = models["cosine_sim"]
    predicted_df      = models["predicted_df"]
    user_movie_matrix = models["user_movie_matrix"]
    ncf_model         = models["ncf_model"]
    device            = models["device"]
    movie_lookup      = movies[["movieId", "title", "genres"]]

    rating_count = get_user_rating_count(user_id, ratings)
    print(f"👤 User {user_id} has rated {rating_count} movies")

    # ── Cold Start ──
    if rating_count < 20:
        print("❄️  Cold-start user — using Content-Based only\n")
        user_ratings = ratings[ratings["userId"] == user_id].sort_values(
            "rating", ascending=False
        )
        if len(user_ratings) == 0:
            print("❌ No ratings found.")
            return pd.DataFrame(columns=["title","genres","hybrid_score","method"])

        seed_movie_id = user_ratings.iloc[0]["movieId"]
        seed_title    = movies[movies["movieId"] == seed_movie_id]["title"].values[0]
        print(f"   Seed movie : '{seed_title}'\n")

        recs = get_cb_recommendations(seed_title, movies_cb, cosine_sim, top_n=top_n)
        if isinstance(recs, list) or len(recs) == 0:
            return pd.DataFrame(columns=["title","genres","hybrid_score","method"])
        recs["hybrid_score"] = recs["similarity_score"]
        recs["method"]       = "Content-Based (Cold Start)"
        return recs[["movieId","title","genres","hybrid_score","method"]]

    # ── Warm User ──
    print(f"🔥 Warm user — Hybrid (CF {int(cf_weight*100)}% + CB {int(cb_weight*100)}%)\n")

    # SVD recommendations
    svd_recs = get_svd_recommendations(
        user_id, user_movie_matrix, predicted_df, movies, top_n=top_n*3
    )
    svd_recs = svd_recs.rename(columns={"predicted_rating": "svd_score"})
    svd_min  = svd_recs["svd_score"].min()
    svd_max  = svd_recs["svd_score"].max()
    svd_recs["svd_norm"] = (svd_recs["svd_score"] - svd_min) / (svd_max - svd_min + 1e-9)

    # NCF recommendations
    ncf_recs = get_ncf_recommendations(
        user_id, ncf_model, ratings, movies, device, top_n=top_n*3
    )
    ncf_recs = ncf_recs.rename(columns={"ncf_score": "ncf_score"})

    # Merge SVD + NCF (outer to keep all candidates)
    cf_merged = pd.merge(
        svd_recs[["movieId", "svd_norm"]],
        ncf_recs[["movieId", "ncf_score"]],
        on="movieId", how="outer"
    ).fillna(0)

    # Attach titles
    cf_merged = cf_merged.merge(movie_lookup, on="movieId", how="left")
    cf_merged = cf_merged[
        cf_merged["title"].apply(lambda x: isinstance(x, str) and len(x) > 1)
    ]
    cf_merged["cf_score"] = (cf_merged["svd_norm"] + cf_merged["ncf_score"]) / 2

    # CB seed from user's top rated movie
    user_top   = ratings[ratings["userId"] == user_id].sort_values("rating", ascending=False)
    seed_id    = user_top.iloc[0]["movieId"]
    seed_title = movies[movies["movieId"] == seed_id]["title"].values[0]
    print(f"   CB seed movie : '{seed_title}'")
    cb_recs    = get_cb_recommendations(seed_title, movies_cb, cosine_sim, top_n=top_n*3)
    cb_recs    = cb_recs.rename(columns={"similarity_score": "cb_score"})

    # Merge CF + CB
    final = pd.merge(
        cf_merged[["movieId", "title", "genres", "cf_score"]],
        cb_recs[["movieId", "cb_score"]],
        on="movieId", how="left"
    ).fillna(0)

    final = final[final["title"].apply(lambda x: isinstance(x, str) and len(x) > 1)]
    final["hybrid_score"] = (cf_weight * final["cf_score"] +
                             cb_weight  * final["cb_score"])
    final["method"] = "Hybrid (CF + CB)"
    final = final.sort_values("hybrid_score", ascending=False).head(top_n)
    final = final.reset_index(drop=True)

    return final[["movieId", "title", "genres", "hybrid_score", "method"]]


# ── 4. Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    models = load_all_models()

    print("=" * 60)
    print("TEST 1 — Warm User (User ID: 1)")
    print("=" * 60)
    recs1 = hybrid_recommend(1, models, top_n=10)
    if len(recs1) > 0:
        print(recs1[["title", "genres", "hybrid_score", "method"]].to_string(index=False))
    else:
        print("No recommendations found.")

    print("\n" + "=" * 60)
    print("TEST 2 — Cold Start User (User ID: 2)")
    print("=" * 60)
    recs2 = hybrid_recommend(2, models, top_n=10)
    if len(recs2) > 0:
        print(recs2[["title", "genres", "hybrid_score", "method"]].to_string(index=False))
    else:
        print("No recommendations found.")