import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix
import os
import pickle

def load_ratings(processed_path="data/processed"):
    ratings = pd.read_csv(os.path.join(processed_path, "ratings.csv"))
    return ratings

def build_confidence_matrix(ratings, alpha=40):
    # ALS works on implicit feedback
    # Convert ratings to confidence: Cui = 1 + alpha * rating
    ratings["confidence"] = 1 + alpha * ratings["rating"]

    # Create userId and movieId integer codes (0-based index)
    ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
    ratings["movie_idx"] = ratings["movieId"].astype("category").cat.codes

    # Build sparse matrix: rows=users, cols=movies, values=confidence
    user_count = ratings["user_idx"].nunique()
    movie_count = ratings["movie_idx"].nunique()

    confidence_matrix = csr_matrix(
        (ratings["confidence"].values,
         (ratings["user_idx"].values, ratings["movie_idx"].values)),
        shape=(user_count, movie_count)
    )

    print(f"✅ Confidence matrix built")
    print(f"   Shape  : {confidence_matrix.shape}")
    print(f"   Alpha  : {alpha}")
    print(f"   Sample confidence values : {ratings['confidence'].describe().round(2).to_dict()}")

    return confidence_matrix, ratings

def train_als(confidence_matrix, factors=50, iterations=30, regularization=0.01):
    print(f"\n⏳ Training ALS model...")
    print(f"   Factors       : {factors}")
    print(f"   Iterations    : {iterations}")
    print(f"   Regularization: {regularization}")

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42
    )

    # ALS expects item-user matrix (transpose of user-item)
    model.fit(confidence_matrix.T)

    print(f"✅ ALS training complete!")
    return model

def get_als_recommendations(user_id, model, confidence_matrix, ratings, movies_df, top_n=10):
    # Get user index from userId
    user_map = ratings[["userId", "user_idx"]].drop_duplicates().set_index("userId")
    movie_map = ratings[["movieId", "movie_idx"]].drop_duplicates().set_index("movie_idx")

    if user_id not in user_map.index:
        print(f"❌ User {user_id} not found.")
        return []

    user_idx = int(user_map.loc[user_id, "user_idx"])

    # Get recommendations
    recommended_items, scores = model.recommend(
        user_idx,
        confidence_matrix[user_idx],
        N=top_n,
        filter_already_liked_items=True
    )

    # Fix: filter only valid movie indices before lookup
    valid_indices = [i for i in recommended_items if i in movie_map.index]
    valid_scores  = [scores[j] for j, i in enumerate(recommended_items) if i in movie_map.index]

    # Map movie indices back to movieIds
    recommended_movie_ids = movie_map.loc[valid_indices, "movieId"].values

    result = pd.DataFrame({
        "movieId": recommended_movie_ids,
        "als_score": np.round(valid_scores, 4)
    })
    result = result.merge(movies_df[["movieId", "title", "genres"]], on="movieId")
    result = result[["movieId", "title", "genres", "als_score"]]

    return result

def save_als_model(model, out_path="models"):
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "als_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ ALS model saved to '{out_path}/als_model.pkl'")

if __name__ == "__main__":
    # Load data
    ratings = load_ratings()
    movies = pd.read_csv("data/processed/movies.csv")

    # Build confidence matrix
    confidence_matrix, ratings = build_confidence_matrix(ratings, alpha=40)

    # Train ALS
    model = train_als(confidence_matrix, factors=50, iterations=30, regularization=0.01)

    # Save model
    save_als_model(model)

    # Test recommendations for user ID 1
    test_user = 1
    print(f"\n🎬 Top 10 ALS recommendations for User {test_user}:\n")
    recommendations = get_als_recommendations(test_user, model, confidence_matrix, ratings, movies)
    print(recommendations.to_string(index=False))