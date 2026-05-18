import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import os
import pickle

def load_ratings(processed_path="data/processed"):
    ratings = pd.read_csv(os.path.join(processed_path, "ratings.csv"))
    return ratings

def build_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    print(f"✅ User-Movie matrix shape : {user_movie_matrix.shape}")
    print(f"   Rows (users)   : {user_movie_matrix.shape[0]:,}")
    print(f"   Cols (movies)  : {user_movie_matrix.shape[1]:,}")
    return user_movie_matrix

def train_svd(user_movie_matrix, n_factors=50):
    matrix = csr_matrix(user_movie_matrix.values)

    # Normalise: subtract each user's mean rating
    user_ratings_mean = np.mean(user_movie_matrix.values, axis=1)
    matrix_demeaned   = user_movie_matrix.values - user_ratings_mean.reshape(-1, 1)

    print(f"\n⏳ Running SVD with {n_factors} factors... (may take 30-60 seconds)")
    U, sigma, Vt = svds(csr_matrix(matrix_demeaned), k=n_factors)

    sigma = np.diag(sigma)

    predicted_ratings = (np.dot(np.dot(U, sigma), Vt)
                         + user_ratings_mean.reshape(-1, 1))

    predicted_df = pd.DataFrame(
        predicted_ratings,
        columns=user_movie_matrix.columns,
        index=user_movie_matrix.index
    )

    print(f"✅ SVD training complete!")
    print(f"   Predicted ratings matrix shape : {predicted_df.shape}")
    return predicted_df, U, sigma, Vt, user_ratings_mean

def get_svd_recommendations(user_id, user_movie_matrix,
                            predicted_df, movies_df, top_n=10):
    # Movies the user has NOT yet rated
    user_row      = user_movie_matrix.loc[user_id]
    unrated_movies = user_row[user_row == 0].index.tolist()

    # ── FIX: filter only movieIds that exist in predicted_df columns ──
    valid_movies = [m for m in unrated_movies if m in predicted_df.columns]
    if not valid_movies:
        return pd.DataFrame()

    # Get predicted ratings for valid unrated movies only
    user_predictions = predicted_df.loc[user_id, valid_movies]

    # Sort by predicted rating descending
    top_movies = user_predictions.sort_values(ascending=False).head(top_n)

    result = pd.DataFrame({
        "movieId"         : top_movies.index,
        "predicted_rating": top_movies.values.round(2)
    })
    result = result.merge(
        movies_df[["movieId", "title", "genres"]], on="movieId"
    )
    result = result[["movieId", "title", "genres", "predicted_rating"]]
    return result

def save_model(predicted_df, out_path="models"):
    os.makedirs(out_path, exist_ok=True)
    predicted_df.to_pickle(os.path.join(out_path, "svd_predicted.pkl"))
    print(f"\n✅ SVD model saved to '{out_path}/svd_predicted.pkl'")

if __name__ == "__main__":
    ratings = load_ratings()
    movies  = pd.read_csv("data/processed/movies.csv")

    user_movie_matrix = build_user_movie_matrix(ratings)

    predicted_df, U, sigma, Vt, user_mean = train_svd(
        user_movie_matrix, n_factors=50
    )

    save_model(predicted_df)

    test_user = 1
    print(f"\n🎬 Top 10 recommendations for User {test_user}:\n")
    recommendations = get_svd_recommendations(
        test_user, user_movie_matrix, predicted_df, movies
    )
    print(recommendations.to_string(index=False))