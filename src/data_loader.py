import pandas as pd
import os

def load_movielens(data_path="data/raw/ml-latest"):
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(data_path, "movies.csv"))

    ratings.columns = [c.strip() for c in ratings.columns]
    movies.columns  = [c.strip() for c in movies.columns]

    print(f"✅ Ratings loaded : {ratings.shape[0]:,} rows")
    print(f"✅ Movies loaded  : {movies.shape[0]:,} rows")
    print(f"📊 Rating scale   : {ratings['rating'].min()} to {ratings['rating'].max()}")
    print(f"👤 Unique users   : {ratings['userId'].nunique():,}")
    print(f"🎬 Unique movies  : {ratings['movieId'].nunique():,}")
    return ratings, movies

def save_processed(ratings, movies, out_path="data/processed"):
    os.makedirs(out_path, exist_ok=True)
    ratings.to_csv(os.path.join(out_path, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(out_path,  "movies.csv"),  index=False)
    # create empty users.csv so other scripts don't break
    pd.DataFrame().to_csv(os.path.join(out_path, "users.csv"), index=False)
    print(f"✅ Saved to '{out_path}/'")

if __name__ == "__main__":
    ratings, movies = load_movielens()
    save_processed(ratings, movies)