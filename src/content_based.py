import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_movies(processed_path="data/processed"):
    movies = pd.read_csv(os.path.join(processed_path, "movies.csv"))
    return movies

def build_movie_soup(movies):
    movies["genre_clean"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").fillna("")
    movies["soup"] = movies["genre_clean"] + " " + movies["year"]
    print("✅ Movie soup built successfully")
    print(f"📽️  Sample soup:\n{movies[['title','soup']].head(3).to_string(index=False)}")
    return movies

def build_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["soup"])
    print(f"\n✅ TF-IDF matrix shape : {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix

def build_cosine_sim(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"✅ Cosine similarity matrix shape : {cosine_sim.shape}")
    return cosine_sim

def get_recommendations(title, movies, cosine_sim, top_n=10):
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    if title not in indices:
        print(f"❌ Movie '{title}' not found in database.")
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 4) for i in sim_scores]
    result = movies[["movieId", "title", "genres"]].iloc[movie_indices].copy()
    result["similarity_score"] = scores
    return result

if __name__ == "__main__":
    movies = load_movies()
    movies = build_movie_soup(movies)
    tfidf, tfidf_matrix = build_tfidf_matrix(movies)
    cosine_sim = build_cosine_sim(tfidf_matrix)
    test_movie = "Toy Story (1995)"
    print(f"\n🎬 Top 10 movies similar to: '{test_movie}'\n")
    recommendations = get_recommendations(test_movie, movies, cosine_sim, top_n=10)
    print(recommendations.to_string(index=False))