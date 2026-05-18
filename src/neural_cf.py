import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle

# ── 1. Dataset ──────────────────────────────────────────────
class RatingsDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users  = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


# ── 2. NCF Model (GMF + MLP) ────────────────────────────────
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=32, layers=[64, 32, 16]):
        super(NCF, self).__init__()

        # GMF embeddings (element-wise product)
        self.gmf_user  = nn.Embedding(num_users, embed_dim)
        self.gmf_movie = nn.Embedding(num_movies, embed_dim)

        # MLP embeddings (concatenation through dense layers)
        self.mlp_user  = nn.Embedding(num_users, embed_dim)
        self.mlp_movie = nn.Embedding(num_movies, embed_dim)

        # MLP layers
        mlp_input_size = embed_dim * 2
        mlp_layers = []
        for out_size in layers:
            mlp_layers.append(nn.Linear(mlp_input_size, out_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            mlp_input_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Final output layer: GMF output + MLP output
        self.output = nn.Linear(embed_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, movie):
        # GMF pathway
        gmf_out = self.gmf_user(user) * self.gmf_movie(movie)

        # MLP pathway
        mlp_in  = torch.cat([self.mlp_user(user), self.mlp_movie(movie)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Combine and output
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.sigmoid(self.output(combined))
        return out.squeeze()


# ── 3. Train ────────────────────────────────────────────────
def train_ncf(ratings_df, epochs=5, batch_size=512, lr=0.001, embed_dim=32):
    # Encode userId and movieId to 0-based indices
    ratings_df["user_idx"]  = ratings_df["userId"].astype("category").cat.codes
    ratings_df["movie_idx"] = ratings_df["movieId"].astype("category").cat.codes

    num_users  = ratings_df["user_idx"].nunique()
    num_movies = ratings_df["movie_idx"].nunique()

    print(f"✅ Users  : {num_users:,}")
    print(f"✅ Movies : {num_movies:,}")

    # Normalise ratings to 0-1 for binary cross-entropy
    ratings_df["rating_norm"] = ratings_df["rating"] / 5.0

    # Train/test split by timestamp (last 20% = test)
    ratings_df = ratings_df.sort_values("timestamp")
    split      = int(len(ratings_df) * 0.8)
    train_df   = ratings_df.iloc[:split]
    test_df    = ratings_df.iloc[split:]

    print(f"   Train samples : {len(train_df):,}")
    print(f"   Test  samples : {len(test_df):,}")

    # DataLoaders
    train_dataset = RatingsDataset(
        train_df["user_idx"].values,
        train_df["movie_idx"].values,
        train_df["rating_norm"].values
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️  Device : {device}")

    model     = NCF(num_users, num_movies, embed_dim=embed_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\n⏳ Training NCF for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for users, movies, ratings in train_loader:
            users   = users.to(device)
            movies  = movies.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            preds = model(users, movies)
            loss  = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch}/{epochs}  →  Loss: {avg_loss:.4f}")

    print(f"\n✅ NCF Training complete!")
    return model, ratings_df, num_users, num_movies, device


# ── 4. Recommend ────────────────────────────────────────────
def get_ncf_recommendations(user_id, model, ratings_df, movies_df, device, top_n=10):
    model.eval()

    user_map  = ratings_df[["userId","user_idx"]].drop_duplicates().set_index("userId")
    movie_map = ratings_df[["movieId","movie_idx"]].drop_duplicates().set_index("movie_idx")

    if user_id not in user_map.index:
        print(f"❌ User {user_id} not found.")
        return []

    user_idx = int(user_map.loc[user_id, "user_idx"])

    # Movies user has NOT rated
    rated_movies = ratings_df[ratings_df["userId"] == user_id]["movie_idx"].values
    all_movies   = ratings_df["movie_idx"].unique()
    unrated      = [m for m in all_movies if m not in rated_movies]

    # Predict scores for all unrated movies
    with torch.no_grad():
        user_tensor  = torch.tensor([user_idx] * len(unrated), dtype=torch.long).to(device)
        movie_tensor = torch.tensor(unrated, dtype=torch.long).to(device)
        scores       = model(user_tensor, movie_tensor).cpu().numpy()

    # Top N
    top_indices  = np.argsort(scores)[::-1][:top_n]
    top_movie_idx = [unrated[i] for i in top_indices]
    top_scores    = [round(float(scores[i]), 4) for i in top_indices]

    top_movie_ids = movie_map.loc[top_movie_idx, "movieId"].values

    result = pd.DataFrame({
        "movieId": top_movie_ids,
        "ncf_score": top_scores
    })
    result = result.merge(movies_df[["movieId","title","genres"]], on="movieId")
    result = result[["movieId","title","genres","ncf_score"]]
    return result


# ── 5. Save ─────────────────────────────────────────────────
def save_ncf_model(model, ratings_df, out_path="models"):
    os.makedirs(out_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_path, "ncf_model.pt"))
    ratings_df[["userId","user_idx","movieId","movie_idx"]].drop_duplicates().to_csv(
        os.path.join(out_path, "ncf_index_map.csv"), index=False
    )
    print(f"✅ NCF model saved to '{out_path}/ncf_model.pt'")


# ── 6. Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    ratings = pd.read_csv("data/processed/ratings.csv")
    movies  = pd.read_csv("data/processed/movies.csv")

    model, ratings_df, num_users, num_movies, device = train_ncf(
        ratings, epochs=5, batch_size=512, lr=0.001, embed_dim=32
    )

    save_ncf_model(model, ratings_df)

    test_user = 1
    print(f"\n🎬 Top 10 NCF recommendations for User {test_user}:\n")
    recs = get_ncf_recommendations(test_user, model, ratings_df, movies, device)
    print(recs.to_string(index=False))