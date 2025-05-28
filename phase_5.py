import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load hybrid embeddings
hybrid_df = pd.read_csv(r"path\product_gnn_embeddings.csv")
embedding_columns = [col for col in hybrid_df.columns if col.startswith("emb")]

# Load full review data
review_df = pd.read_parquet(r"path\combined_cleaned.parquet")

def recommend_for_user(username, top_k=10):
    liked_items = review_df[
        (review_df["reviews.username"] == username) &
        (review_df["reviews.rating"] >= 4)
    ]["id"].unique()

    liked_embeddings = hybrid_df[hybrid_df["product_id"].isin(liked_items)]

    if liked_embeddings.empty:
        return pd.DataFrame(columns=["product_id", "score"])

    # Compute average vector
    user_vector = liked_embeddings[embedding_columns].mean(axis=0).values
    item_vectors = hybrid_df[embedding_columns].values

    scores = cosine_similarity([user_vector], item_vectors).flatten()
    hybrid_df["score"] = scores

    recommendations = hybrid_df.sort_values("score", ascending=False).head(top_k)
    return recommendations[["product_id", "score"]]
