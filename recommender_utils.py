import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load hybrid product embeddings (GNN + CF)
try:
    hybrid_df = pd.read_csv(r"D:\Projects\Internship_Project\data\product_gnn_embeddings.csv")
except FileNotFoundError:
    raise FileNotFoundError("❌ models/hybrid_df.csv not found. Please ensure the file exists.")

# Identify embedding columns (starts with emb or emb_cf)
embedding_columns = [col for col in hybrid_df.columns if col.startswith("emb")]

# Load review dataset
try:
    review_df = pd.read_parquet(r"D:\Projects\Internship_Project\data\combined_cleaned.parquet")
except FileNotFoundError:
    raise FileNotFoundError("❌ combined_cleaned.parquet not found. Please ensure the review file exists.")

def recommend_for_user(username: str, top_k: int = 10) -> pd.DataFrame:
    """
    Recommend top_k products for a given username using hybrid (GNN + CF) embeddings.
    
    Parameters:
        username (str): The username to generate recommendations for.
        top_k (int): Number of top recommendations to return.
        
    Returns:
        pd.DataFrame: A dataframe with columns ['product_id', 'score'].
    """
    # Step 1: Get liked items (rated >= 4)
    liked_items = review_df[
        (review_df["reviews.username"] == username) & 
        (review_df["reviews.rating"] >= 4)
    ]["id"].unique()

    # Step 2: Get hybrid embeddings for liked items
    liked_embeddings = hybrid_df[hybrid_df["product_id"].isin(liked_items)]

    if liked_embeddings.empty:
        return pd.DataFrame(columns=["product_id", "score"])  # Empty fallback

    # Step 3: Compute user vector (mean of liked embeddings)
    embedding_matrix = liked_embeddings[embedding_columns].values
    user_vector = embedding_matrix.mean(axis=0).reshape(1, -1)

    # Step 4: Compute similarity scores
    item_vectors = hybrid_df[embedding_columns].values
    scores = cosine_similarity(user_vector, item_vectors).flatten()

    # Step 5: Append scores and sort
    hybrid_df["score"] = scores
    recommendations = hybrid_df.sort_values(by="score", ascending=False).head(top_k)

    return recommendations[["product_id", "score"]]
