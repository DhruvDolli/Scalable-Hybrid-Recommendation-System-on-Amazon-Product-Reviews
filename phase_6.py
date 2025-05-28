import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load data ---
df = pd.read_parquet(r"path\combined_cleaned.parquet")

# --- Step 2: Filter users with at least 5 reviews ---
user_counts = df["reviews.username"].value_counts()
valid_users = user_counts[user_counts >= 5].index
df = df[df["reviews.username"].isin(valid_users)]

# --- Step 3: Train/Test Split for each user ---
train_data, test_data = [], []
for user in valid_users:
    user_reviews = df[df["reviews.username"] == user]
    train, test = train_test_split(user_reviews, test_size=0.2, random_state=42)
    train_data.append(train)
    test_data.append(test)

train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

# --- Step 4: Re-encode users and items ---
train_df["user_index"] = train_df["reviews.username"].astype("category").cat.codes
train_df["item_index"] = train_df["id"].astype("category").cat.codes

user_ids = train_df["reviews.username"].astype("category")
item_ids = train_df["id"].astype("category")

# --- Step 5: Build sparse matrix for ALS ---
train_matrix = coo_matrix(
    (train_df["reviews.rating"], (train_df["user_index"], train_df["item_index"]))
).tocsr()

# --- Step 6: Train ALS Model ---
model = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=20)
model.fit(train_matrix)

# --- Step 7: Load hybrid_df (GNN + CF embeddings) ---
gnn_df = pd.read_csv(r"path\product_gnn_embeddings.csv")

cf_embeddings = pd.DataFrame(model.item_factors, columns=[f"emb_cf_{i}" for i in range(32)])
cf_embeddings["item_index"] = range(len(cf_embeddings))
cf_embeddings["product_id"] = item_ids.cat.categories[cf_embeddings["item_index"]].values

hybrid_df = pd.merge(gnn_df, cf_embeddings, on="product_id")
popular_products = df["id"].value_counts()[df["id"].value_counts() >= 5].index
hybrid_df = hybrid_df[hybrid_df["product_id"].isin(popular_products)]



# --- Step 8: Evaluation Metrics ---
def precision_at_k(recommended, actual, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(actual)) / k

def recall_at_k(recommended, actual, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(actual)) / len(actual) if actual else 0

# --- Step 9: Evaluate for Each User ---
embedding_columns = [col for col in hybrid_df.columns if col.startswith("emb")]
item_vectors = hybrid_df[embedding_columns].values

user_precision_scores = []
user_recall_scores = []

for username in test_df["reviews.username"].unique():
    # Get liked products from training set
    liked_items = train_df[(train_df["reviews.username"] == username) & (train_df["reviews.rating"] >= 4)]["id"].unique()
    liked_embeddings = hybrid_df[hybrid_df["product_id"].isin(liked_items)]
    

    if liked_embeddings.empty:
        continue  # Skip cold-start users

    # Compute user vector as average of liked product vectors
    user_vector = liked_embeddings[embedding_columns].mean(axis=0).values

    # Score all products
    scores = cosine_similarity([user_vector], item_vectors).flatten()
    hybrid_df["score"] = scores

    # Top 10 recommendations
    top_k = hybrid_df.sort_values("score", ascending=False)["product_id"].head(10).tolist()

    # Actual items in test set
    actual_items = test_df[test_df["reviews.username"] == username]["id"].unique().tolist()

    user_precision_scores.append(precision_at_k(top_k, actual_items, k=10))
    user_recall_scores.append(recall_at_k(top_k, actual_items, k=10))

# --- Step 10: Final Metrics ---
print(f"📈 Precision@10: {np.mean(user_precision_scores):.4f}")
print(f"📈 Recall@10:    {np.mean(user_recall_scores):.4f}")
