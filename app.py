import streamlit as st
import pandas as pd
from recommender_utils import recommend_for_user

st.title("🔍 Amazon Hybrid Recommender System")

df = pd.read_parquet(r"D:\Projects\Internship_Project\data\combined_cleaned.parquet")
usernames = sorted(df["reviews.username"].dropna().unique())

username = st.selectbox("Select a user", usernames)

if st.button("Get Recommendations"):
    recommendations = recommend_for_user(username)

    if recommendations.empty:
        st.warning("No recommendations available for this user.")
    else:
        st.success(f"Top {len(recommendations)} Recommendations for {username}")
        st.dataframe(recommendations)
