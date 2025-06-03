# Scalable Hybrid Recommendation System on Amazon Product Reviews

This project builds a scalable and intelligent **hybrid recommendation system** using Amazon product review data. It combines **Graph Neural Networks (GNNs)**, **Collaborative Filtering (ALS)**, and **Natural Language Processing (NLP)** to provide accurate and explainable product recommendations.


---

## 🛠️ Tech Stack

- `Python`, `Pandas`, `NumPy`, `scikit-learn`
- `DGL` (Graph Neural Networks)
- `PySpark` / `Dask` for scalable data processing
- `Hugging Face Transformers` for NLP
- `Streamlit` for UI
- `Matplotlib`, `Plotly` for visualizations

# Dataset
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
- Run phase_1.py script for all 3 datasets.

---
# How to Run the Project Step-by-Step
| Phase | Script       | Description                                                                                        |
| ----- | ------------ | -------------------------------------------------------------------------------------------------- |
| 1️⃣   | `phase_1.py` | **Data Loading & Preprocessing** — Cleans raw Amazon review data and saves it in Parquet format    |
| 2️⃣   | `phase_2.py` | **Exploratory Data Analysis (EDA)** — Analyzes ratings, brand/category trends, review distribution |
| 3️⃣   | `phase_3.py` | **NLP Preprocessing** — Cleans review text, applies sentiment analysis using BERT                  |
| 4️⃣   | `phase_4.py` | **Graph Analysis** — Builds co-purchase graph and computes product embeddings with GNN (GraphSAGE) |
| 5️⃣   | `phase_5.py` | **Hybrid Recommendation Modeling** — Merges GNN, CF, and review data to create a hybrid model      |
| 6️⃣   | `phase_6.py` | **Model Evaluation** — Calculates Precision\@10 and Recall\@10 on a validation split               |

Run the following phases sequentially.
Use Python version 3.10

# Install Dependencies
Run pip install -r requirements.txt in your IDE (VS Code or any other) terminal.

# Run the Dashboard
streamlit run app/streamlit_app.py


