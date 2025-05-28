import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from transformers import pipeline

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Example usage
df = pd.read_parquet(r"path\combined.parquet")
df["cleaned_review"] = df["reviews.text"].astype(str).apply(clean_text)
df.to_parquet(r"path\combined_cleaned.parquet", index=False)
print("✅ Cleaned reviews saved.")


# Load sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze a subset (full dataset can be very slow)
sample = df["cleaned_review"].sample(1000, random_state=42)

sentiments = sentiment_pipeline(sample.tolist(), truncation=True)
df_sent = sample.reset_index().copy()
df_sent["sentiment"] = [s["label"] for s in sentiments]
df_sent["score"] = [s["score"] for s in sentiments]

df_sent.head()


topic_model = BERTopic()
topics, _ = topic_model.fit_transform(df["cleaned_review"].dropna().tolist())
topic_model.get_topic_info()

