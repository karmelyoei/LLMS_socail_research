import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load filtered comments (or submissions)
df = pd.read_parquet("../data/processed/comments_semantic.parquet")
texts = df["body"].dropna().astype(str).tolist()

# Use embeddings from sentence-transformers
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(embedding_model=embed_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

df["topic"] = topics
df["topic_prob"] = probs   # <-- FIXED

# Add year
df["year"] = pd.to_datetime(df["created_utc"], unit="s").dt.year

# Save
df.to_parquet("data/processed/comments_with_topics.parquet", index=False)

# Inspect topics
topic_model.get_topic_info().head(20)

# Trends over years
freq_by_year = df.groupby(["year", "topic"]).size().unstack(fill_value=0)
freq_by_year.to_csv("data/processed/topic_trends.csv")

topic_model.save("models/bertopic_model")
