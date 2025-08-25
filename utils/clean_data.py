# semantic_filter.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load your parquet dump
submissions = pd.read_parquet("../data/raw/submissions.parquet")
comments = pd.read_parquet("../data/raw/comments.parquet")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define anchor queries (can expand this list)
queries = [
    "AI jobs",
    "automation job loss",
    "future of work",
    "AI retraining reskilling",
    "AI replacing jobs",
    "career change due to AI"
]
query_embs = model.encode(queries, convert_to_tensor=True)

def filter_relevant(df, text_col="selftext", threshold=0.35):
    texts = df[text_col].fillna("").astype(str).tolist()
    embs = model.encode(texts, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    sims = util.cos_sim(embs, query_embs).max(dim=1).values.cpu().numpy()
    df["semantic_score"] = sims
    return df[df["semantic_score"] >= threshold]

# Filter submissions by title+selftext
submissions["text_blob"] = submissions["title"].fillna("") + " " + submissions["selftext"].fillna("")
filtered_subs = filter_relevant(submissions, "text_blob")

# Filter comments
filtered_comments = filter_relevant(comments, "body")

print("Kept:", len(filtered_subs), "submissions,", len(filtered_comments), "comments")

# Save
filtered_subs.to_parquet("data/processed/submissions_semantic.parquet", index=False)
filtered_comments.to_parquet("data/processed/comments_semantic.parquet", index=False)
