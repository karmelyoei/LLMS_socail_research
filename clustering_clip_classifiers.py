#!/usr/bin/env python3
"""
AI Job Risk Labeling from Reddit Comments using CLIP (text-only)
----------------------------------------------------------------
- Loads data from data/processed/comments_with_topics.parquet
- Computes CLIP text embeddings
- Scores each post by cosine similarity to an "AI job risk" prototype prompt
- Adds binary label via threshold (configurable)
- (Optional) uses weak ground-truth from a keyword heuristic or an existing column for evaluation
- Plots histograms and threshold curves (PR / ROC if ground truth available)
- Clustering baseline (KMeans, 2 clusters) with automatic label mapping

Dependencies:
  pip install pandas pyarrow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install transformers scikit-learn matplotlib tqdm

Note: Runs on CPU by default. Adjust BATCH_SIZE for speed/memory trade-offs.
"""
from __future__ import annotations

import os
import re
import math
import json
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    silhouette_score,
    adjusted_rand_score,
    f1_score,
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from transformers import CLIPTokenizer, CLIPModel

# ---------------------------- Config ---------------------------- #
@dataclass
class Config:
    parquet_path: str = "data/processed/comments_with_topics.parquet"
    text_col: str = "body"
    id_col: str = "comment_id"
    output_path: str = "data/processed/comments_with_ai_job_risk.parquet"
    cache_emb_path: str = "data/processed/clip_text_embeddings.npy"

    # CLIP model
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    max_length: int = 77  # CLIP text max tokens

    # Threshold for similarity -> label (can be tuned)
    initial_threshold: float = 0.25

    # If you already have a ground-truth-like column, set here (optional)
    ground_truth_col: Optional[str] = None  # e.g., "ai_job_risk_gt" if you add it later

    # Weak label heuristic (used only if ground_truth_col is None)
    use_weak_labels: bool = True

CFG = Config()

# Positive prototype prompts (ensemble for robustness)
POSITIVE_PROMPTS = [
    "This post is about fear of losing a job due to AI or automation.",
    "The writer worries their job may be replaced by artificial intelligence.",
    "Concerns about AI causing unemployment or layoffs.",
    "Risk of losing employment because of AI technologies.",
]
NEGATIVE_PROMPTS = [
    "This post is unrelated to AI causing job loss.",
    "General discussion not about losing a job to AI.",
]

# Weak keyword heuristic (very conservative):
AI_JOB_KEYWORDS = re.compile(
    r"\b(ai|artificial intelligence|automation|automated|llm|gpt|chatgpt|model|robots?)\b",
    re.I,
)
RISK_KEYWORDS = re.compile(
    r"\b(job(s)?|career|work|role|position|employment|layoff(s)?|redundant|fired|replace(d|ment)\b|risk|threat)\b",
    re.I,
)


# ---------------------------- Utils ---------------------------- #
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # basic cleaning
    df = df.dropna(subset=[CFG.text_col]).copy()
    df[CFG.text_col] = df[CFG.text_col].astype(str).str.strip()
    return df


def get_clip_components(model_name: str, device: str):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval().to(device)
    return tokenizer, model


def embed_texts(texts: List[str], tokenizer: CLIPTokenizer, model: CLIPModel, device: str) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(texts), CFG.batch_size), desc="Embedding texts"):
        batch = texts[i : i + CFG.batch_size]
        with torch.no_grad():
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=CFG.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            feats = model.get_text_features(**tokens)
            # CLIP recommends normalizing features for cosine similarity
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            embs.append(feats.cpu().numpy())
    return np.vstack(embs)


def ensure_embeddings(df: pd.DataFrame, force: bool = False) -> np.ndarray:
    # Cache to avoid recomputation
    if (not force) and os.path.exists(CFG.cache_emb_path):
        arr = np.load(CFG.cache_emb_path)
        if arr.shape[0] == len(df):
            return arr
    tokenizer, model = get_clip_components(CFG.model_name, CFG.device)
    arr = embed_texts(df[CFG.text_col].tolist(), tokenizer, model, CFG.device)
    os.makedirs(os.path.dirname(CFG.cache_emb_path), exist_ok=True)
    np.save(CFG.cache_emb_path, arr)
    return arr


def prototype_embedding(prompts: List[str], tokenizer: CLIPTokenizer, model: CLIPModel, device: str) -> np.ndarray:
    emb = embed_texts(prompts, tokenizer, model, device)
    # Simple average of normalized vectors stays normalized if we renormalize
    emb = emb.mean(axis=0, keepdims=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, D], b: [M, D] -> returns [N, M]
    return (a @ b.T)


def add_semantic_score_and_label(df: pd.DataFrame, text_emb: np.ndarray, threshold: float) -> pd.DataFrame:
    tokenizer, model = get_clip_components(CFG.model_name, CFG.device)
    pos_emb = prototype_embedding(POSITIVE_PROMPTS, tokenizer, model, CFG.device)  # [1, D]

    # Score is cosine similarity to positive prototype (range approx [-1, 1])
    sims = cosine_sim(text_emb, pos_emb).ravel()
    df = df.copy()
    df["ai_job_risk_score"] = sims
    df["ai_job_risk_label"] = (df["ai_job_risk_score"] >= threshold).astype(int)
    return df


def get_ground_truth(df: pd.DataFrame) -> Optional[pd.Series]:
    if CFG.ground_truth_col and CFG.ground_truth_col in df.columns:
        y = df[CFG.ground_truth_col].astype(int)
        return y
    if CFG.use_weak_labels:
        # Weak GT: requires at least one AI keyword and one risk/employment keyword
        txt = df[CFG.text_col].fillna("")
        ai_hit = txt.str.contains(AI_JOB_KEYWORDS)
        risk_hit = txt.str.contains(RISK_KEYWORDS)
        weak = (ai_hit & risk_hit).astype(int)
        return weak
    return None


# ---------------------------- Evaluation & Plots ---------------------------- #

def evaluate_thresholds(df: pd.DataFrame, y_true: Optional[pd.Series] = None, outdir: str = "figs"):
    os.makedirs(outdir, exist_ok=True)
    scores = df["ai_job_risk_score"].values
    y_pred = df["ai_job_risk_label"].values

    if y_true is not None:
        # Classification metrics
        try:
            auroc = roc_auc_score(y_true, scores)
        except ValueError:
            auroc = float("nan")
        try:
            ap = average_precision_score(y_true, scores)
        except ValueError:
            ap = float("nan")
        print(f"AUROC: {auroc:.4f} | Average Precision (AUPRC): {ap:.4f}")
        print("\nClassification Report @ threshold=", CFG.initial_threshold)
        print(classification_report(y_true, y_pred, digits=3))

        # PR curve
        precision, recall, thr = precision_recall_curve(y_true, scores)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (CLIP similarity)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=160)

        # ROC curve
        fpr, tpr, thr = roc_curve(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC (CLIP similarity)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=160)

        # Threshold sweep for F1
        ts = np.linspace(scores.min(), scores.max(), 101)
        f1s = []
        for t in ts:
            f1s.append(f1_score(y_true, (scores >= t).astype(int)))
        best_idx = int(np.argmax(f1s))
        best_t = float(ts[best_idx])
        best_f1 = float(f1s[best_idx])
        print(f"Best F1={best_f1:.3f} at threshold={best_t:.3f}")
        plt.figure()
        plt.plot(ts, f1s)
        plt.xlabel("Threshold")
        plt.ylabel("F1 score")
        plt.title("F1 vs Threshold (CLIP similarity)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "f1_vs_threshold.png"), dpi=160)

    # Score histogram
    plt.figure()
    plt.hist(scores, bins=50)
    plt.axvline(CFG.initial_threshold, linestyle="--")
    plt.xlabel("Cosine similarity to AI-job-risk prototype")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_hist.png"), dpi=160)


# ---------------------------- Clustering Baseline ---------------------------- #

def clustering_baseline(text_emb: np.ndarray, y_true: Optional[pd.Series], df: pd.DataFrame) -> dict:
    # KMeans into 2 clusters
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    cl = km.fit_predict(text_emb)

    # Map cluster -> label using proximity to positive prototype
    tokenizer, model = get_clip_components(CFG.model_name, CFG.device)
    pos_emb = prototype_embedding(POSITIVE_PROMPTS, tokenizer, model, CFG.device)  # [1, D]
    centroids = km.cluster_centers_  # already in embedding space; normalize to match cosine
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sim_to_pos = cosine_sim(centroids, pos_emb).ravel()
    # Cluster with higher similarity -> label 1
    mapping = {np.argmax(sim_to_pos): 1, np.argmin(sim_to_pos): 0}
    cl_labels = np.vectorize(mapping.get)(cl)

    out = {
        "labels": cl_labels,
        "raw_clusters": cl,
        "silhouette": float(silhouette_score(text_emb, cl)) if len(np.unique(cl)) > 1 else float("nan"),
    }

    if y_true is not None:
        out["f1"] = float(f1_score(y_true, cl_labels))
        out["ari"] = float(adjusted_rand_score(y_true, cl))

    df["ai_job_risk_cluster"] = cl_labels
    return out

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to Python list
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)  # Convert NumPy scalar types to Python float/int
    return obj

# ---------------------------- Main ---------------------------- #

def main():
    print("Loading data …")
    df = load_df(CFG.parquet_path)
    print(df.shape, df.columns.tolist())

    print("Ensuring embeddings …")
    text_emb = ensure_embeddings(df)

    print("Scoring & threshold labeling …")
    df_scored = add_semantic_score_and_label(df, text_emb, CFG.initial_threshold)

    print("Evaluating …")
    y_true = get_ground_truth(df_scored)
    if y_true is not None:
        print("Using ground truth of shape:", y_true.shape)
    else:
        print("No ground truth available. Using histograms only; set CFG.ground_truth_col or enable weak labels.")

    evaluate_thresholds(df_scored, y_true)

    print("Clustering baseline …")
    cl_res = clustering_baseline(text_emb, y_true, df_scored)
    print("Clustering results:", json.dumps(
        {k: convert_to_serializable(v) for k, v in cl_res.items() if k != "labels"},
        indent=2
    ))

    # Compare methods if y_true available
    if y_true is not None:
        thr_f1 = f1_score(y_true, df_scored["ai_job_risk_label"].values)
        cl_f1 = cl_res.get("f1", float("nan"))
        print(f"F1 (threshold) = {thr_f1:.3f} | F1 (clustering) = {cl_f1:.3f}")

    # Save
    os.makedirs(os.path.dirname(CFG.output_path), exist_ok=True)
    df_scored.to_parquet(CFG.output_path, index=False)
    print("Saved:", CFG.output_path)


if __name__ == "__main__":
    main()
