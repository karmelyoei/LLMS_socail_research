import os
import time
import re
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# ---- CLIP embeddings ----
from transformers import CLIPTokenizer, CLIPModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load API keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

client_openai = OpenAI(api_key=openai_key) if openai_key else None

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",  # put on GPU if available
    torch_dtype="auto"  # FP16/FP32 depending on hardware
)

# -------------------
# Step 1: Classification with GPT/HF
# -------------------
def classify_ai_risk(text, source="openai", max_retries=3):
    """
    Label text into 'AI Job Risk Related' or 'Not AI Job Risk Related'
    using OpenAI or Hugging Face.
    """
    prompt = f"""
    Classify the following comment into one of two categories:
    AI Job Risk Related
    Not AI Job Risk Related

    Comment: "{text}"
    Answer with exactly one category.
    """

    if source == "openai" and client_openai:
        for attempt in range(max_retries):
            try:
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ OpenAI error: {e} (attempt {attempt+1})")
                time.sleep(2)

    if source == "hf" and hf_token:
        try:

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # deterministic
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Sometimes the model repeats the prompt, so strip it off
            # Keep only the part after the prompt
            answer = decoded.replace(prompt, "").strip()

            # Optional: force map to label set
            if "Not" in answer:
                return "Not AI Job Risk Related"
            else:
                return answer # fallback raw text


            # Through request didnt work
            # hf_model = "mistralai/Mistral-7B-Instruct"
            # headers = {"Authorization": f"Bearer {hf_token}"}
            # payload = {"inputs": prompt, "parameters": {"max_new_tokens": 50}}
            # response = requests.post(
            #     f"https://api-inference.huggingface.co/models/{hf_model}",
            #     headers=headers, json=payload, timeout=20
            # )
            #
            # # Debug raw response
            # if response.status_code != 200:
            #     print("❌ HF API returned error:", response.status_code, response.text)
            #     return "Not Classified"
            #
            # try:
            #     result = response.json()
            # except Exception:
            #     print("❌ Could not parse JSON. Raw text:", response.text[:500])
            #     return "Not Classified"
            #
            # if isinstance(result, list) and "generated_text" in result[0]:
            #     return result[0]["generated_text"].strip()
            # else:
            #     print("⚠️ Unexpected HF format:", result)
        except Exception as e:
            print(f"❌ Hugging Face request failed: {e}")

    return "Not Classified"


# -------------------
# Step 2: Training classifiers
# -------------------
def train_classifier(df, embedder, model_name, save_path):
    """
    Train a classifier using embeddings and Logistic Regression.
    Save both embeddings model (name) and classifier to disk.
    """
    X = embedder.encode(df["body"].tolist(), show_progress_bar=True)
    y = df["ai_job_risk"].apply(lambda x: 0 if "Not" in x else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ {model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save classifier + metadata
    joblib.dump({"clf": clf, "embedder": model_name}, save_path)
    print(f"💾 Model saved: {save_path}")
    return clf, acc


# -------------------
# Step 3: CLIP embeddings
# -------------------
def encode_clip(texts):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.cpu().numpy()


# ---------- PCA Visualization ----------
def plot_labels_data(df, label_col="ai_job_risk"):
    def clean_label(label):
        return re.sub(r'^\d+\.\s*', '', str(label)).strip()

    df['cleaned_label'] = df[label_col].apply(clean_label)

    # Count occurrences of each label
    label_counts = df['cleaned_label'].value_counts()
    print("Label Counts:")
    print(label_counts)

    # Plot bar chart of label counts
    plt.figure(figsize=(8, 4))
    label_counts.plot(kind='bar', color=['red', 'blue', 'green'])
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Distribution of Labels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("label_distribution.png")


    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(df["body"].tolist(), show_progress_bar=True)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    plt.figure(figsize=(8, 6))
    colors = {
        "AI Job Risk Related": "red",
        "Not AI Job Risk Related": "blue",
        "Not Classified": "green"
    }

    # Scatter plot with all three labels
    for label, color in colors.items():
        mask = label == df['cleaned_label']
        plt.scatter(df[mask]["x"], df[mask]["y"], c=color, label=label, alpha=0.5)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization of Comments")
    plt.legend()
    plt.savefig("small_data_classified.png")
    print("Saved the pics")


# -------------------
# Step 4: Main
# -------------------
if __name__ == "__main__":
    processed_file = "data/processed/full_data_labeled.parquet"
    partial_file = "data/processed/partial_labels.parquet"
    train_file = "data/processed/data_train.parquet"
    continue_labeling = False

    if not os.path.exists(train_file):
        if os.path.exists(processed_file):
            df = pd.read_parquet(processed_file)
            print("✅ Loaded existing dataframe")
        else:
            df = pd.read_parquet("data/processed/comments_with_topics.parquet")
            print("⚡ Loaded raw dataset")

        # Load partial labels if exists
        if os.path.exists(partial_file):
            labeled_df = pd.read_parquet(partial_file)
            labeled_df["ai_job_risk"] = labeled_df["ai_job_risk"].str.replace(r"^\d+\.\s*", "", regex=True)
            print(f"✅ Loaded partial labels: {len(labeled_df)} rows")
        else:
            labeled_df = pd.DataFrame(columns=df.columns.tolist() + ["ai_job_risk"])

        unclassified_df = labeled_df[labeled_df["ai_job_risk"] == "Not Classified"].copy()
        classified_df = labeled_df[labeled_df["ai_job_risk"] != "Not Classified"].copy()

        if len(unclassified_df) > 0:
            unclassified_df['ai_job_risk'] = unclassified_df["body"].apply(lambda x: classify_ai_risk(x, source="hf"))
            labeled_df = pd.concat([classified_df, unclassified_df], ignore_index=False)
            labeled_df.to_parquet(partial_file, index=False)
            labeled_df.to_parquet(train_file, index=False)

        if continue_labeling:
            # Sample rows to label
            unlabeled_df = df[~df.index.isin(labeled_df.index)]
            batch_size = 1000  # smaller batches to avoid hitting rate limits
            sources = ["hf"]

            for source in sources:
                samples = unlabeled_df.sample(min(batch_size, len(unlabeled_df)), random_state=42)
                for idx, row in samples.iterrows():
                    label = classify_ai_risk(row["body"], source=source)
                    row["ai_job_risk"] = label
                    labeled_df = pd.concat([labeled_df, pd.DataFrame([row])])
                    # Save after each classification to avoid losing progress
                    labeled_df.to_parquet(partial_file, index=False)
                unlabeled_df = df[~df.index.isin(labeled_df.index)]

            # Save final training data
            labeled_df.to_parquet(train_file, index=False)
    else:
        labeled_df = pd.read_parquet(train_file)


    # Plot PCA
    plot_labels_data(labeled_df)

    # ---- Train MiniLM Classifier ----
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
    _, minilm_acc = train_classifier(
        labeled_df, minilm_model, "MiniLM", "models/minilm_classifier.pkl"
    )

    # ---- Train CLIP Classifier ----
    clip_embeddings = encode_clip(labeled_df["body"].tolist())
    y = labeled_df["ai_job_risk"].apply(lambda x: 0 if "Not" in x else 1)
    X_train, X_test, y_train, y_test = train_test_split(
        clip_embeddings, y, test_size=0.2, random_state=42
    )

    clip_clf = LogisticRegression(max_iter=500)
    clip_clf.fit(X_train, y_train)

    preds = clip_clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ CLIP Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump({"clf": clip_clf}, "models/clip_classifier.pkl")
    print("💾 CLIP classifier saved.")
