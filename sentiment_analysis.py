

import numpy as np
import torch
import torch.nn as nn
import os
import json
from xai_sdk import Client
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from xai_sdk.chat import user, system
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# Set API key for Grok-4 (replace with your own)
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")


client = Client(api_key=XAI_API_KEY)

# Define emotion classes
EMOTIONS = ['fear', 'anxiety', 'hope']
NUM_LABELS = len(EMOTIONS)
SAVE_FILE = "data/processed/predictions_grok.json"

# Few-shot examples (prompt engineering)
# FEW_SHOT_EXAMPLES = """
# Example 1:
# Comment: I'm terrified that AI will replace all programmers soon.
# Sentiment: fear
# """
#
# # Prompt template
# PROMPT_TEMPLATE = """
# Classify this comment as fear, anxiety, or hope.
# Comment: {comment}
# Sentiment:
# """



FEW_SHOT_EXAMPLES = """
    Example 1:
    Comment: I'm terrified that AI will replace all programmers soon.
    Sentiment: fear
    """

PROMPT_TEMPLATE = """
    Classify the following comment as one of: fear, anxiety, or hope.
    Output only one label.
    {examples}
    Comment: {comment}
    Sentiment:
    """

LABELS = ['hope', 'anxiety', 'fear']

def load_data(file_path,Full_data):
    df = pd.read_parquet(file_path)
    if Full_data:
        df = df[df['ai_job_risk_label'] == 1].reset_index(drop=True)
    else:
        df['ai_job_risk_label'] = df["ai_job_risk"].apply(lambda x: 0 if "Not" in x else 1)

    texts = df['body'].tolist()
    print(f"Loaded {len(texts)} comments.")
    return df, texts


def run_few_shot_grok4(texts, comment_ids):
    # Uses API (cannot download Grok-4 locally)
    url = "https://api.x.ai/v1/chat/completions"
    all_predictions = load_existing_predictions()
    over_all_predictions = []
    for i,(text, comment_id) in enumerate(zip(texts, comment_ids)):
        #truncated_text = text[:1000] + "..." if len(text) > 1000 else text
        prompt = PROMPT_TEMPLATE.format(examples=FEW_SHOT_EXAMPLES, comment=text)

        try:
            # Create a chat session
            chat = client.chat.create(model="grok-4-0709", temperature=0)

            # Add system + user messages
            chat.append(system("You are a sentiment analyzer."))
            chat.append(user(prompt))

            # Sample a response
            response = chat.sample()
            content = response.content.strip().lower()

            if "fear" in content or "anxiety" in content or  "hope" in content:
                predictions = content
                over_all_predictions.append(content)
            else:
                predictions = "unknown"
                over_all_predictions.append("unknown")

        except Exception as e:
            print(f"Exception for comment_id {comment_id}: {str(e)}")
            predictions = 'error'
            over_all_predictions.append('error')

        # Save prediction with comment_id
        all_predictions[f'{str(comment_id)}_{i}'] = predictions
        save_predictions(all_predictions)  # Save after each ste

    return over_all_predictions


def load_or_run_grok4_few_shot(df, texts):
    prediction_file = "outputs/rok4_predictions.csv"
    if os.path.exists(prediction_file):
        print(f"Loading existing Grok-4 predictions from {prediction_file}...")
        grok4_df = pd.read_csv(prediction_file)
        if len(grok4_df) == len(df) and 'grok4_sentiment' in grok4_df.columns and 'comment_id' in grok4_df.columns:
            df = df.merge(grok4_df[['comment_id', 'grok4_sentiment']], on='comment_id', how='left')
            if df['grok4_sentiment'].isna().sum() == 0:
                print("Successfully loaded all Grok-4 predictions.")
            else:
                raise ValueError("Missing predictions in saved file.")
        else:
            raise ValueError("Mismatch in saved predictions file length or columns.")
    else:
        print("Running Grok-4 few-shot (API)...")
        comment_ids = df['comment_id'].tolist()
        df['grok4_sentiment'] = run_few_shot_grok4(texts, comment_ids)
        # Save predictions
        df[['comment_id', 'grok4_sentiment']].to_csv(prediction_file, index=False)
        print(f"Saved Grok-4 predictions to {prediction_file}")
    return df



def run_few_shot_llama3_local(texts):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    predictions = []
    for text in texts:
        prompt = PROMPT_TEMPLATE.format(examples=FEW_SHOT_EXAMPLES, comment=text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = generated_text.split("Sentiment:")[-1].strip().lower()
        predictions.append(label if label in EMOTIONS else 'unknown')
        np.save("Llama-3-70B-Instruct.npy", predictions)

    return predictions

def run_few_shot_bart(texts):
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli",
                          device=0)  # -1 = CPU

    predictions = []
    labels = ["fear", "anxiety", "hope"]
    for text in texts:
        res = classifier(text, candidate_labels=labels)
        label = res["labels"][0]
        predictions.append(label if label in EMOTIONS else 'unknown')
        np.save("bart_prediction.npy", predictions)

    return predictions

class SimpleSentimentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def run_sentiment_from_embeddings(npy_file, model_weights=None, num_classes=3):
    """
    Run sentiment classification on precomputed embeddings stored in a .npy file.

    Args:
        npy_file (str): Path to the .npy file with embeddings (N x D).
        model_weights (str, optional): Path to trained classifier weights.
        num_classes (int): Number of sentiment classes (default=2 for pos/neg).

    Returns:
        list[int]: Predicted class indices.
    """
    # Load embeddings
    embeddings = np.load(npy_file)   # shape: (N, D)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    # Initialize classifier
    model = SimpleSentimentClassifier(input_dim=embeddings.shape[1], num_classes=num_classes)

    # Load pretrained weights if available
    if model_weights:
        model.load_state_dict(torch.load(model_weights))

    # Inference
    with torch.no_grad():
        logits = model(embeddings)
        preds = torch.argmax(logits, dim=1).tolist()

    return preds


def run_sentiment_local(texts, model_id="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Run local sentiment analysis on a list of texts using Hugging Face models.

    Args:
        texts (list[str]): A list of input sentences.
        model_id (str): Hugging Face model ID (default is DistilBERT fine-tuned on SST-2).

    Returns:
        list[str]: Predicted sentiment labels for each text.
    """

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Auto GPU/CPU placement
    device = 0 if torch.cuda.is_available() else -1

    # Build sentiment pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    predictions = []
    results = sentiment_pipeline(texts)

    for res in results:
        label = res["label"].lower()
        if label in LABELS:
            predictions.append(label)
        else:
            predictions.append("unknown")

    return predictions

def load_existing_predictions():
    """Load saved predictions if they exist."""
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_predictions(predictions):
    """Save predictions dict to JSON."""
    with open(SAVE_FILE, "w") as f:
        json.dump(predictions, f)

def prepare_pseudo_labels(df):
    # Use Llama-3 few-shot as pseudo-labels for fine-tuning (assuming it's good; you can switch)
    df['pseudo_label'] = df['bart_sentiment']
    df = df[df['pseudo_label'] != 'unknown'].reset_index(drop=True)  # Drop errors
    label_to_id = {emo: idx for idx, emo in enumerate(EMOTIONS)}
    df['label_id'] = df['pseudo_label'].map(label_to_id)
    return df


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)
    train_texts = train_df['body'].tolist()
    train_labels = train_df['label_id'].tolist()
    test_texts = test_df['body'].tolist()
    test_labels = test_df['label_id'].tolist()  # Pseudo-gold for eval
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    return train_df, test_df, train_dataset, test_dataset


def fine_tune_roberta(train_dataset, test_dataset):
    model_dir = "./roberta_finetuned"
    model_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(model_path):
        print(f"Loading fine-tuned RoBERTa from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=NUM_LABELS)
    else:
        print("Initializing RoBERTa for fine-tuning...")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion",
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True  # <--- THIS FIX
        )

    def tokenize(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="epoch",
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='weighted')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    if not os.path.exists(model_path):
        print("Training RoBERTa...")
        trainer.train()
        print(f"Saving fine-tuned RoBERTa to {model_dir}...")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

    return trainer, test_dataset


def fine_tune_llama(train_dataset, test_dataset):
    model_dir = "./llama_finetuned"
    adapter_path = os.path.join(model_dir, "adapter_model.bin")
    model_id = "meta-llama/Meta-Llama-3-8B"

    if os.path.exists(adapter_path):
        print(f"Loading fine-tuned Llama-3-8B with LoRA from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(model, LoraConfig.from_pretrained(model_dir))
    else:
        print("Initializing Llama-3-8B for fine-tuning with LoRA...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)

    def tokenize(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda eval_pred: {
            "accuracy": accuracy_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1)),
            "f1": f1_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1), average='weighted')
        }
    )

    if not os.path.exists(adapter_path):
        print("Training Llama-3-8B with LoRA...")
        trainer.train()
        print(f"Saving fine-tuned Llama-3-8B with LoRA to {model_dir}...")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

    return trainer, test_dataset


def infer_and_evaluate(test_df, df, roberta_trainer, roberta_test_dataset, llama_trainer, llama_test_dataset):
    # Infer with RoBERTa
    roberta_preds = roberta_trainer.predict(roberta_test_dataset).predictions.argmax(-1)
    test_df['roberta_pred'] = [EMOTIONS[p] for p in roberta_preds]

    # Infer with Llama-LoRA
    llama_preds = llama_trainer.predict(llama_test_dataset).predictions.argmax(-1)
    test_df['llama_ft_pred'] = [EMOTIONS[p] for p in llama_preds]

    # Few-shot on test
    test_df['grok4_sentiment'] = df.loc[test_df.index, 'grok4_sentiment']
    test_df['llama3_sentiment'] = df.loc[test_df.index, 'llama3_sentiment']

    # Metrics (using pseudo-labels as ground truth)
    methods = {
        'Grok-4 Few-Shot': test_df['grok4_sentiment'],
        'Llama-3 Few-Shot': test_df['llama3_sentiment'],
        'RoBERTa FT': test_df['roberta_pred'],
        'Llama FT': test_df['llama_ft_pred']
    }
    ground_truth = test_df['pseudo_label']

    for name, preds in methods.items():
        acc = accuracy_score(ground_truth, preds)
        f1 = f1_score(ground_truth, preds, average='weighted')
        print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(ground_truth, preds, target_names=EMOTIONS))

    return methods


def visualize_results(methods):
    # 1. Distribution bar charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (name, preds) in enumerate(methods.items()):
        ax = axes[i // 2, i % 2]
        sns.countplot(x=preds, ax=ax, order=EMOTIONS)
        ax.set_title(f"{name} Emotion Distribution")
    plt.tight_layout()
    plt.savefig("emotion_distributions.png")
    print("Saved emotion_distributions.png")

    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ground_truth = test_df['pseudo_label']  # Assuming global test_df
    for i, (name, preds) in enumerate(methods.items()):
        cm = confusion_matrix(ground_truth, preds, labels=EMOTIONS)
        ax = axes[i // 2, i % 2]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    print("Saved confusion_matrices.png")

def normalize_label(value):
    """
    Clean prediction labels:
    - If 'sentiment:' prefix exists, remove it.
    - Ensure only 'fear', 'anxiety', 'hope', or 'unknown' remain.
    """
    if isinstance(value, str):
        value = value.lower().strip()
        if value.startswith("sentiment:"):
            value = value.split(":", 1)[1]  # take after 'sentiment:'
    return value


def plot_label_distribution(df, label_col="pseudo_label", save_path="label_distribution.png"):
    """
    Count and plot the distribution of labels in a dataframe.

    Args:
        df (pd.DataFrame): dataframe containing labels
        label_col (str): column name with labels (default = 'pseudo_label')
        save_path (str): file name to save the plot
    """
    # Count samples per label
    label_counts = df[label_col].value_counts().sort_index()

    print("Label counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Plot
    plt.figure(figsize=(6,4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="Set2")

    plt.title("Label Distribution", fontsize=14)
    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # all data
    #file_path = "data/processed/comments_with_ai_job_risk.parquet"
    # only 747
    file_path = "data/processed/data_train.parquet"
    Full_data = False
    FEW_SHOT_Grok = False

    df, texts = load_data(file_path,Full_data)
    print("Running Grok-4 few-shot (API)...")

    if FEW_SHOT_Grok:
        df = load_or_run_grok4_few_shot(df, texts)
    else:
        all_predictions = load_existing_predictions()

        preds_df = pd.DataFrame(
            [(k.split("_")[0], normalize_label(v)) for k, v in all_predictions.items()],
            columns=["comment_id", "sentiment_analysis_few_shot_label"]
        )

        # Ensure correct type
        preds_df["comment_id"] = preds_df["comment_id"].astype(str)

        df["comment_id"] = df["comment_id"].astype(str)

        # Merge predictions into df
        df = df.merge(preds_df, on="comment_id", how="left")

    print("Running Llama-3 few-shot (local)...")
    #df['llama3_sentiment'] = run_few_shot_llama3_local(texts)
    texts = df['body'].tolist()
    if os.path.exists("outputs/bart_sentiment.csv"):
        df = pd.read_csv("outputs/bart_sentiment.csv")
    else:
        df['bart_sentiment'] = run_few_shot_bart(texts)
        #df['local_sentiment'] = run_sentiment_local(texts)
        #run_sentiment_from_embeddings(r'C:\Users\karmel\Desktop\LLMS\reddit_collection_starter\reddit_collection_starter\data\processed\clip_text_embeddings.npy')
        df.to_csv("bart_sentiment.csv", index=False)

    df = prepare_pseudo_labels(df)
    plot_label_distribution(df, label_col="pseudo_label")
    train_df, test_df, train_dataset, test_dataset = split_data(df)

    print("Fine-tuning RoBERTa...")
    roberta_trainer, roberta_test_dataset = fine_tune_roberta(train_dataset, test_dataset)

    print("Fine-tuning Llama-3-8B with LoRA...")
    llama_trainer, llama_test_dataset = fine_tune_llama(train_dataset, test_dataset)

    methods = infer_and_evaluate(test_df, df, roberta_trainer, roberta_test_dataset, llama_trainer, llama_test_dataset)

    visualize_results(methods)

    # Save results
    df.to_csv("sentiment_results.csv", index=False)
    test_df.to_csv("test_evaluation.csv", index=False)
    print("Done! Results saved.")