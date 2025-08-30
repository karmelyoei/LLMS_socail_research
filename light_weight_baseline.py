import os, json, random, math
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, pipeline
)

# ---------- CONFIG ----------
EMOTIONS = ['fear', 'anxiety', 'hope']
NUM_LABELS = len(EMOTIONS)
LABEL2ID = {emo: i for i, emo in enumerate(EMOTIONS)}
ID2LABEL = {i: emo for emo, i in LABEL2ID.items()}
RANDOM_SEED = 42
OVERSAMPLE_MINORITY = False

# ---------- UTIL ----------
def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_all_seeds(RANDOM_SEED)

def normalize_label(value: str):
    if isinstance(value, str):
        v = value.lower().strip()
        if v.startswith("sentiment:"): v = v.split(":",1)[1].strip()
        if v in EMOTIONS: return v
    return "unknown"

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(path,obj): ensure_dir(os.path.dirname(path)); json.dump(obj, open(path,"w"), indent=2)

# ---------- Datasets ----------
def to_hf_datasets(train_df, test_df, tokenizer, max_len=256):
    def tok(batch): return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
    train_dataset = Dataset.from_dict({'text': train_df['body'].tolist(), 'labels': train_df['label_id'].tolist()})
    test_dataset = Dataset.from_dict({'text': test_df['body'].tolist(), 'labels': test_df['label_id'].tolist()})
    train_dataset = train_dataset.map(tok, batched=True).remove_columns('text')
    test_dataset  = test_dataset.map(tok, batched=True).remove_columns('text')
    return train_dataset, test_dataset

# ---------- Weighted Trainer ----------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, device=self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs,return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ---------- Metrics & evaluation ----------
def build_compute_metrics():
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple): preds = preds[0]
        y_pred = np.argmax(preds, axis=-1)
        y_true = eval_pred.label_ids
        report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        per_class_f1 = {f"f1_{EMOTIONS[i]}": report[EMOTIONS[i]]["f1-score"] for i in range(NUM_LABELS)}
        metrics = {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "mcc": mcc}
        metrics.update(per_class_f1)
        return metrics
    return compute_metrics

def plot_learning_curves_from_trainer(trainer, save_path_prefix):
    logs = trainer.state.log_history
    # parse
    epochs_train = [x['epoch'] for x in logs if 'loss' in x and x.get('epoch') is not None]
    train_losses = [x['loss'] for x in logs if 'loss' in x and x.get('epoch') is not None]
    eval_epochs = [x['epoch'] for x in logs if 'eval_loss' in x]
    eval_losses = [x['eval_loss'] for x in logs if 'eval_loss' in x]
    # macro f1 keys may vary
    macro_keys = [k for k in logs[0].keys() if 'macro_f1' in str(k)] if logs else []
    eval_macro = []
    if macro_keys:
        key = [k for k in logs[-1].keys() if 'macro_f1' in str(k)]
        for x in logs:
            if 'epoch' in x and any('macro_f1' in k for k in x):
                for k in x:
                    if 'macro_f1' in k:
                        eval_macro.append((x['epoch'], x[k]))
    # plotting
    ensure_dir('figs')
    if train_losses:
        plt.figure(); plt.plot(epochs_train, train_losses); plt.xlabel('epoch'); plt.ylabel('train_loss'); plt.title('Training loss'); plt.tight_layout(); plt.savefig(f"figs/{save_path_prefix}_train_loss.png", dpi=150); plt.close()
    if eval_losses:
        plt.figure(); plt.plot(eval_epochs, eval_losses); plt.xlabel('epoch'); plt.ylabel('eval_loss'); plt.title('Eval loss'); plt.tight_layout(); plt.savefig(f"figs/{save_path_prefix}_eval_loss.png", dpi=150); plt.close()
    if eval_macro:
        x = [e for e,_ in eval_macro]; y = [v for _,v in eval_macro]
        plt.figure(); plt.plot(x,y); plt.xlabel('epoch'); plt.ylabel('eval_macro_f1'); plt.title('Eval macro F1'); plt.tight_layout(); plt.savefig(f"figs/{save_path_prefix}_eval_macro_f1.png", dpi=150); plt.close()

def evaluate_and_save(test_df, preds_dict, save_prefix="eval"):
    ensure_dir("figs"); ensure_dir("outputs")
    y_true_str = test_df['pseudo_label'].tolist()
    y_true = [LABEL2ID[s] for s in y_true_str]

    results = {}
    for name, pred_str in preds_dict.items():
        y_pred = [LABEL2ID.get(s, -1) for s in pred_str]
        keep = [i for i,v in enumerate(y_pred) if v!=-1]
        y_pred_f = [y_pred[i] for i in keep]
        y_true_f = [y_true[i] for i in keep]

        rep = classification_report(y_true_f, y_pred_f, target_names=EMOTIONS, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_f, y_pred_f, labels=list(range(NUM_LABELS)))
        acc = accuracy_score(y_true_f, y_pred_f)
        bal_acc = balanced_accuracy_score(y_true_f, y_pred_f)
        macro_f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true_f, y_pred_f)

        results[name] = {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "mcc": mcc, "report": rep, "confusion_matrix": cm.tolist()}

        plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Blues"); plt.title(f"{name} CM"); plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(f"figs/{save_prefix}_{name}_cm.png", dpi=150); plt.close()
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        plt.figure(figsize=(4,3)); sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Greens"); plt.title(f"{name} CM normalized"); plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(f"figs/{save_prefix}_{name}_cm_norm.png", dpi=150); plt.close()

    save_json(os.path.join("outputs", f"{save_prefix}_summary.json"), results)
    return results

# ---------- Model training helper ----------
def train_model(base_id, model_name_short, train_df, test_df, num_epochs=5, per_device_train_batch_size=16, max_len=256):
    print(f"\n--- Fine-tuning {model_name_short} ({base_id}) ---")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForSequenceClassification.from_pretrained(
    base_id,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
    use_safetensors=True,      # <--
    trust_remote_code=True     # <--
    )
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    # optional oversampling
    if OVERSAMPLE_MINORITY:
        counts = train_df['label_id'].value_counts()
        max_n = counts.max()
        parts = []
        for lid, n in counts.items():
            cls_df = train_df[train_df['label_id']==lid]
            if n<max_n:
                extra = cls_df.sample(max_n-n, replace=True, random_state=RANDOM_SEED)
                cls_df = pd.concat([cls_df, extra], axis=0)
            parts.append(cls_df)
        train_df = pd.concat(parts, axis=0).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_dataset, test_dataset = to_hf_datasets(train_df, test_df, tokenizer, max_len=max_len)
    class_w = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_LABELS), y=train_df['label_id'])
    class_w = torch.tensor(class_w, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=f"./results_{model_name_short}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=build_compute_metrics(), class_weights=class_w,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(f"./{model_name_short}_finetuned")
    tokenizer.save_pretrained(f"./{model_name_short}_finetuned")
    # plots & logs
    plot_learning_curves_from_trainer(trainer, f"{model_name_short}_learning")
    # inference
    logits = trainer.predict(test_dataset).predictions
    if isinstance(logits, tuple): logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    pred_labels = [ID2LABEL[i] for i in preds]
    return pred_labels, trainer

# ---------- MAIN ----------
def main():
    # load and prepare
    df = pd.read_parquet("data/processed/data_train.parquet")
    if os.path.exists("outputs/bart_sentiment.csv"):
        temp = pd.read_csv("outputs/bart_sentiment.csv")
        if 'comment_id' in temp.columns and 'bart_sentiment' in temp.columns:
            df = df.merge(temp[['comment_id','bart_sentiment']], on='comment_id', how='left')
        else:
            df['bart_sentiment'] = temp['bart_sentiment']
    else:
        # fallback: few-shot with BART (may be slower) - optional
        print("No bart_sentiment.csv found. Please provide pseudo-labels or create them externally.")
        return

    df['pseudo_label'] = df['bart_sentiment'].apply(normalize_label)
    df = df[df['pseudo_label']!='unknown'].reset_index(drop=True)
    df['label_id'] = df['pseudo_label'].map(LABEL2ID)

    # quick imbalance check
    counts = df['pseudo_label'].value_counts().reindex(EMOTIONS, fill_value=0)
    print("Label counts:", counts.to_dict())

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=RANDOM_SEED)

    # TRAIN RoBERTa (your original candidate)
    roberta_base = "cardiffnlp/twitter-roberta-base-emotion"
    roberta_preds, roberta_trainer = train_model(roberta_base, "RoBERTa_Twitter", train_df.copy(), test_df.copy(), num_epochs=4, per_device_train_batch_size=16, max_len=256)

    # TRAIN DistilRoBERTa (light baseline)
    distil_base = "distilroberta-base"
    distil_preds, distil_trainer = train_model(distil_base, "DistilRoBERTa", train_df.copy(), test_df.copy(), num_epochs=4, per_device_train_batch_size=16, max_len=256)

    # Evaluate and save
    test_df['roberta_pred'] = roberta_preds
    test_df['distil_pred'] = distil_preds
    if 'bart_sentiment' in test_df.columns:
        test_df['bart_fs'] = test_df['bart_sentiment']

    methods = {"RoBERTa_FT": test_df['roberta_pred'].tolist(), "DistilRoBERTa_FT": test_df['distil_pred'].tolist()}
    if 'bart_fs' in test_df.columns:
        methods["BART_ZS"] = test_df['bart_fs'].tolist()

    results = evaluate_and_save(test_df, methods, save_prefix="reddit_ai_job_risk_light")
    # save CSVs
    ensure_dir("outputs")
    df.to_csv("outputs/sentiment_all_rows.csv", index=False)
    test_df.to_csv("outputs/test_predictions.csv", index=False)
    save_json("outputs/metrics_summary.json", results)

    print("\n=== Summary ===")
    for m,v in results.items():
        print(f"{m:20s} | acc={v['accuracy']:.3f} bal_acc={v['balanced_accuracy']:.3f} macroF1={v['macro_f1']:.3f} wF1={v['weighted_f1']:.3f} mcc={v['mcc']:.3f}")

if __name__ == "__main__":
    main()
