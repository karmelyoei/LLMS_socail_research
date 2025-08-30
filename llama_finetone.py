import os, json, random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------- CONFIG ----------
EMOTIONS = ['fear', 'anxiety', 'hope']
NUM_LABELS = len(EMOTIONS)
LABEL2ID = {emo: i for i, emo in enumerate(EMOTIONS)}
ID2LABEL = {i: emo for emo, i in LABEL2ID.items()}
RANDOM_SEED = 42

# ---------- UTILITIES ----------
def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_all_seeds(RANDOM_SEED)

def normalize_label(value: str):
    if isinstance(value, str):
        v = value.lower().strip()
        if v.startswith("sentiment:"):
            v = v.split(":",1)[1].strip()
        if v in EMOTIONS:
            return v
    return "unknown"

def to_hf_datasets(train_df, test_df, tokenizer, max_len=256):
    def tok(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
    train_dataset = Dataset.from_dict({'text': train_df['body'].tolist(), 'labels': train_df['label_id'].tolist()})
    test_dataset  = Dataset.from_dict({'text': test_df['body'].tolist(),  'labels': test_df['label_id'].tolist()})
    train_dataset = train_dataset.map(tok, batched=True).remove_columns('text')
    test_dataset  = test_dataset.map(tok, batched=True).remove_columns('text')
    return train_dataset, test_dataset

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

def build_compute_metrics():
    from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
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
    import matplotlib.pyplot as plt
    logs = trainer.state.log_history
    # Collect lists
    epochs = []
    train_losses = []
    eval_losses = []
    eval_macro_f1 = []
    for item in logs:
        if 'loss' in item:
            train_losses.append((item.get('epoch', None), item['loss']))
        if 'eval_loss' in item:
            eval_losses.append((item['epoch'], item['eval_loss']))
        # possible metric key name
        if 'eval_macro_f1' in item:
            eval_macro_f1.append((item['epoch'], item['eval_macro_f1']))
        elif 'eval_macro_f1' not in item and any(k.startswith('eval_') for k in item.keys()):
            # pick macro_f1-like if present
            for k in item:
                if k.endswith('macro_f1') or 'macro_f1' in k:
                    eval_macro_f1.append((item.get('epoch'), item[k]))
    # convert
    if train_losses:
        x = [e for e,_ in train_losses]
        y = [v for _,v in train_losses]
        plt.figure(); plt.plot(x,y); plt.xlabel('epoch'); plt.ylabel('train_loss'); plt.title('Train loss'); plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_train_loss.png", dpi=150); plt.close()
    if eval_losses:
        x = [e for e,_ in eval_losses]; y = [v for _,v in eval_losses]
        plt.figure(); plt.plot(x,y); plt.xlabel('epoch'); plt.ylabel('eval_loss'); plt.title('Eval loss'); plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_eval_loss.png", dpi=150); plt.close()
    if eval_macro_f1:
        x = [e for e,_ in eval_macro_f1]; y = [v for _,v in eval_macro_f1]
        plt.figure(); plt.plot(x,y); plt.xlabel('epoch'); plt.ylabel('eval_macro_f1'); plt.title('Eval macro F1'); plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_eval_macro_f1.png", dpi=150); plt.close()

# ---------- MAIN: attempt LLaMA LoRA ----------
def main():
    # load data
    df = pd.read_parquet("data/processed/data_train.parquet")
    if os.path.exists("outputs/bart_sentiment.csv"):
        temp = pd.read_csv("outputs/bart_sentiment.csv")
        if 'comment_id' in temp.columns and 'bart_sentiment' in temp.columns:
            df = df.merge(temp[['comment_id','bart_sentiment']], on='comment_id', how='left')
        else:
            df['bart_sentiment'] = temp['bart_sentiment']
    else:
        raise RuntimeError("Please provide bart_sentiment.csv (few-shot labels) or edit script to create pseudo-labels.")

    df['pseudo_label'] = df['bart_sentiment'].apply(normalize_label)
    df = df[df['pseudo_label']!='unknown'].reset_index(drop=True)
    df['label_id'] = df['pseudo_label'].map(LABEL2ID)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=RANDOM_SEED)

    # model setup
    base_id = "meta-llama/Meta-Llama-3-8B"  # large
    try:
        # try to load with device map + 4-bit qnt to reduce memory
        from transformers import BitsAndBytesConfig, AutoConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", llm_int8_threshold=6.0)
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Try to instantiate model with low-memory config (may still fail)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_id,
            num_labels=NUM_LABELS,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
        )
        # apply LoRA (PEFT)
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"])
        model = get_peft_model(model, lora_config)
        # enable gradient checkpointing if supported
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

        train_dataset, test_dataset = to_hf_datasets(train_df, test_df, tokenizer, max_len=256)

        class_w = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_LABELS), y=train_df['label_id'])
        class_w = torch.tensor(class_w, dtype=torch.float32).to(next(model.parameters()).device)

        args = TrainingArguments(
            output_dir="./llama_lora_out",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            fp16=True,
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
        )

        trainer = WeightedTrainer(
            model=model, args=args,
            train_dataset=train_dataset, eval_dataset=test_dataset,
            compute_metrics=build_compute_metrics(),
            class_weights=class_w,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model("./llama_lora_out")
        tokenizer.save_pretrained("./llama_lora_out")
        # plots
        plot_learning_curves_from_trainer(trainer, "llama_lora_learning")
        print("LLaMA training finished successfully.")

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print("ERROR: LLaMA training failed due to memory or environment constraints.")
        print("Exception:", str(e))
        print("\nGuidance:")
        print("- Try Colab Pro+ or an A100/H100 GPU with >=40GB VRAM.")
        print("- Alternatively, use a smaller LLM (7B or smaller) or train only adapters on CPU with offloading.")
        print("- For a quick deliverable, run lightweight_baselines.py instead (Roberta + DistilRoBERTa).")
        return

if __name__ == "__main__":
    main()
