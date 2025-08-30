# README — End-to-End Pipeline (Reddit → AI-Job-Risk → Emotions)

## 0) Repository Map

**Data prep & context**
- `clean_data.py` — cleans and semantically filters Reddit data (keeps AI/work-related). Saves processed parquet files.
- `topic_modeling.py` — BERTopic on filtered comments; exports per-comment topics and trends.

**Task 1: AI Job Risk labeling**
- `clustering_clip_classifiers.py` — CLIP text embeddings + (A) prototype-similarity thresholding, (B) KMeans(2) clustering baseline; writes labels, metrics, and plots.
- `fine_tune_classifers.py` — staged LLM labeling (Grok → GPT-4o-mini → Mistral-7B) to produce **final binary labels**; then trains MiniLM & CLIP logistic-regression baselines. Also plots label distribution/PCA.

**Task 2: Emotion (fear / anxiety / hope)**
- `light_weight_baseline.py` — fine-tunes **cardiffnlp/twitter-roberta-base-emotion** and **distilroberta-base**; evaluates (acc, macro-F1, MCC) and saves confusion matrices & a metrics summary.
- `llama_finetone.py` — **attempted** LLaMA-3 (8B) LoRA fine-tuning for the emotion task with 4-bit quantization; includes graceful OOM handling and guidance. *(Documented as failing on 12 GB VRAM.)*

## data structure
raw/
    submissions.parquet
    comments.parquet
processed/
    submissions_semantic.parquet
    comments_semantic.parquet
    comments_with_topics.parquet
    comments_with_ai_job_risk.parquet     # CLIP baseline output
    partial_labels.parquet                 # incremental LLM labels (if present)
    data_train.parquet                     # final labeled set for training
models/
  bertopic_model/
  minilm_classifier.pkl
  clip_classifier.pkl
figs/                                      # PR/ROC/F1-threshold & CMs, auto-created
outputs/ 
---

## 1) Environment & Setup

- **Python**: 3.11  
- **GPU (optional)**: NVIDIA RTX 4080 Laptop GPU (12 GB VRAM tested).  
- **IDE**: PyCharm on Windows 11 (Linux/macOS also OK).  
- **API keys (optional)**:  
  - `OPENAI_API_KEY` (GPT-4o-mini)  
  - `HUGGINGFACE_TOKEN` (e.g., `mistralai/Mistral-7B-Instruct-v0.3`)

### Create venv & install
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install pandas pyarrow torch torchvision torchaudio transformers \
            sentence-transformers bertopic scikit-learn matplotlib seaborn \
            tqdm python-dotenv datasets peft

```

Add .env (repo root)
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...


# 2. How to Run — Full Pipeline (in order)
Step A — Place raw exports
data/raw/submissions.parquet
data/raw/comments.parquet

Step B — Clean + Semantic filter (AI/work focus)
**python clean_data.py**


Outputs:
data/processed/submissions_semantic.parquet, comments_semantic.parquet

Step C — Topic modeling (context/trends)
**python topic_modeling.py**

Outputs:
data/processed/comments_with_topics.parquet, models/bertopic_model/, trends files

# 3. Task 1 — AI Job Risk (Binary)
C1. CLIP baselines (threshold + clustering)
python clustering_clip_classifiers.py


Outputs:

data/processed/comments_with_ai_job_risk.parquet

figs/pr_curve.png, roc_curve.png, f1_vs_threshold.png, score_hist.png

Console: AUROC, AUPRC, Best-F1, clustering F1/ARI/silhouette

C2. LLM-assisted labeling (+ small baselines)

Make sure this exists first: data/processed/data_train.parquet

**python fine_tune_classifers.py**


Outputs:

data/processed/data_train.parquet (final labels, ~747 posts)

models/minilm_classifier.pkl, models/clip_classifier.pkl

label_distribution.png, small_data_classified.png

Console: MiniLM & CLIP classification reports (~0.64 accuracy)

# 4. Task 2 — Emotion (fear / anxiety / hope)

Uses ~747 AI Job Risk posts, split 675 train / 169 test, labels encoded {fear:0, anxiety:1, hope:2}.

E1. Lightweight fine-tuning baselines

(Rename file once for convenience)

git mv "light_weight_ baseline.py" **light_weight_baseline.py**
**python light_weight_baseline.py**


Outputs:

figs/reddit_ai_job_risk_light_*_cm.png & *_cm_norm.png

outputs/metrics_summary.json, outputs/test_predictions.csv

Console: Summary table (RoBERTa_FT, DistilRoBERTa_FT, BART_ZS)

E2. LLaMA-3 LoRA attempt
**python llama_finetone.py**


Expected:

On laptop GPU → OOM failure with guidance.

On large GPU (A100/H100) → fine-tunes under llama_lora_out/ and saves learning curves.

# How to Run Locally

1. Clone the repository:
   ```bash
   git clone git@github.com:karmelyoei/LLMS_socail_research.git
   cd LLMS_socail_research
   ```
2. Create and activate a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

# 6. Hardware Notes

Tested in PyCharm (Windows 11, Python 3.11) with RTX 4080 Laptop GPU (12 GB VRAM).

LLaMA-3 LoRA fine-tuning fails on 12 GB VRAM (documented limitation).