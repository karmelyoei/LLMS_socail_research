# Reddit Collection Starter (AI Jobs & Future of Work)

This starter helps you **collect Reddit posts and comments** relevant to topics like *AI, job changes, retraining, layoffs, and future of work* using the **official Reddit Data API via PRAW**.

> ⚖️ **Compliance & ethics (important)**
> - Follow Reddit's Developer Terms & Data API Terms. Use a descriptive user-agent and authenticate with OAuth.
> - Prefer storing **IDs** and metadata; keep full text only as long as necessary for research. Rehydrate before publication if needed.
> - Respect deletions: if a submission/comment is removed or deleted on rehydration, treat it accordingly in downstream analysis.
> - Anonymize when exporting for modeling; avoid publishing personally identifiable content.
> - Rate-limit responsibly and backoff on errors; do **not** hammer endpoints.

## Quick start

1. **Create a Reddit app** at <https://www.reddit.com/prefs/apps> (choose *script* or *installed app*).
2. Put your credentials in an `.env` file at project root (see `.env.example`).  
3. Edit `src/config.yaml` to set **subreddits**, **keywords**, **date range**, and whether to **fetch comments**.
4. Install deps and run the collector:
   ```bash
   pip install -r requirements.txt
   python -m src.run_collect --config src/config.yaml --out data/raw
   ```

### Notes on auth
- PRAW handles OAuth token acquisition/refresh automatically for common flows.
- If your environment requires a refresh token flow, check PRAW docs; this starter uses the common script/installed app setup.

## What gets saved
- `data/raw/submissions.parquet` — posts (one row per submission).
- `data/raw/comments.parquet` — comments (optional; configurable limit per post).
- Files are **incrementally appended** and **deduplicated by ID**.

## Filtering strategy (phase 1)
- During collection we apply a **light keyword filter** (title/selftext).
- In the next phase, we’ll do **semantic filtering** with sentence embeddings to keep only on-topic items.

## CLI usage

```
python -m src.run_collect --config src/config.yaml --out data/raw --since 2024-01-01 --max-posts 1000
```

Arguments:
- `--config`: path to YAML config (subreddits, keywords, etc.).
- `--out`: output dir for parquet files.
- `--since`: ISO date; only include posts created at/after this date (UTC).
- `--until`: ISO date; optional end date (UTC).
- `--max-posts`: cap total posts fetched (safety).
- `--fetch-comments`: 0/1 to override config.
- `--max-comments-per-post`: cap comments per post.

## Project structure

```
reddit_collection_starter/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── run_collect.py
│   ├── collect_posts.py
│   ├── utils.py
│   └── config.yaml
└── data/
    └── raw/
```

## Next steps
After we collect, we’ll add a **semantic filtering** step (Sentence-Transformers) and then model pipelines for **topic discovery** and **sentiment/emotion** classification.



# Sentiment Analysis Script for AI Job Risk Reddit Comments
# Requirements: pip install pandas pyarrow requests groq transformers peft datasets torch scikit-learn matplotlib seaborn
# Note: You need API keys for xAI (Grok-4) and Groq (Llama-3). Set them as environment variables or replace placeholders.
# For xAI API: Get from https://x.ai/api
# For Groq API: Get from https://console.groq.com/
# Assumptions: Emotions are 'fear', 'anxiety', 'hope'. Add 'neutral' if needed.
# Few-shot examples are made up; replace with real ones if available.
# Data path: "data/processed/comments_with_ai_job_risk.parquet"
# This script runs few-shot with Grok-4 and Llama-3, uses one for pseudo-labels, fine-tunes RoBERTa and Llama-3-8B with LoRA,
# evaluates, compares, and visualizes.


