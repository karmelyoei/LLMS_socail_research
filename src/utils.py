import os
import re
import time
import json
import pandas as pd
from pathlib import Path
from typing import Iterable, Dict, Any, List

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def keyword_match(text: str, keywords: list[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return True
    return False

def to_parquet_append(df: pd.DataFrame, dest: Path, unique_key: str) -> None:
    """Append-deduplicate by unique_key. Creates file if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            old = pd.read_parquet(dest)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=[unique_key])
        except Exception:
            # If file is corrupt or schema drift, write a new file with timestamp
            ts = int(time.time())
            dest = dest.with_name(dest.stem + f".{ts}.parquet")
    df.to_parquet(dest, index=False)

def clean_markdown(text: str | None) -> str | None:
    if text is None:
        return None
    # Very light cleanup; preserve content for downstream NLP
    return text.replace("\r\n", "\n").strip()

def coerce_int(x):
    try:
        return int(x)
    except Exception:
        return None

def flatten_comment(comment) -> dict[str, Any]:
    return {
        "comment_id": comment.id,
        "submission_id": getattr(comment, "link_id", "").replace("t3_", "") if hasattr(comment, "link_id") else None,
        "parent_id": getattr(comment, "parent_id", None),
        "subreddit": str(getattr(comment, "subreddit", "")),
        "author": str(getattr(comment, "author", "")) if getattr(comment, "author", None) else None,
        "created_utc": getattr(comment, "created_utc", None),
        "score": getattr(comment, "score", None),
        "body": clean_markdown(getattr(comment, "body", None)),
        "is_submitter": getattr(comment, "is_submitter", None),
        "permalink": f"https://www.reddit.com{getattr(comment, 'permalink', '')}" if hasattr(comment, "permalink") else None,
    }

def flatten_submission(sub) -> dict[str, Any]:
    return {
        "submission_id": sub.id,
        "subreddit": str(getattr(sub, "subreddit", "")),
        "author": str(getattr(sub, "author", "")) if getattr(sub, "author", None) else None,
        "title": clean_markdown(getattr(sub, "title", None)),
        "selftext": clean_markdown(getattr(sub, "selftext", None)),
        "url": getattr(sub, "url", None),
        "permalink": f"https://www.reddit.com{getattr(sub, 'permalink', '')}",
        "created_utc": getattr(sub, "created_utc", None),
        "num_comments": getattr(sub, "num_comments", None),
        "score": getattr(sub, "score", None),
        "over_18": getattr(sub, "over_18", None),
        "flair": getattr(sub, "link_flair_text", None),
        "upvote_ratio": getattr(sub, "upvote_ratio", None),
    }
