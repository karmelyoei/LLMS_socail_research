from __future__ import annotations
import os
import time
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import yaml
import praw
from praw.models import MoreComments

from .utils import ensure_dir, keyword_match, to_parquet_append, flatten_submission, flatten_comment

UTC = timezone.utc

class RedditCollector:
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 username: str | None = None, password: str | None = None):
        # Initialize PRAW (read-only by default; PRAW will fetch tokens as needed)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username or None,
            password=password or None,
        )
        self.reddit.read_only = True if not username else False

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60),
           stop=stop_after_attempt(5),
           retry=retry_if_exception_type(Exception))
    def search_submissions(self, subreddit: str, query: str, time_filter: str = "year", limit: int = 1000):
        sub = self.reddit.subreddit(subreddit)
        # Reddit search supports 'time_filter' in {'all','day','hour','month','week','year'}
        return list(sub.search(query=query, sort="new", time_filter=time_filter, limit=limit))

    def recent_submissions(self, subreddit: str, limit: int = 500):
        sub = self.reddit.subreddit(subreddit)
        return list(sub.new(limit=limit))

    def fetch_comments(self, submission, max_comments: int | None = None) -> list[dict]:
        # Expand 'MoreComments' to fetch deeper threads (bounded by max_comments)
        submission.comments.replace_more(limit=None)
        flat = []
        count = 0
        for c in submission.comments.list():
            flat.append(flatten_comment(c))
            count += 1
            if max_comments and count >= max_comments:
                break
        return flat

def parse_date(s: str | None) -> Optional[int]:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collect(config_path: Path, out_dir: Path, since: str | None, until: str | None,
            max_posts: int | None, fetch_comments_override: Optional[bool],
            max_comments_per_post_override: Optional[int]) -> dict[str, int]:
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "ai-jobs-research-bot/0.1")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")

    cfg = load_config(config_path)
    subreddits = cfg.get("subreddits", [])
    keywords = cfg.get("keywords", [])
    cfg_since = cfg.get("since")
    cfg_until = cfg.get("until")
    fetch_comments = cfg.get("fetch_comments", True)
    max_comments_per_post = cfg.get("max_comments_per_post", 300)
    max_posts_per_subreddit = cfg.get("max_posts_per_subreddit", 5000)

    if fetch_comments_override is not None:
        fetch_comments = fetch_comments_override
    if max_comments_per_post_override is not None:
        max_comments_per_post = max_comments_per_post_override

    since_ts = parse_date(since) if since else parse_date(cfg_since)
    until_ts = parse_date(until) if until else parse_date(cfg_until)

    rc = RedditCollector(client_id, client_secret, user_agent, username, password)

    out_dir = ensure_dir(out_dir)
    sub_path = out_dir / "submissions.parquet"
    com_path = out_dir / "comments.parquet"

    totals = {"submissions": 0, "comments": 0}

    for s in subreddits:
        # Strategy: combine targeted search + recent posts; filter locally by keywords + date range.
        # Query uses a broad AI & jobs OR automation clause; adjust as needed.
        query = '(AI OR "artificial intelligence" OR automation) AND (job OR jobs OR career OR retraining OR reskilling OR upskilling OR layoff OR retirement)'
        try:
            results = rc.search_submissions(subreddit=s, query=query, time_filter="year", limit=max_posts_per_subreddit)
        except Exception:
            # Fallback to recent posts if search fails
            results = rc.recent_submissions(subreddit=s, limit=min(max_posts_per_subreddit, 1000))

        rows = []
        comment_rows = []

        for sub in results:
            created_utc = getattr(sub, "created_utc", None)
            if since_ts and created_utc and created_utc < since_ts:
                continue
            if until_ts and created_utc and created_utc > until_ts:
                continue

            # Light keyword filter on title/selftext
            title = (getattr(sub, "title", "") or "")
            selftext = (getattr(sub, "selftext", "") or "")
            text_blob = f"{title}\n{selftext}"
            if keywords and not any(kw.lower() in text_blob.lower() for kw in keywords):
                continue

            rows.append(flatten_submission(sub))

            if fetch_comments:
                try:
                    comment_rows.extend(rc.fetch_comments(sub, max_comments=max_comments_per_post))
                except Exception:
                    # Skip comments for this submission on error
                    pass

            if max_posts and len(rows) >= max_posts:
                break

        if rows:
            df = pd.DataFrame(rows)
            to_parquet_append(df, sub_path, unique_key="submission_id")
            totals["submissions"] += len(df)

        if fetch_comments and comment_rows:
            cdf = pd.DataFrame(comment_rows)
            to_parquet_append(cdf, com_path, unique_key="comment_id")
            totals["comments"] += len(cdf)

    return totals
