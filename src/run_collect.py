from __future__ import annotations
import argparse
from pathlib import Path
from .collect_posts import collect

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--since", type=str, default=None, help="ISO date (UTC), e.g., 2024-01-01")
    ap.add_argument("--until", type=str, default=None, help="ISO date (UTC)")
    ap.add_argument("--max-posts", type=int, default=None, help="Max posts across all subreddits")
    ap.add_argument("--fetch-comments", type=int, default=None, help="Override config: 1 or 0")
    ap.add_argument("--max-comments-per-post", type=int, default=None, help="Override config")
    args = ap.parse_args()

    totals = collect(
        config_path=Path(args.config),
        out_dir=Path(args.out),
        since=args.since,
        until=args.until,
        max_posts=args.max_posts,
        fetch_comments_override=bool(args.fetch_comments) if args.fetch_comments is not None else None,
        max_comments_per_post_override=args.max_comments_per_post
    )

    print("Done.")
    print(totals)

if __name__ == "__main__":
    main()
