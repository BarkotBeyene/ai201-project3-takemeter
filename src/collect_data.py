"""
Manual data collection helper for TakeMeter.

Usage:
  1. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in env or .env
  2. Run: python src/collect_data.py
  3. Output: data/raw_posts.csv  (text, subreddit, permalink — unlabeled)

Then open data/raw_posts.csv, add a 'label' column, and annotate manually
(optionally assisted by src/prelabel.py).
"""

import os
import csv
import time
import praw
from pathlib import Path

LABEL_MAP = {"analysis": 0, "reaction": 1, "prediction": 2}

TARGETS = [
    # (subreddit, sort, time_filter, limit)
    ("television", "top", "month", 150),
    ("television", "top", "year", 100),
]

# Episode-discussion and prediction thread search queries
SEARCH_QUERIES = [
    "episode discussion",
    "finale prediction",
    "season prediction",
    "theory",
]

OUTPUT_PATH = Path("data/raw_posts.csv")


def get_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "takemeter/1.0"),
        read_only=True,
    )


def collect_posts(reddit: praw.Reddit) -> list[dict]:
    seen = set()
    rows = []

    for subreddit_name, sort, time_filter, limit in TARGETS:
        sub = reddit.subreddit(subreddit_name)
        listing = getattr(sub, sort)(time_filter=time_filter, limit=limit)
        for post in listing:
            if post.id in seen or not post.selftext or post.selftext == "[removed]":
                continue
            seen.add(post.id)
            rows.append({
                "id": post.id,
                "text": post.title + "\n\n" + post.selftext,
                "subreddit": subreddit_name,
                "permalink": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "label": "",
                "pre_labeled": "no",
                "notes": "",
            })
        time.sleep(1)

    for query in SEARCH_QUERIES:
        for post in reddit.subreddit("television").search(query, limit=50, time_filter="month"):
            if post.id in seen or not post.selftext or post.selftext == "[removed]":
                continue
            seen.add(post.id)
            rows.append({
                "id": post.id,
                "text": post.title + "\n\n" + post.selftext,
                "subreddit": "television",
                "permalink": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "label": "",
                "pre_labeled": "no",
                "notes": "",
            })
        time.sleep(1)

    return rows


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    reddit = get_reddit()
    rows = collect_posts(reddit)
    fieldnames = ["id", "text", "subreddit", "permalink", "score", "label", "pre_labeled", "notes"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} posts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
