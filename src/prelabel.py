"""
AI-assisted pre-labeling for TakeMeter.

Reads data/raw_posts.csv, sends unlabeled rows to Groq in batches,
writes predicted labels back with pre_labeled=yes.

You MUST review and correct every pre-labeled row before using the CSV
for training.

Usage:
  GROQ_API_KEY=... python src/prelabel.py [--batch-size 30]
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path
from groq import Groq

INPUT_PATH = Path("data/raw_posts.csv")
OUTPUT_PATH = Path("data/prelabeled_posts.csv")

LABEL_DEFINITIONS = """
Label definitions for r/television discourse classification:

- analysis: A post that evaluates a show's writing, direction, performances, or themes
  using specific, verifiable references to the show's text (scenes, dialogue, characters,
  structure) — not just a verdict.

- reaction: An immediate emotional or evaluative response to a specific episode or moment,
  with little to no supporting reasoning — the post is expressing a feeling or verdict
  rather than making an argument.

- prediction: A speculative claim about what will happen in a future episode or season,
  stated as a forecast — the primary content is the prediction itself, not evaluation of
  something that already aired.

Edge case rule: if a post mixes prediction and analysis, ask "could you remove the
speculative sentence and still have a complete evaluative post?" If yes → analysis.
If the speculation is the point → prediction.
"""

SYSTEM_PROMPT = f"""You are a dataset annotator. Your job is to classify Reddit posts from r/television
into exactly one of three labels: analysis, reaction, or prediction.

{LABEL_DEFINITIONS}

You will receive a JSON array of posts. For each post, output ONLY a JSON array of objects
with "id" and "label" fields. No explanation, no markdown. Example output:
[{{"id": "abc123", "label": "reaction"}}, {{"id": "def456", "label": "analysis"}}]
"""


def prelabel_batch(client: Groq, batch: list[dict]) -> dict[str, str]:
    payload = [{"id": row["id"], "text": row["text"][:600]} for row in batch]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    try:
        results = json.loads(raw)
        return {item["id"]: item["label"] for item in results}
    except Exception:
        print(f"  WARNING: could not parse batch response: {raw[:200]}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable.")

    client = Groq(api_key=api_key)

    with INPUT_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unlabeled = [r for r in rows if not r.get("label")]
    print(f"Unlabeled rows: {len(unlabeled)} / {len(rows)}")

    batch_size = args.batch_size
    updates: dict[str, str] = {}
    for i in range(0, len(unlabeled), batch_size):
        batch = unlabeled[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} posts...")
        result = prelabel_batch(client, batch)
        updates.update(result)
        time.sleep(1)

    valid_labels = {"analysis", "reaction", "prediction"}
    for row in rows:
        if row["id"] in updates and not row.get("label"):
            predicted = updates[row["id"]]
            if predicted in valid_labels:
                row["label"] = predicted
                row["pre_labeled"] = "yes"

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fieldnames = rows[0].keys()
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labeled_count = sum(1 for r in rows if r.get("label"))
    pre_labeled_count = sum(1 for r in rows if r.get("pre_labeled") == "yes")
    print(f"Saved {OUTPUT_PATH}: {labeled_count} labeled ({pre_labeled_count} pre-labeled, needs review)")


if __name__ == "__main__":
    main()
