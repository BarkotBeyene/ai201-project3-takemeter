"""
Zero-shot Groq baseline for TakeMeter.

Reads data/labeled_dataset.csv, classifies the test split using
llama-3.3-70b-versatile with no fine-tuning, and writes:
  - data/baseline_results.json

Usage:
  GROQ_API_KEY=... python src/baseline.py
"""

import os
import csv
import json
import time
from pathlib import Path
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = Path("data/labeled_dataset.csv")
OUTPUT_PATH = Path("data/baseline_results.json")

LABEL_DEFINITIONS = """
You are classifying Reddit posts from r/television into exactly one of three labels.

Labels:
- analysis: A post that evaluates a show's writing, direction, performances, or themes
  using specific, verifiable references to the show's text (scenes, dialogue, characters,
  structure) rather than just delivering a verdict.

- reaction: An immediate emotional or evaluative response to a specific episode or moment,
  with little to no supporting reasoning. The post expresses a feeling or verdict rather
  than making an argument.

- prediction: A speculative claim about what will happen in a future episode or season,
  stated as a forecast. The primary content is the prediction itself, not evaluation of
  something that already aired.

Edge case rule: if a post mixes prediction and analysis, ask whether removing the
speculative sentence would leave a complete evaluative post. If yes → analysis.
If the speculation is the point → prediction.

Respond with ONLY the label name — one of: analysis, reaction, prediction.
No punctuation, no explanation, no other text.
"""


def classify_one(client: Groq, text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": LABEL_DEFINITIONS},
            {"role": "user", "content": text[:800]},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip().lower()
    valid = {"analysis", "reaction", "prediction"}
    for label in valid:
        if label in raw:
            return label
    return "unparseable"


def load_test_split(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    return X_test, y_test


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable.")

    client = Groq(api_key=api_key)
    X_test, y_test = load_test_split(DATA_PATH)
    print(f"Test set size: {len(X_test)}")

    predictions = []
    unparseable = 0
    for i, text in enumerate(X_test):
        pred = classify_one(client, text)
        if pred == "unparseable":
            unparseable += 1
            pred = "reaction"  # fallback to majority class
        predictions.append(pred)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(X_test)}...")
        time.sleep(0.3)  # respect rate limits

    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions, labels=["analysis", "reaction", "prediction"])

    print(f"\nBaseline overall accuracy: {acc:.3f}")
    print(f"Unparseable responses: {unparseable}/{len(X_test)}")
    print("\nPer-class metrics:")
    for label in ["analysis", "reaction", "prediction"]:
        m = report[label]
        print(f"  {label:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("             analysis  reaction  prediction")
    for i, row_label in enumerate(["analysis", "reaction", "prediction"]):
        print(f"  {row_label:12s}  {cm[i][0]:8d}  {cm[i][1]:8d}  {cm[i][2]:10d}")

    results = {
        "model": "llama-3.3-70b-versatile (zero-shot)",
        "accuracy": acc,
        "unparseable": unparseable,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "labels": ["analysis", "reaction", "prediction"],
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
