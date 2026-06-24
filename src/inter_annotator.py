"""
Inter-annotator reliability helper for TakeMeter.

Generates a 30-example CSV for a second labeler and computes Cohen's kappa
once they return it.

Usage:
  # Step 1 — generate the second-labeler CSV (30 rows, no labels)
  python src/inter_annotator.py generate

  # Step 2 — after the second labeler fills in labels, compute agreement
  python src/inter_annotator.py score --second data/second_labeler.csv
"""

import csv
import random
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

PRIMARY_CSV = Path("data/labeled_dataset.csv")
IAA_SAMPLE  = Path("data/iaa_sample.csv")   # sent to second labeler (no labels)
IAA_FILLED  = Path("data/second_labeler.csv") # returned by second labeler (with labels)

VALID_LABELS = ["analysis", "reaction", "prediction"]  # list preserves order for reproducible sampling


def generate(n: int = 30, seed: int = 42):
    """Sample n rows from the primary dataset, strip labels, save for second labeler."""
    random.seed(seed)
    with PRIMARY_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Stratified sample: 10 per label
    per_label = n // len(VALID_LABELS)
    sample = []
    for label in VALID_LABELS:
        pool = [r for r in rows if r["label"] == label]
        sample.extend(random.sample(pool, min(per_label, len(pool))))

    random.shuffle(sample)

    IAA_SAMPLE.parent.mkdir(exist_ok=True)
    with IAA_SAMPLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for row in sample:
            writer.writerow({"text": row["text"], "label": ""})

    print(f"Saved {len(sample)} rows to {IAA_SAMPLE}")
    print("Send this file to your second labeler.")
    print("They should fill in the 'label' column with: analysis, reaction, or prediction")
    print(f"Then save as {IAA_FILLED} and run: python src/inter_annotator.py score")


def score(second_path: Path):
    """Compute percentage agreement and Cohen's kappa between primary and second labeler."""
    with PRIMARY_CSV.open(encoding="utf-8") as f:
        primary_rows = {r["text"]: r["label"] for r in csv.DictReader(f)}

    with second_path.open(encoding="utf-8") as f:
        second_rows = list(csv.DictReader(f))

    matched, primary_labels, second_labels = [], [], []
    skipped = 0
    for row in second_rows:
        text  = row["text"]
        label = row.get("label", "").strip().lower()
        if label not in VALID_LABELS:
            skipped += 1
            continue
        if text not in primary_rows:
            skipped += 1
            continue
        matched.append(text)
        primary_labels.append(primary_rows[text])
        second_labels.append(label)

    n = len(matched)
    if skipped:
        print(f"⚠️  Skipped {skipped} rows (unknown label or text not found in primary CSV)")
    if n < 10:
        print(f"Only {n} matched examples — check that text values match the primary CSV exactly.")
        return

    pct_agree = sum(p == s for p, s in zip(primary_labels, second_labels)) / n
    kappa     = cohen_kappa_score(primary_labels, second_labels)

    print(f"Matched examples:     {n}")
    print(f"Percentage agreement: {pct_agree:.1%}")
    print(f"Cohen's kappa:        {kappa:.3f}")
    print()

    if kappa >= 0.80:
        print("Interpretation: strong agreement (κ ≥ 0.80)")
    elif kappa >= 0.60:
        print("Interpretation: substantial agreement (κ 0.60–0.79)")
    elif kappa >= 0.40:
        print("Interpretation: moderate agreement (κ 0.40–0.59)")
    else:
        print("Interpretation: fair/poor agreement (κ < 0.40) — label definitions may need tightening")

    disagreements = [(t, p, s) for t, p, s in zip(matched, primary_labels, second_labels) if p != s]
    print(f"\nDisagreements: {len(disagreements)} / {n}")
    for text, p, s in disagreements:
        print(f"  Primary: {p:<12} Second: {s:<12}  Text: {text[:100]}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--n", type=int, default=30)

    score_p = sub.add_parser("score")
    score_p.add_argument("--second", type=Path, default=IAA_FILLED)

    args = parser.parse_args()

    if args.command == "generate":
        generate(n=args.n)
    elif args.command == "score":
        score(args.second)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
