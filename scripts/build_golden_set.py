"""
Sample threads for hand-labeling into the golden evaluation set.

This script does NOT auto-label anything — it samples candidates and writes a CSV
for you to label by hand (intent, correct escalate/auto decision, and a short note
on what a good reply would ground on).

Sampling strategy:
  - Stratify by thread length so the golden set is not dominated by one type.
  - Prefer short and long multi-turn threads because the current dataset has
    very few/no single-turn threads.
  - Fill any remaining slots randomly from the unsampled threads.
  - Use a fixed seed for reproducibility.

Usage:
    python scripts/build_golden_set.py --n 200
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml


THREADS_PATH = Path("data/processed/threads.jsonl")
OUT_PATH = Path("eval/golden_set/to_label.csv")


def load_threads():
    with open(THREADS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def bucket(thread):
    n = len(thread["turns"])

    if n <= 1:
        return "single"
    elif n <= 3:
        return "short"

    return "long"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = args.n or cfg["eval"]["golden_set_size"]

    random.seed(args.seed)

    threads = load_threads()

    if n > len(threads):
        raise ValueError(
            f"Requested {n} threads, but only {len(threads)} are available."
        )

    # Create thread-length buckets.
    buckets = {
        "single": [],
        "short": [],
        "long": [],
    }

    for thread in threads:
        buckets[bucket(thread)].append(thread)

    # Target distribution:
    # 15% single, 50% short, 35% long.
    #
    # The dataset currently has no/very few single-turn threads,
    # so unavailable slots will be filled from the remaining pool.
    targets = {
        "single": int(n * 0.15),
        "short": int(n * 0.50),
        "long": int(n * 0.35),
    }

    sampled = []

    # Sample from each bucket.
    for name, items in buckets.items():
        random.shuffle(items)

        take = min(targets[name], len(items))

        sampled.extend(items[:take])

    # Fill remaining slots if some bucket did not have enough examples.
    remaining = n - len(sampled)

    if remaining > 0:
        sampled_ids = {
            thread["thread_id"]
            for thread in sampled
        }

        remaining_pool = [
            thread
            for thread in threads
            if thread["thread_id"] not in sampled_ids
        ]

        random.shuffle(remaining_pool)

        sampled.extend(remaining_pool[:remaining])

    # Final shuffle so the CSV is not grouped by thread length.
    random.shuffle(sampled)

    rows = []

    for thread in sampled:

        customer_msg = next(
            (
                turn["text"]
                for turn in thread["turns"]
                if turn["author"] == "customer"
            ),
            "",
        )

        full_thread = " | ".join(
            f"[{turn['author']}] {turn['text']}"
            for turn in thread["turns"]
        )

        rows.append(
            {
                "thread_id": thread["thread_id"],
                "bucket": bucket(thread),
                "customer_message": customer_msg,
                "full_thread": full_thread,

                # Columns to fill manually.
                "label_intent": "",
                "label_escalate": "",
                "label_escalate_reason": "",
                "label_good_reply_notes": "",
            }
        )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        f"Wrote {len(rows)} candidates to {OUT_PATH} "
        f"— label them by hand, then save as "
        f"eval/golden_set/golden.jsonl"
    )

    # Print sampling distribution for reproducibility.
    print("\nSampling distribution:")

    distribution = pd.Series(
        [bucket(thread) for thread in sampled]
    ).value_counts()

    for name in ["single", "short", "long"]:
        print(
            f"  {name}: {distribution.get(name, 0)}"
        )


if __name__ == "__main__":
    main()