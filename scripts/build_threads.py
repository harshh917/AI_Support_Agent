"""
Reconstruct customer<->brand conversation threads from the flat, reply-chain-linked
tweet table produced by download_data.py.

The raw data links tweets via `in_response_to_tweet_id` / `response_tweet_id`, not by
a thread/conversation id, so we walk the chain ourselves.

Output: data/processed/threads.jsonl, one JSON object per thread:
    {
        "thread_id": str,
        "turns": [{"author": "customer"|"brand", "text": str, "created_at": str}, ...],
        "resolved": bool  # heuristic — see decision_log.md for definition + caveats
    }

Usage:
    python scripts/build_threads.py
"""
import json
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/brand_subsample.csv")
OUT_PATH = Path("data/processed/threads.jsonl")


def build_threads(df: pd.DataFrame, brand: str) -> list[dict]:
    df = df.set_index("tweet_id", drop=False)
    by_id = df.to_dict(orient="index")

    # A thread root is a customer tweet with no in_response_to (start of conversation)
    # that eventually got a reply from the brand.
    threads = []
    customer_roots = df[
        (df["inbound"] == "True") & (df["in_response_to_tweet_id"].isna())
    ]

    for _, root in customer_roots.iterrows():
        turns = [{
            "author": "customer",
            "text": root["text"],
            "created_at": root["created_at"],
        }]
        # Walk forward via response_tweet_id (comma-separated list of next tweet ids)
        current = root
        seen = {root["tweet_id"]}
        while pd.notna(current.get("response_tweet_id")):
            next_ids = str(current["response_tweet_id"]).split(",")
            next_id = next_ids[0].strip()
            if next_id not in by_id or next_id in seen:
                break
            nxt = by_id[next_id]
            seen.add(next_id)
            author = "brand" if nxt["author_id"] == brand else "customer"
            turns.append({
                "author": author,
                "text": nxt["text"],
                "created_at": nxt["created_at"],
            })
            current = nxt

        if any(t["author"] == "brand" for t in turns):
            threads.append({
                "thread_id": root["tweet_id"],
                "turns": turns,
                # Heuristic: thread is "resolved" if it ends on a brand turn (customer
                # didn't reply again) AND has >= 2 turns. This is crude — see
                # reports/decision_log.md for why and what it misses.
                "resolved": turns[-1]["author"] == "brand" and len(turns) >= 2,
            })

    return threads


def main():
    df = pd.read_csv(IN_PATH, dtype=str)
    brand = df.loc[df["inbound"] == "False", "author_id"].mode().iat[0]
    threads = build_threads(df, brand)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")

    resolved_pct = sum(t["resolved"] for t in threads) / max(len(threads), 1) * 100
    print(f"Wrote {len(threads)} threads -> {OUT_PATH} ({resolved_pct:.1f}% flagged resolved)")


if __name__ == "__main__":
    main()
