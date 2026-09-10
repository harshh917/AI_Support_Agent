
import json
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/brand_subsample.csv")
OUT_PATH = Path("data/processed/threads.jsonl")


def build_threads(df: pd.DataFrame, brand: str) -> list[dict]:
    df = df.set_index("tweet_id", drop=False)
    by_id = df.to_dict(orient="index")

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
