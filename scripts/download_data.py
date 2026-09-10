
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

RAW_PATH = Path("data/raw/twcs.csv")
OUT_PATH = Path("data/processed/brand_subsample.csv")


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=None, help="Twitter handle, e.g. AppleSupport")
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    brand = args.brand or cfg["brand"]["twitter_handle"]
    max_threads = args.max_threads or cfg["brand"]["max_threads"]

    if not RAW_PATH.exists():
        sys.exit(
            f"Missing {RAW_PATH}. Download the dataset first — see README 'Getting the dataset'."
        )

    print(f"Loading {RAW_PATH} (this is the full ~3M row file, may take a bit)...")
    df = pd.read_csv(RAW_PATH, dtype=str)

  
    brand_rows = df[(df["author_id"] == brand) & (df["inbound"] == "False")]
    print(f"Found {len(brand_rows)} tweets authored by {brand}")

    
    related_ids = set(brand_rows["in_response_to_tweet_id"].dropna())
    related_ids |= set(brand_rows["tweet_id"])
    customer_rows = df[df["tweet_id"].isin(related_ids)]

    subsample = pd.concat([brand_rows, customer_rows]).drop_duplicates(subset="tweet_id")

    
    root_ids = brand_rows["tweet_id"].unique()[:max_threads]
    keep_ids = set(root_ids) | related_ids
    subsample = subsample[subsample["tweet_id"].isin(keep_ids)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subsample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(subsample)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
