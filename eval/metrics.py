"""
Evaluation harness for the support-agent pipeline.

Usage:
    python -m eval.metrics --golden eval/golden_set/golden.jsonl

By default, the LLM pipeline is evaluated on a small capped subset
to stay within free-tier API quotas. Cached results are reused.

Examples:
    python -m eval.metrics --golden eval/golden_set/golden.jsonl

    python -m eval.metrics --golden eval/golden_set/golden.jsonl --llm-limit 5

    python -m eval.metrics --golden eval/golden_set/golden.jsonl --llm-limit 0
"""

import argparse
import json
import os

from sklearn.metrics import classification_report

from src.classify import (
    trivial_baseline_predict,
    simple_baseline_predict,
)
from src.pipeline import run


CACHE_FILE = "eval/results_cache.jsonl"


def load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}

    cache = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            cache[item["customer_message"]] = item

    return cache


def save_cache(
    path: str,
    cache: dict[str, dict],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        for item in cache.values():
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_llm_results(
    golden: list[dict],
    limit: int,
) -> list[dict]:
    """
    Run the full LLM pipeline on a capped subset.

    Cached results are reused whenever possible.

    limit:
        0 = evaluate all examples
        N = evaluate at most N examples
    """

    cache = load_cache(CACHE_FILE)

    if limit == 0:
        selected = golden
    else:
        selected = golden[:limit]

    results = []

    total = len(selected)

    print(
        f"\nLLM evaluation subset: "
        f"{total}/{len(golden)} examples"
    )

    for i, example in enumerate(
        selected,
        start=1,
    ):
        message = example["customer_message"]

        if message in cache:
            result = cache[message]

            print(
                f"[{i}/{total}] cached"
            )

        else:
            print(
                f"[{i}/{total}] "
                f"calling Gemini..."
            )

            result = run(message)

            cache[message] = result

            save_cache(
                CACHE_FILE,
                cache,
            )

        results.append(result)

    return results


def eval_classifier(
    golden: list[dict],
    predict_fn,
    name: str,
):
    y_true = [
        g["gold_intent"]
        for g in golden
    ]

    y_pred = [
        predict_fn(
            g["customer_message"]
        )
        for g in golden
    ]

    print(
        f"\n=== {name} ==="
    )

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def eval_llm_classifier(
    golden: list[dict],
    results: list[dict],
):
    if not results:
        return

    y_true = [
        golden[i]["gold_intent"]
        for i in range(len(results))
    ]

    y_pred = [
        r["intent"]
        for r in results
    ]

    print(
        "\n=== LLM classifier "
        "(pipeline subset) ==="
    )

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def eval_escalation(
    golden: list[dict],
    results: list[dict],
):
    if not results:
        return

    correct = sum(
        golden[i]["gold_escalate"]
        == results[i]["escalate"]
        for i in range(len(results))
    )

    total = len(results)

    print(
        f"\nEscalation decision accuracy: "
        f"{correct}/{total} = "
        f"{correct / total:.2%}"
    )

    tp = sum(
        golden[i]["gold_escalate"]
        and results[i]["escalate"]
        for i in range(total)
    )

    fn = sum(
        golden[i]["gold_escalate"]
        and not results[i]["escalate"]
        for i in range(total)
    )

    fp = sum(
        not golden[i]["gold_escalate"]
        and results[i]["escalate"]
        for i in range(total)
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else float("nan")
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else float("nan")
    )

    print(
        f"  Escalation recall "
        f"(catching cases that SHOULD escalate): "
        f"{recall:.2%}"
    )

    print(
        f"  Escalation precision: "
        f"{precision:.2%}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--golden",
        required=True,
        help="Path to golden.jsonl",
    )

    parser.add_argument(
        "--llm-limit",
        type=int,
        default=10,
        help=(
            "Number of examples for LLM evaluation. "
            "0 evaluates all examples. Default: 10"
        ),
    )

    args = parser.parse_args()

    golden = load_golden(
        args.golden
    )

    print(
        f"Loaded {len(golden)} "
        f"golden examples."
    )

    # ---------------------------------------------------------
    # Baseline 1: trivial majority-class baseline
    # ---------------------------------------------------------

    eval_classifier(
        golden,
        trivial_baseline_predict,
        "Trivial baseline (majority class)",
    )

    # ---------------------------------------------------------
    # Baseline 2: TF-IDF + Logistic Regression
    # ---------------------------------------------------------

    eval_classifier(
        golden,
        simple_baseline_predict,
        "Simple baseline (TF-IDF + LogReg)",
    )

    # ---------------------------------------------------------
    # LLM + full support pipeline
    # ---------------------------------------------------------

    print(
        "\nRunning / loading cached "
        "LLM pipeline results..."
    )

    results = get_llm_results(
        golden,
        args.llm_limit,
    )

    # ---------------------------------------------------------
    # LLM classification
    # ---------------------------------------------------------

    eval_llm_classifier(
        golden,
        results,
    )

    # ---------------------------------------------------------
    # Escalation
    # ---------------------------------------------------------

    eval_escalation(
        golden,
        results,
    )

    print(
        f"\nCache saved at: "
        f"{CACHE_FILE}"
    )


if __name__ == "__main__":
    main()