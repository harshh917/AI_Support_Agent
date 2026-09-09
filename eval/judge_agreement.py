"""
Validate the LLM judge against human ratings. This is a REQUIRED deliverable
("evidence of how well your judge agrees with a human") — don't skip it.

Workflow:
  1. Take a subset of judge_scores.jsonl (e.g. 30-40 examples — enough to compute a
     real agreement stat, not so many it's a second full labeling project).
  2. YOU independently score those same replies on the same rubric, blind to the
     judge's scores. Save as eval/golden_set/human_scores.jsonl (same schema, plus
     "thread_id" to join on).
  3. Run this script to compute agreement.

Usage:
    python -m eval.judge_agreement
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

JUDGE_PATH = Path("eval/golden_set/judge_scores.jsonl")
HUMAN_PATH = Path("eval/golden_set/human_scores.jsonl")

DIMS = ["grounded", "relevance", "actionability", "tone"]


def load(path):
    return {json.loads(l)["thread_id"]: json.loads(l) for l in open(path)}


def main():
    if not HUMAN_PATH.exists():
        raise SystemExit(
            f"Missing {HUMAN_PATH}. You need to hand-score a subset first — see the "
            "workflow in this file's docstring. This isn't optional; the assignment "
            "explicitly requires judge-vs-human agreement evidence."
        )

    judge = load(JUDGE_PATH)
    human = load(HUMAN_PATH)
    shared_ids = sorted(set(judge) & set(human))
    if len(shared_ids) < 20:
        print(f"WARNING: only {len(shared_ids)} shared examples — agreement stats will be noisy. "
              f"Aim for 30+.")

    print(f"Comparing {len(shared_ids)} examples judge vs. human:\n")
    for dim in DIMS:
        j_scores = [judge[i][dim] for i in shared_ids]
        h_scores = [human[i][dim] for i in shared_ids]
        # Quadratic-weighted kappa: penalizes big disagreements (1 vs 5) more than
        # small ones (3 vs 4), appropriate for an ordinal 1-5 rubric.
        kappa = cohen_kappa_score(j_scores, h_scores, weights="quadratic")
        mae = np.mean(np.abs(np.array(j_scores) - np.array(h_scores)))
        print(f"  {dim}: quadratic-weighted kappa={kappa:.2f}, MAE={mae:.2f}")

    print(
        "\nInterpretation guide: kappa > 0.6 = substantial agreement, 0.4-0.6 = moderate "
        "(usable but report it as a limitation), < 0.4 = judge isn't reliable enough to "
        "trust on its own for this dimension."
    )


if __name__ == "__main__":
    main()
