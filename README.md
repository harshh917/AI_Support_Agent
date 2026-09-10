 AI Support Agent

An auditable AI customer-support agent built on the Customer Support on Twitter dataset.

## What it does

1. Classifies incoming customer messages into 10 support intents.
2. Retrieves historically similar AppleSupport conversations.
3. Drafts a reply grounded in historical resolutions.
4. Decides whether to auto-handle or escalate to a human.

> **The proof is worth more than the system.**

## Architecture

```text
Customer message
       |
       v
Intent Classification (Gemini)
       |
       v
Historical Retrieval (TF-IDF + Cosine Similarity)
       |
       v
Grounded Reply Draft (Gemini)
       |
       v
Escalation Decision (Rule-based)
       |
       v
Reply + Intent + Evidence + Escalation Reason
```

## Dataset

Source: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)

Target brand: **AppleSupport**

After preprocessing:

| Data | Count |
|---|---:|
| AppleSupport authored tweets | 106,860 |
| Processed rows | 213,483 |
| Customer-authored messages | 106,623 |
| Reconstructed threads | 71,386 |

The dataset contains multi-turn conversations connected through Twitter reply IDs.

### Resolution limitation

The dataset does not provide a reliable explicit `resolved` label. For retrieval, a thread is treated as resolved when it ends on a brand-authored response. This is only a heuristic and does not prove customer satisfaction.

## Intent taxonomy

| Intent | Description |
|---|---|
| `keyboard_input` | Keyboard, autocorrect and unexpected character/input problems |
| `battery_power` | Battery drain, charging and power problems |
| `software_update` | Problems primarily related to software updates |
| `app_media_issue` | Apps, media, Photos, Music, iMessage and similar issues |
| `connectivity` | Wi-Fi, Bluetooth, cellular and network issues |
| `device_hardware` | Hardware, camera, screen, buttons and sensors |
| `account_security` | Apple ID, password, recovery and security |
| `purchase_order` | Orders, shipping, payment, gift cards and returns |
| `how_to_support` | Configuration and general how-to questions |
| `other` | Unclear or miscellaneous requests |

Ambiguous messages follow a primary-symptom rule. For example, battery drain after an update is `battery_power`, while an app failure after an update is `app_media_issue`.

## Repository structure

```text
hiver-support-agent/
├── src/
│   ├── classify.py
│   ├── retrieve.py
│   ├── llm_client.py
│   ├── draft_reply.py
│   ├── escalate.py
│   └── pipeline.py
├── scripts/
│   ├── download_data.py
│   ├── build_threads.py
│   └── build_golden_set.py
├── eval/
│   ├── metrics.py
│   ├── llm_judge.py
│   ├── judge_agreement.py
│   └── golden_set/
│       ├── golden.jsonl
│       └── README.md
├── reports/
│   ├── report.md
│   └── decision_log.md
├── data/
│   ├── raw/
│   └── processed/
├── config.yaml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Gemini API key
- Kaggle Customer Support on Twitter dataset

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment setup

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env`.

## Dataset setup

Place the downloaded Kaggle CSV at:

```text
data/raw/twcs.csv
```

Then run:

```bash
python scripts/download_data.py
python scripts/build_threads.py
```

This creates the AppleSupport processed data and:

```text
data/processed/threads.jsonl
```

## Run the pipeline

```python
from src.pipeline import run

result = run("My iPhone battery is draining very quickly")
print(result)
```

The result contains:

```text
message
intent
intent_confidence
reply
grounding_threads
escalate
escalate_reason
```

## Retrieval

Historical conversations are retrieved using TF-IDF and cosine similarity.

```yaml
retrieval:
  top_k: 5
  min_similarity: 0.35
```

Retrieved conversations are supplied to the reply-drafting stage as historical evidence. The model is instructed not to invent brand-specific policies unsupported by the evidence.

## Escalation

Escalation is intentionally rule-based:

```yaml
escalation:
  min_classifier_confidence: 0.75
  min_grounding_similarity: 0.5
  always_escalate_intents:
    - "account_security"
```

A case is escalated when:

1. the intent is `account_security`;
2. classifier confidence is below `0.75`; or
3. best historical similarity is below `0.50`.

## Evaluation

The golden evaluation set contains **200 examples**.

Run:

```bash
python -m eval.metrics --golden eval/golden_set/golden.jsonl
```

LLM pipeline results are cached in:

```text
eval/results_cache.jsonl
```

### Baseline results

| Metric | Trivial Baseline | TF-IDF + Logistic Regression |
|---|---:|---:|
| Accuracy | 43% | 46% |
| Macro F1 | 0.06 | 0.31 |
| Weighted F1 | 0.26 | 0.44 |

The trivial baseline always predicts `app_media_issue`, the majority class. Therefore, its 43% accuracy is misleading; macro F1 is more informative.

## LLM-as-judge

Run:

```bash
python -m eval.llm_judge --golden eval/golden_set/golden.jsonl
```

The judge scores:

- Groundedness
- Relevance
- Actionability
- Tone

Each dimension is scored from 1–5.

Judge/human agreement is evaluated with:

```bash
python -m eval.judge_agreement
```

using quadratic-weighted Cohen's kappa and mean absolute error.

### API quota limitation

During evaluation, the Gemini free-tier daily generation quota was exhausted. Therefore, a full 200-example LLM evaluation and judge run could not be completed without additional API quota.

No unsupported LLM performance number is reported.

## Golden set

The golden set contains 200 examples sampled using a fixed random seed and thread-length buckets:

```text
Short threads: 126
Long threads:   74
Total:         200
```

The initial labels were model-assisted and require manual review before being treated as independently hand-labelled ground truth.

## Failure modes

1. **Class imbalance** — the majority class makes accuracy misleading.
2. **Ambiguous multi-issue messages** — keyword-only classification can choose the wrong issue.
3. **Sparse minority intents** — low-frequency categories have uneven recall.
4. **Resolution heuristic errors** — ending on a brand response does not prove successful resolution.
5. **Noisy historical replies** — shortened links and conversational fragments can propagate dataset noise.

Detailed failure analysis is in `reports/report.md`.

## Headline-number caveat

The 43% majority-baseline accuracy is the most misleading headline number because it comes entirely from predicting the dominant class.

Its macro F1 is only 0.06.

Likewise, the 46% TF-IDF accuracy should not be interpreted as production accuracy: the evaluation set contains only 200 examples, is imbalanced, and its labels require review.

The number of threads marked resolved is also not a business resolution rate because resolution is inferred from conversation structure.

## Next steps

With one additional week:

1. Build a genuinely hand-labelled balanced evaluation set.
2. Compare TF-IDF retrieval with embedding retrieval.
3. Tune escalation thresholds using validation data.
4. Expand human-vs-LLM judge agreement testing.
5. Analyze errors by intent, message length, retrieval similarity and escalation outcome.

## Reproducibility

```bash
python scripts/download_data.py
python scripts/build_threads.py
python -m eval.metrics --golden eval/golden_set/golden.jsonl
python -m eval.llm_judge --golden eval/golden_set/golden.jsonl
```

## Tests

```bash
python -m pytest
```

## Configuration

Main settings are in `config.yaml`.

```yaml
brand:
  twitter_handle: "AppleSupport"
  max_threads: 20000

model:
  classify: "gemini-3.8-flash"
  draft: "gemini-3.8-flash"
  judge: "gemini-3.8-flash"

retrieval:
  top_k: 5
  min_similarity: 0.35

escalation:
  min_classifier_confidence: 0.75
  min_grounding_similarity: 0.5
  always_escalate_intents:
    - "account_security"
```

## Documentation

- `reports/report.md` — problem framing, evaluation, failure analysis and next steps
- `reports/decision_log.md` — non-obvious engineering decisions
- `eval/golden_set/README.md` — golden-set documentation

## Limitations

This is a research/prototype system, not a production deployment.

Important limitations:

- resolution is inferred rather than explicitly labelled;
- Twitter data contains noisy conversational artifacts;
- the golden set requires human verification;
- the dataset is class-imbalanced;
- LLM evaluation was constrained by free-tier API quota;
- retrieval uses TF-IDF rather than semantic embeddings;
- escalation thresholds are manually configured.

## Submission principle

The project prioritizes measurable evidence over architectural complexity.

The goal is to clearly show what was built, what was measured, what worked, what failed, what remains uncertain, and what should be improved next.
