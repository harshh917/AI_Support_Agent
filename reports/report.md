# Report — AI Support Agent

## 1. Problem framing

The goal is to build an AI support agent for AppleSupport that can:

1. classify incoming customer messages into a small, data-derived set of support intents;
2. draft a concise reply grounded in how AppleSupport historically handled similar conversations; and
3. decide whether the case can be auto-handled or should be escalated to a human.

The system is intentionally designed as a conservative support assistant rather than a fully autonomous agent. The key principle is that the model should rely on historical support evidence and avoid inventing brand-specific policies.

### What "good" means

A good system should:

- correctly identify the customer's primary issue;
- retrieve relevant historical conversations;
- produce a useful and concise reply;
- avoid unsupported policy claims;
- escalate uncertain or higher-risk cases.

### What I chose not to build

I did not build:

- an embedding/vector database;
- a complex multi-agent architecture;
- a production web UI;
- automatic account actions;
- a payment or order-management integration.

The focus is on an auditable prototype where the evaluation evidence is more important than system complexity.

---

## 2. Data and intent taxonomy

The primary dataset is the Kaggle Customer Support on Twitter dataset.

The selected brand is **AppleSupport**.

After brand filtering:

- AppleSupport authored tweets: **106,860**
- processed rows: **213,483**
- customer-authored messages: **106,623**
- reconstructed conversation threads: **71,386**

Threads were reconstructed using tweet reply relationships rather than time-window clustering.

### Important data limitation

The dataset does not explicitly contain a reliable "resolved" label.

For retrieval, a thread is therefore considered resolved when the conversation ends on a brand-authored response. This is a heuristic and should not be interpreted as true customer satisfaction or successful resolution.

### Final intent taxonomy

The taxonomy was derived by inspecting AppleSupport customer messages.

| Intent | Description |
|---|---|
| `keyboard_input` | Keyboard, autocorrect and unexpected character/input issues |
| `battery_power` | Battery drain, charging and power problems |
| `software_update` | Problems primarily related to software updates |
| `app_media_issue` | Apps, media, Photos, Music, iMessage and similar application issues |
| `connectivity` | Wi-Fi, Bluetooth, cellular and network connectivity |
| `device_hardware` | Physical/device hardware, camera, screen, buttons and sensors |
| `account_security` | Apple ID, password, recovery and security-related issues |
| `purchase_order` | Orders, shipping, payment, gift cards and returns |
| `how_to_support` | Configuration and general how-to support |
| `other` | Unclear or miscellaneous requests |

For ambiguous messages, the classifier follows a primary-symptom rule. For example, battery drain after an iOS update is classified as `battery_power` rather than `software_update`.

---

## 3. System design

The pipeline is:

```text
Customer message
       |
       v
Intent classification
       |
       v
Historical retrieval
       |
       v
Grounded reply drafting
       |
       v
Escalation decision
       |
       v
Reply + intent + evidence + escalation reason
```

Retrieval uses TF-IDF cosine similarity over resolved threads rather than embeddings — chosen for transparency and auditability over marginal quality gains (see decision log #9). Escalation is rule-based on classifier confidence and retrieval grounding strength rather than another LLM call, so the decision stays cheap and inspectable (decision log #11).

---

## 4. Golden evaluation set

200 examples, sampled with a fixed random seed stratified by thread length (short/long buckets: 126/74). Intent distribution is imbalanced by construction — it reflects the real distribution of AppleSupport traffic rather than being artificially balanced:

| Intent | Count |
|---|---:|
| app_media_issue | 86 |
| keyboard_input | 21 |
| battery_power | 18 |
| software_update | 16 |
| connectivity | 12 |
| device_hardware | 13 |
| how_to_support | 13 |
| purchase_order | 9 |
| account_security | 7 |
| other | 5 |

Escalation is rare in this set: 16/200 (8%) are labeled escalate=true. That imbalance matters for interpreting escalation recall/precision later — with only 16 positive examples, a couple of misses move the recall number a lot.

**Labeling provenance**: initial labels were model-assisted, not independently hand-labeled as the assignment specifies. A stratified 26-example sample (13%) was subsequently reviewed by hand — see decision log #16 for the exact procedure and outcomes (1 relabeled, 25 confirmed, 3 borderline cases deliberately left as-is with reasoning recorded). The remaining 174 examples are unreviewed. Any metric below should be read with that caveat: it reflects agreement with a partially-verified reference, not a fully independent gold standard.

---

## 5. Results vs. baselines

| Metric | Trivial baseline (majority class) | Simple baseline (TF-IDF + LogReg) | LLM pipeline |
|---|---:|---:|---:|
| Accuracy | 43% | 46% | *not yet measured* |
| Macro F1 | 0.06 | 0.31 | *not yet measured* |
| Weighted F1 | 0.26 | 0.44 | *not yet measured* |
| Escalation recall | — | — | *not yet measured* |
| Escalation precision | — | — | *not yet measured* |
| Judge: grounded/relevance/actionability/tone | — | — | *not yet measured* |
| Judge-human agreement (kappa) | — | — | *not yet measured* |

**The LLM pipeline's own results are the missing piece.** Evaluation was blocked mid-run by exhausted Gemini free-tier quota. The trivial-vs-simple baseline comparison above is real and reproducible, but on its own it does not answer the assignment's actual question — whether the built agent is good enough to trust. That requires running `eval/metrics.py`, `eval/llm_judge.py`, and `eval/judge_agreement.py` against the LLM pipeline itself, which has not yet completed.

---

## 6. Failure analysis (real examples)

1. **Class imbalance inflates naive accuracy.** The trivial baseline's 43% accuracy comes entirely from always predicting `app_media_issue`; its macro F1 of 0.06 shows it has learned nothing. Any headline "accuracy" number for this dataset needs macro F1 alongside it.
2. **Intent-boundary ambiguity at the edges.** E.g. "can you use cellular data when reinstalling offloaded apps" was initially labeled `connectivity` but is really a how-it-works question (`how_to_support`) — caught during human review, see decision log #16. This kind of boundary error is likely systematic wherever a message mentions a connectivity *keyword* without describing a connectivity *complaint*.
3. **Warranty/liability questions get bundled into device-symptom intents.** "If an update made my phone faulty, do I pay for repair?" sits between `device_hardware` and `purchase_order` — kept as `device_hardware` on review, but the classifier's few-shot prompt doesn't currently distinguish "device is broken" from "who pays to fix it," which could matter for downstream routing even if intent-level accuracy looks fine.
4. **Resolution heuristic is a proxy, not ground truth.** A thread ending on a brand reply is marked "resolved" for retrieval purposes; a thread where the customer simply gave up looks identical to one where the issue was actually fixed. This directly affects what "grounded" means for drafted replies pulled from these threads.
5. **Sparse minority intents (`other`, `account_security`, `purchase_order`) have unmeasured recall.** With single-digit counts in some buckets, per-intent recall estimates from this golden set will be noisy regardless of pipeline quality — a few examples either way swing the number substantially.

---

## 7. What is misleading about my headline number?

The 43%/46% baseline accuracy figures are the most quotable numbers in this report and also the most likely to be misread:
- They describe two baselines' agreement with a *partially human-reviewed* label set (13% verified), not ground truth.
- They say nothing about the actual LLM pipeline, which has no measured results yet — there is currently no evidence, positive or negative, that the built agent is trustworthy.
- The golden set's intent distribution mirrors real traffic (good), but its escalation distribution is thin (16 positives), so any escalation precision/recall computed later should be read as a rough signal, not a precise rate.
- The "resolved" flag used for retrieval grounding is a structural heuristic (thread ends on a brand turn), not a resolution outcome — so "grounded in a resolved thread" is weaker evidence of correctness than it sounds.

---

## 8. What I'd do next with one more week

1. Get the LLM pipeline evaluation actually run end-to-end (new API quota or a smaller subsample) and fill in the missing row of the results table above — this is the single highest-priority gap.
2. Hand-score a genuine judge-vs-human agreement set (~30-40 examples) rather than leaving the harness unpopulated.
3. Extend human review of the golden set beyond the current 26/200 (13%) toward full coverage, or at minimum a larger random supplement to the current targeted sample.
4. Compare TF-IDF retrieval against embedding-based retrieval on the same golden set to see whether grounding quality — not just intent classification — improves.
5. Break down error rates by intent, thread length, and retrieval similarity to see whether failures cluster (e.g., do low-similarity-grounding replies get worse judge scores, as the escalation design assumes?).

---

## 9. Cited / borrowed

- Dataset: Kaggle "Customer Support on Twitter" (`thoughtvector/customer-support-on-twitter`).
- iOS 11.1 autocorrect bug background (used to explain a golden-set data artifact, not copied into any code or prompt): public reporting from Emojipedia and contemporary tech press, November 2017.
- No external code, prompts, or architecture were copied from a specific source; general LLM-pipeline patterns (classify → retrieve → generate → decide) are a standard structure, not attributed to one origin.
