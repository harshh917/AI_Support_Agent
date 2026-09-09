# Report — AppleSupport AI Support Agent

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