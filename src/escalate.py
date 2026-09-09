"""Decide auto-handle vs escalate-to-human, with a stated reason.

Deliberately rule-based (not another LLM call) on top of signals the earlier stages
already produced — classifier confidence and retrieval grounding strength — so the
decision is auditable and cheap. See reports/decision_log.md for why this wasn't
made an LLM judgment call too.
"""
import yaml

_cfg = yaml.safe_load(open("config.yaml"))


def decide(classification: dict, draft: dict) -> dict:
    cfg = _cfg["escalation"]
    intent = classification["intent"]
    confidence = classification.get("confidence", 0.0)
    grounding = draft.get("max_grounding_similarity", 0.0)

    if intent in cfg["always_escalate_intents"]:
        return {"escalate": True, "reason": f"intent '{intent}' is always escalated by policy"}

    if confidence < cfg["min_classifier_confidence"]:
        return {
            "escalate": True,
            "reason": f"low classifier confidence ({confidence:.2f} < "
                      f"{cfg['min_classifier_confidence']})",
        }

    if grounding < cfg["min_grounding_similarity"]:
        return {
            "escalate": True,
            "reason": f"no strong historical precedent found (best similarity "
                      f"{grounding:.2f} < {cfg['min_grounding_similarity']})",
        }

    return {
        "escalate": False,
        "reason": f"confident intent ({confidence:.2f}) with strong precedent "
                  f"({grounding:.2f}); auto-handling",
    }
