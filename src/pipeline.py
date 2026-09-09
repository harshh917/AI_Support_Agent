"""Single entry point: customer message in, full agent decision out."""
import yaml

from .classify import classify
from .draft_reply import draft_reply
from .escalate import decide

_cfg = yaml.safe_load(open("config.yaml"))


def run(message: str) -> dict:
    brand = _cfg["brand"]["twitter_handle"]

    classification = classify(message)
    draft = draft_reply(message, classification["intent"], brand)
    decision = decide(classification, draft)

    return {
        "message": message,
        "intent": classification["intent"],
        "intent_confidence": classification.get("confidence"),
        "reply": draft["reply"],
        "grounding_threads": draft["grounding_threads"],
        "escalate": decision["escalate"],
        "escalate_reason": decision["reason"],
    }


if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "my order hasn't arrived and it's been 2 weeks"
    result = run(msg)
    for k, v in result.items():
        print(f"{k}: {v}")
