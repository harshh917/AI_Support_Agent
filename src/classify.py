
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .llm_client import complete_json
INTENTS = [
    "keyboard_input",
    "battery_power",
    "software_update",
    "app_media_issue",
    "connectivity",
    "device_hardware",
    "account_security",
    "purchase_order",
    "how_to_support",
    "other",
]

BASELINE_MODEL_PATH = Path("data/processed/baseline_clf.pkl")


def trivial_baseline_predict(_text: str, majority_class: str = "app_media_issue") -> str:
    return majority_class

def train_simple_baseline(texts: list[str], labels: list[str]) -> None:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump((vec, clf), f)


def simple_baseline_predict(text: str) -> str:
    with open(BASELINE_MODEL_PATH, "rb") as f:
        vec, clf = pickle.load(f)
    return clf.predict(vec.transform([text]))[0]


CLASSIFY_PROMPT = """You are classifying a customer support message into exactly one intent.

Intents: {intents}

Customer message:
\"\"\"{message}\"\"\"

Return JSON: {{"intent": "<one of the intents above>", "confidence": <0-1 float>, "reason": "<one short sentence>"}}
"""


def classify(message: str) -> dict:
    prompt = CLASSIFY_PROMPT.format(intents=", ".join(INTENTS), message=message)
    result = complete_json(prompt, role="classify")
    if result.get("intent") not in INTENTS:
        result["intent"] = "other"
    return result
