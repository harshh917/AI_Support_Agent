import src.classify as classify_module
from src.classify import classify


def test_classify_returns_known_intent(monkeypatch):
    """Classifier unit test should not depend on a live Gemini API call."""

    def fake_complete_json(*args, **kwargs):
        return {
            "intent": "purchase_order",
            "confidence": 0.95,
            "reason": "The customer is asking about an order that has not shipped.",
        }

    monkeypatch.setattr(
        classify_module,
        "complete_json",
        fake_complete_json,
    )

    result = classify(
        "my order still hasn't shipped after 2 weeks"
    )

    assert result["intent"] == "purchase_order"
    assert 0 <= result["confidence"] <= 1
    assert result["reason"]


def test_trivial_baseline_returns_valid_intent():
    from src.classify import trivial_baseline_predict

    result = trivial_baseline_predict("anything")

    assert isinstance(result, str)
    assert result == "app_media_issue"


def test_simple_baseline_returns_valid_intent():
    from src.classify import simple_baseline_predict

    result = simple_baseline_predict(
        "my iphone battery is draining very quickly"
    )

    assert isinstance(result, str)