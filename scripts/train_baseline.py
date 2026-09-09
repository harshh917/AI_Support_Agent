import pandas as pd
from src.classify import train_simple_baseline

INPUT = "data/processed/brand_subsample.csv"

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


def weak_label(text: str) -> str:
    t = str(text).lower()

    if any(x in t for x in [
        "keyboard", "autocorrect", "auto correct", "typing",
        "type ", "letters", "keypad"
    ]):
        return "keyboard_input"

    if any(x in t for x in [
        "battery", "charging", "charger", "charge", "drain",
        "power", "shutdown", "shut down", "won't turn on",
        "wont turn on", "battery life"
    ]):
        return "battery_power"

    if any(x in t for x in [
        "update", "ios ", "ios.", "upgrade", "install ios",
        "software update", "verification"
    ]):
        return "software_update"

    if any(x in t for x in [
        "app", "itunes", "apple music", "music", "photo",
        "photos", "calendar", "imessage", "message", "video",
        "youtube", "playback"
    ]):
        return "app_media_issue"

    if any(x in t for x in [
        "wifi", "wi-fi", "bluetooth", "cellular", "network",
        "signal", "mms", "mobile data", "internet", "service"
    ]):
        return "connectivity"

    if any(x in t for x in [
        "screen", "camera", "home button", "touch id",
        "speaker", "microphone", "broken", "crack",
        "physical", "hardware", "iphone won't", "iphone wont"
    ]):
        return "device_hardware"

    if any(x in t for x in [
        "apple id", "icloud", "password", "account",
        "security", "locked", "login", "sign in",
        "verification code", "recovery"
    ]):
        return "account_security"

    if any(x in t for x in [
        "order", "shipping", "delivery", "refund",
        "return", "payment", "purchase", "gift card",
        "bought", "buy"
    ]):
        return "purchase_order"

    if any(x in t for x in [
        "how do i", "how can i", "where do i", "can i",
        "how to", "help me", "setting", "settings",
        "enable", "disable", "turn on", "turn off"
    ]):
        return "how_to_support"

    return "other"


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    # Customer messages only
    df = df[df["inbound"] == True].copy()

    df["label"] = df["text"].fillna("").apply(weak_label)

    # Remove empty messages
    df = df[df["text"].str.strip().ne("")]

    print(f"Training examples: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    train_simple_baseline(
        df["text"].tolist(),
        df["label"].tolist()
    )

    print("\nBaseline model created:")
    print("data/processed/baseline_clf.pkl")


if __name__ == "__main__":
    main()