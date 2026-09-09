import json
import os
import time

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))


def _is_quota_error(error: Exception) -> bool:
    """Return True for API quota/rate-limit errors that should not be retried."""
    message = str(error).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "rate limit" in message
    )


def complete(
    prompt: str,
    *,
    role: str = "draft",
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Call Gemini and return plain-text output."""

    model = _cfg["model"][role]

    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"

    for attempt in range(4):
        try:
            response = _client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            text = response.text

            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response.")

            return text

        except Exception as e:
            # Do NOT waste retries when the free-tier quota is exhausted.
            if _is_quota_error(e):
                raise RuntimeError(
                    "Gemini API quota/rate limit reached. "
                    "No retry attempted."
                ) from e

            if attempt == 3:
                raise

            wait_time = 2 ** attempt
            print(f"Gemini API temporary error: {e}")
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError("Gemini request failed.")


def complete_json(
    prompt: str,
    *,
    role: str = "classify",
    system: str | None = None,
    max_tokens: int = 512,
    response_schema: dict | None = None,
) -> dict:
    """Call Gemini with structured JSON output."""

    strict_system = (system or "") + "\nReturn only the requested JSON object."

    full_prompt = prompt

    if strict_system:
        full_prompt = f"{strict_system}\n\n{prompt}"

    if response_schema is None:
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "intent": {"type": "STRING"},
                "confidence": {"type": "NUMBER"},
                "reason": {"type": "STRING"},
            },
            "required": [
                "intent",
                "confidence",
                "reason",
            ],
        }

    for attempt in range(4):
        try:
            response = _client.models.generate_content(
                model=_cfg["model"][role],
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            raw = response.text.strip()

            if not raw:
                raise ValueError(
                    "Gemini returned an empty JSON response."
                )

            result = json.loads(raw)

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini JSON response is not an object."
                )

            return result

        except Exception as e:
            # Quota errors should fail immediately instead of retrying.
            if _is_quota_error(e):
                raise RuntimeError(
                    "Gemini API quota/rate limit reached. "
                    "No retry attempted."
                ) from e

            if attempt == 3:
                raise

            wait_time = 2 ** attempt
            print(f"Gemini structured-output error: {e}")
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(
        "Gemini failed to return a valid JSON response."
    )