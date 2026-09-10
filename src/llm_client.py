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


def _is_daily_quota_error(error: Exception) -> bool:
    """Return True only for a genuine DAILY quota exhaustion (RPD) — this will not
    clear until the quota resets (midnight Pacific Time), so there's no point
    retrying within the same run.

    Google's free-tier 429 errors distinguish which limit fired via the quota_id /
    metric name in the error body, e.g. "...PerDay..." vs "...PerMinute...". We check
    for the "day" marker specifically rather than treating every 429 as fatal —
    a per-minute (RPM) 429 is transient and should just be retried with backoff.
    """
    message = str(error).lower()
    if "429" not in message and "resource_exhausted" not in message:
        return False
    return "perday" in message.replace(" ", "").replace("_", "") or "requests per day" in message


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True for a transient 429 (RPM/TPM) that should be retried with backoff,
    as opposed to a daily quota exhaustion (see _is_daily_quota_error)."""
    message = str(error).lower()
    return ("429" in message or "resource_exhausted" in message) and not _is_daily_quota_error(error)


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

    max_attempts = 6
    for attempt in range(max_attempts):
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
            # Daily quota (RPD) genuinely won't recover until reset — stop immediately.
            if _is_daily_quota_error(e):
                raise RuntimeError(
                    "Gemini daily quota (RPD) exhausted. Resets at midnight Pacific "
                    "Time. No retry attempted — re-run later; cached results are "
                    "preserved."
                ) from e

            # Per-minute/TPM rate limit is transient — back off longer than a normal
            # error and keep retrying instead of aborting the whole run.
            if _is_rate_limit_error(e):
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        "Gemini rate limit (RPM/TPM) still failing after "
                        f"{max_attempts} attempts."
                    ) from e
                wait_time = min(65, 15 * (attempt + 1))
                print(f"Gemini rate limit hit (transient): {e}")
                print(f"Backing off {wait_time}s before retrying...")
                time.sleep(wait_time)
                continue

            if attempt == max_attempts - 1:
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

    max_attempts = 6
    for attempt in range(max_attempts):
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
            if _is_daily_quota_error(e):
                raise RuntimeError(
                    "Gemini daily quota (RPD) exhausted. Resets at midnight Pacific "
                    "Time. No retry attempted — re-run later; cached results are "
                    "preserved."
                ) from e

            if _is_rate_limit_error(e):
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        "Gemini rate limit (RPM/TPM) still failing after "
                        f"{max_attempts} attempts."
                    ) from e
                wait_time = min(65, 15 * (attempt + 1))
                print(f"Gemini rate limit hit (transient): {e}")
                print(f"Backing off {wait_time}s before retrying...")
                time.sleep(wait_time)
                continue

            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Gemini structured-output error: {e}")
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(
        "Gemini failed to return a valid JSON response."
    )
