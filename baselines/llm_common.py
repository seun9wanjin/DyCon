"""OpenAI-compatible client for the released comparison methods.

Credentials are read exclusively from environment variables. Model access and
any transmission of local data are controlled by the person running the code.
"""
import json
import logging
import os
import time

MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
_KEY_ENV = os.environ.get("LLM_KEY_ENV", "DEEPSEEK_API_KEY")
TEMPERATURE = 0
MAX_TOKENS = 1000
_client = None


def get_api_key():
    key = os.environ.get(_KEY_ENV, "")
    if not key:
        raise RuntimeError("Set the API-key environment variable selected by LLM_KEY_ENV "
                           "(default: DEEPSEEK_API_KEY)")
    return key


def client():
    global _client
    if _client is None:
        key = get_api_key()
        from openai import OpenAI
        _client = OpenAI(api_key=key, base_url=_BASE_URL)
    return _client


def ask_llm(prompt, retries=5):
    """Return a parsed JSON object, or None after exhausted API/format retries."""
    api = client()  # Missing credentials fail before retries and before any request.
    for attempt in range(retries + 1):
        try:
            response = api.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("No JSON object in response")
            result = json.loads(content[start:end + 1])
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except Exception as error:
            # Exception messages can contain request/response bodies. Do not log them.
            logging.warning("LLM request failed (attempt %d/%d; %s)",
                            attempt + 1, retries + 1, type(error).__name__)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    return None
