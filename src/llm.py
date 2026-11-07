"""OpenAI-compatible client; no credentials or participant inputs are bundled."""
import json
import logging
import os
import time
from functools import lru_cache

from openai import OpenAI
import config


def api_model_name(model):
    """Allow a pinned provider model ID instead of a changing API alias."""
    override = os.environ.get("DYCON_MODEL_ID")
    if override:
        return override
    if os.environ.get("DYCON_OPENROUTER") and model == "deepseek-chat":
        return "deepseek/deepseek-chat"
    return model


@lru_cache(maxsize=None)
def get_client(model):
    if os.environ.get("DYCON_OPENROUTER"):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        url = "https://openrouter.ai/api/v1"
        env_name = "OPENROUTER_API_KEY"
    elif model == "deepseek-chat":
        key, url, env_name = config.deepseek_api_key, "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    elif model == "llama3.1-8b-instruct":
        key, url, env_name = config.llama_api_key, "https://api.lambda.ai/v1", "LLAMA_API_KEY"
    else:
        key, url, env_name = config.openai_api_key, "https://api.openai.com/v1", "OPENAI_API_KEY"
    if not key:
        raise RuntimeError(f"Set {env_name} in your environment before running inference.")
    return OpenAI(api_key=key, base_url=os.environ.get("DYCON_BASE_URL", url),
                  timeout=90.0, max_retries=0)


def ask_llm(prompt, response_format, retries=8):
    """Return a validated response or fail; never substitute a fabricated score."""
    client = get_client(config.model)
    for attempt in range(retries + 1):
        try:
            kwargs = dict(model=api_model_name(config.model),
                          messages=[{"role": "user", "content": prompt}],
                          temperature=config.temperature, max_tokens=config.max_tokens)
            if config.model in ("deepseek-chat", "llama3.1-8b-instruct"):
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"})
                data = json.loads(response.choices[0].message.content)
                return response_format.model_validate(data).model_dump()
            response = client.beta.chat.completions.parse(**kwargs, response_format=response_format)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Provider returned no structured response")
            return parsed.model_dump()
        except Exception as err:
            # Provider exception messages may contain request bodies: do not log them.
            logging.warning("LLM request failed (%s), attempt %d/%d",
                            type(err).__name__, attempt + 1, retries + 1)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("LLM request failed after retries; no prediction was recorded.") from None
