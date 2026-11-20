"""The only file that talks to an AI provider:  generate(messages) -> reply text.

Chosen in .env:
  LLM_PROVIDER=gemini | openai | anthropic     (default: gemini)
  LLM_MODEL=<model name>                       (old .env files: GEMINI_MODEL still works for gemini)
  LLM_FALLBACK_MODEL=<optional, same provider> (used for an episode after all 503 retries fail)
  OPENAI_BASE_URL=<optional>                   (openai only: Groq, OpenRouter, any OpenAI-compatible API)
API keys: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.
Provider packages are imported only when that provider is used, so a missing package
only matters for the provider you actually pick.
"""
import os
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash-lite", "anthropic": "claude-opus-5-5"}  # openai: set LLM_MODEL
MODEL = (os.getenv("LLM_MODEL") or (os.getenv("GEMINI_MODEL") if PROVIDER == "gemini" else None)
         or DEFAULT_MODELS.get(PROVIDER))
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL") or None
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MAX_TOKENS = 16000          # reply length limit (Anthropic requires one)
USAGE = Counter()           # input/output tokens reported by the provider, added up

_client = None              # created on first use
_api_key = None             # set by the app's password box; kept in memory only


def api_key_name():
    """The .env name of the key for the chosen provider, e.g. 'OPENAI_API_KEY'."""
    return KEY_NAMES.get(PROVIDER, "API key")


def has_api_key():
    return bool(_api_key or os.getenv(api_key_name()))


def set_api_key(key):
    """Use this key instead of the one in .env (never written to disk)."""
    global _api_key, _client
    _api_key, _client = key or None, None


def get_client():
    """Create the provider's client once. The import happens here, not at the top of the file."""
    global _client
    if _client is None:
        if PROVIDER == "gemini":
            from google import genai
            _client = genai.Client(api_key=_api_key) if _api_key else genai.Client()
        elif PROVIDER == "openai":
            from openai import OpenAI
            _client = OpenAI(api_key=_api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        elif PROVIDER == "anthropic":
            import anthropic
            _client = anthropic.Anthropic(api_key=_api_key) if _api_key else anthropic.Anthropic()
        else:
            raise ValueError(f"LLM_PROVIDER must be gemini, openai or anthropic, not {PROVIDER!r}")
    return _client


def strip_fences(text):
    """Remove a ```json ... ``` wrapper some models put around JSON."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def generate(messages, model=None):
    """messages: [{"role": "user" or "assistant", "content": "..."}, ...]. Returns the reply text."""
    model = model or MODEL
    if not model:
        raise ValueError("Set LLM_MODEL in .env")
    client = get_client()
    if PROVIDER == "gemini":
        from google.genai import types
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in messages]
        reply = client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        usage = getattr(reply, "usage_metadata", None)
        count(usage and usage.prompt_token_count, usage and usage.candidates_token_count)
        text = reply.text
    elif PROVIDER == "openai":
        reply = client.chat.completions.create(model=model, messages=messages)
        usage = getattr(reply, "usage", None)
        count(usage and usage.prompt_tokens, usage and usage.completion_tokens)
        text = reply.choices[0].message.content
    elif PROVIDER == "anthropic":
        reply = client.messages.create(model=model, max_tokens=MAX_TOKENS, messages=messages)
        if reply.stop_reason == "refusal":
            raise RuntimeError(f"model refused the request: {getattr(reply, 'stop_details', None)}")
        count(reply.usage.input_tokens, reply.usage.output_tokens)
        text = "".join(block.text for block in reply.content if block.type == "text")
    else:
        get_client()  # raises the clear "LLM_PROVIDER must be ..." error
    return strip_fences(text)


def count(input_tokens, output_tokens):
    """Add one reply's token usage to USAGE (providers that don't report it add 0)."""
    USAGE["input"] += input_tokens or 0
    USAGE["output"] += output_tokens or 0


def is_quota_error(error):
    """Quota or credit used up: retrying won't help (Gemini 429, OpenAI, Anthropic wording)."""
    text = str(error)
    return any(s in text for s in ("RESOURCE_EXHAUSTED", "exceeded your current quota",
                                   "insufficient_quota", "credit balance is too low"))


def is_overloaded(error):
    """The model is busy (Gemini/OpenAI 503 'high demand', Anthropic 529 'overloaded')."""
    text = str(error)
    return any(s in text for s in ("503", "UNAVAILABLE", "high demand", "overloaded", "529"))
