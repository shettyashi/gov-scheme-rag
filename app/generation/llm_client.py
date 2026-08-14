"""
Groq API client wrapper.

Model choice: openai/gpt-oss-120b, not llama-3.3-70b-versatile. Groq
deprecated the Llama 3.x chat models (llama-3.3-70b-versatile,
llama-3.1-8b-instant) — if you're following an older tutorial or a
training-data-era suggestion, you'll hit a 400 model_decommissioned
error. gpt-oss-120b is Groq's stated migration target for
llama-3.3-70b-versatile's use case as of mid-2026. Check
https://console.groq.com/docs/deprecations before assuming any model
string still works — Groq's lineup changes fast, and it's an ongoing
maintenance item for this project, not a one-time pick.

GROQ_API_KEY must be set as an environment variable — never hardcode
it. Get a free-tier key at https://console.groq.com/keys.
"""

import os

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Loads variables from a local .env file (if present) into the environment.
# .env is gitignored — this is how GROQ_API_KEY reaches the app without
# ever being typed into code or committed to version control. If .env
# doesn't exist, this is a harmless no-op and falls back to whatever's
# already in the shell environment (e.g. set via `setx` on Windows).
load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at "
                "https://console.groq.com/keys and set it as an "
                "environment variable before running this."
            )
        _client = Groq(api_key=api_key)
    return _client


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def chat(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
          temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Single-turn chat completion. temperature=0.0 by default — for both
    the self-check guardrail and grounded generation, you want
    deterministic, conservative behavior, not creative variation. This
    matters especially for the eval harness in Phase 5: a
    non-deterministic guardrail makes your hit-rate/hallucination
    numbers non-reproducible run to run.

    Retries automatically on Groq's free-tier rate limit (429), with
    exponential backoff (1s, 2s, 4s, 8s, 16s, capped at 30s), up to 5
    attempts before giving up for real. The eval harness makes several
    calls per question in quick succession (self-check + generation +
    faithfulness check), which can outrun the free tier's per-minute
    token budget well before hitting any daily cap — this is expected
    behavior on a free-tier account, not a bug, and worth mentioning
    honestly in your project writeup as a real operational constraint
    you handled, not something to hide.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
