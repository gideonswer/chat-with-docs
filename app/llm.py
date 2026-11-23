"""Provider-agnostic LLM layer.

Every provider exposes the same `complete(system, user)` method, so the rest of the
app never knows (or cares) which model is behind it. Switching from Gemini to Claude
is a one-line change in .env: LLM_PROVIDER=claude
"""
from __future__ import annotations

import logging
from typing import Protocol

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

OPENAI_COMPATIBLE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class OpenAICompatibleLLM:
    """Works for Gemini, Groq, Ollama and anything else that speaks the OpenAI API."""

    def __init__(self, provider: str, model: str, base_url: str, api_key: str):
        from openai import OpenAI

        if not api_key:
            raise LLMError(f"No API key set for provider '{provider}'")
        self.name = f"{provider}:{model}"
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=1, timeout=60)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception as e:  # rate limits, network, bad key...
            raise LLMError(f"{self.name} failed: {e}") from e
        return (resp.choices[0].message.content or "").strip()


class ClaudeLLM:
    def __init__(self, model: str, api_key: str):
        import anthropic

        if not api_key:
            raise LLMError("No API key set for provider 'claude'")
        self.name = f"claude:{model}"
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=1)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            raise LLMError(f"{self.name} failed: {e}") from e
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class FallbackLLM:
    """Try the primary model; if it errors (e.g. free-tier daily limit), use the backup."""

    def __init__(self, primary: LLM, fallback: LLM | None):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    def complete_traced(self, system: str, user: str, temperature: float = 0.0) -> tuple[str, str]:
        """Returns (text, name_of_model_that_answered). No shared state, so it's request-safe."""
        try:
            return self.primary.complete(system, user, temperature), self.primary.name
        except LLMError as e:
            if self.fallback is None:
                raise
            log.warning("Primary LLM failed, using fallback: %s", e)
            try:
                return self.fallback.complete(system, user, temperature), self.fallback.name
            except LLMError as e2:
                raise LLMError(f"Both models failed. MAIN -> {e} || BACKUP -> {e2}") from e2

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_traced(system, user, temperature)[0]


def call_llm(llm: LLM, system: str, user: str, temperature: float = 0.0) -> tuple[str, str]:
    """Call any LLM and also report which model actually produced the answer."""
    if isinstance(llm, FallbackLLM):
        return llm.complete_traced(system, user, temperature)
    return llm.complete(system, user, temperature), llm.name


def build_provider(provider: str, model: str, s: Settings) -> LLM:
    provider = provider.lower().strip()
    if provider == "gemini":
        return OpenAICompatibleLLM("gemini", model, OPENAI_COMPATIBLE_URLS["gemini"], s.gemini_api_key)
    if provider == "groq":
        return OpenAICompatibleLLM("groq", model, OPENAI_COMPATIBLE_URLS["groq"], s.groq_api_key)
    if provider == "claude":
        return ClaudeLLM(model, s.anthropic_api_key)
    if provider == "openai_compatible":
        return OpenAICompatibleLLM(
            "openai_compatible", model, s.openai_compatible_base_url, s.openai_compatible_api_key
        )
    raise LLMError(f"Unknown LLM provider '{provider}'")


_llm: LLM | None = None


def get_llm() -> LLM:
    """FastAPI dependency. Built once, reused for every request."""
    global _llm
    if _llm is None:
        s = get_settings()
        primary = build_provider(s.llm_provider, s.llm_model, s)
        fallback = None
        if s.llm_fallback_provider:
            try:
                fallback = build_provider(s.llm_fallback_provider, s.llm_fallback_model, s)
            except LLMError as e:
                log.warning("Fallback LLM disabled: %s", e)
        _llm = FallbackLLM(primary, fallback)
    return _llm