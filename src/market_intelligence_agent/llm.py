"""Model access for the two structured calls this agent makes: a plan and a brief.

Both providers are reached through one façade, `LLMClient`, so the planner and the
synthesiser never learn which model is behind them. Everything returns a validated
Pydantic instance - no free-text parsing happens anywhere else in the pipeline.

Provider selection is by configuration, defaulting to whichever credential exists. With
no credential at all the client reports itself unavailable and the pipeline degrades to
its heuristic plan and extractive brief rather than failing.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when no usable credential is configured for the selected provider."""


class LLMError(RuntimeError):
    """Raised when a model call fails or returns something unparseable."""


class LLMBackend(ABC):
    """One provider. Implementations own their SDK and translate its errors."""

    name: str = "base"

    @abstractmethod
    async def structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        effort: str,
        max_tokens: int,
        timeout: float | None,
    ) -> T:
        """Return a validated instance of `schema`, or raise LLMError."""

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- anthropic


class AnthropicBackend(LLMBackend):
    """Claude via the Anthropic SDK, using the structured-output parse helper."""

    name = "anthropic"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    async def structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        effort: str,
        max_tokens: int,
        timeout: float | None,
    ) -> T:
        import anthropic

        client = self._ensure_client()
        kwargs: dict = {
            "model": self._settings.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": schema,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = await client.messages.parse(**kwargs, output_config={"effort": effort})
        except TypeError:
            # Older SDKs reject output_config on the parse helper; effort is optional.
            response = await client.messages.parse(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic returned {exc.status_code}: {exc.message}") from exc
        except (anthropic.APIConnectionError, TimeoutError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"Anthropic returned no parseable {schema.__name__}.")
        return parsed

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()


# --------------------------------------------------------------------------- gemini


class GeminiBackend(LLMBackend):
    """Gemini via the google-genai SDK.

    Structured output is native: passing a Pydantic model as `response_schema` with a
    JSON mime type makes the SDK validate the response into that model, which is the
    same contract the Anthropic path provides.
    """

    name = "gemini"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        effort: str,
        max_tokens: int,
        timeout: float | None,
    ) -> T:
        from google.genai import errors as genai_errors
        from google.genai import types

        client = self._ensure_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=max_tokens,
        )

        try:
            call = client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=user,
                config=config,
            )
            response = await (asyncio.wait_for(call, timeout) if timeout else call)
        except TimeoutError as exc:
            raise LLMError(f"Gemini timed out after {timeout}s") from exc
        except genai_errors.APIError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise LLMError(f"Gemini returned no parseable {schema.__name__}.")
        if not isinstance(parsed, schema):
            # The SDK hands back the validated model; anything else means the response
            # did not match the schema and must not be trusted downstream.
            raise LLMError(
                f"Gemini returned {type(parsed).__name__}, expected {schema.__name__}."
            )
        return parsed


# --------------------------------------------------------------------------- façade


def resolve_provider(settings: Settings) -> str | None:
    """Which provider this configuration can actually use, or None.

    "auto" prefers Anthropic when both keys are present, purely because its structured
    output path here is the older and better-exercised of the two.
    """
    choice = (settings.llm_provider or "auto").lower()
    if choice == "anthropic":
        return "anthropic" if settings.anthropic_api_key else None
    if choice == "gemini":
        return "gemini" if settings.gemini_api_key else None
    if choice == "none":
        return None
    if choice != "auto":
        raise LLMUnavailable(f"Unknown llm_provider: {settings.llm_provider!r}")
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.gemini_api_key:
        return "gemini"
    return None


class LLMClient:
    """Provider-agnostic entry point used by the planner and the synthesiser."""

    def __init__(
        self,
        settings: Settings,
        client: object | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._injected_client = client
        self._provider = resolve_provider(settings) if backend is None else backend.name

    @property
    def available(self) -> bool:
        """True when a provider is selected and has a credential (or was injected)."""
        return self._backend is not None or self._provider is not None

    @property
    def provider(self) -> str | None:
        return self._provider

    def _ensure_backend(self) -> LLMBackend:
        if self._backend is not None:
            return self._backend
        if self._provider == "anthropic":
            self._backend = AnthropicBackend(self._settings, self._injected_client)
        elif self._provider == "gemini":
            self._backend = GeminiBackend(self._settings, self._injected_client)
        else:
            raise LLMUnavailable(
                "No model credential found. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in "
                ".env, or run with --no-llm to use the heuristic plan and extractive brief."
            )
        return self._backend

    async def structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int = 8000,
        timeout: float | None = None,
    ) -> T:
        """Call the configured model and return a validated instance of `schema`."""
        backend = self._ensure_backend()
        return await backend.structured(
            schema,
            system=system,
            user=user,
            effort=effort,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()
