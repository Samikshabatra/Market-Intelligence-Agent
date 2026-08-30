"""Thin async wrapper around the Anthropic SDK.

Only two things are needed from the model: a plan (structured) and a brief (structured).
Both go through `structured()`, which validates the response into a Pydantic model so no
free-text parsing happens anywhere else in the pipeline.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when no usable Anthropic credential is configured."""


class LLMError(RuntimeError):
    """Raised when the model call fails or returns something unparseable."""


class LLMClient:
    """Async Anthropic client scoped to this agent's two structured calls."""

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    @property
    def available(self) -> bool:
        """True when a client was injected or a credential is resolvable from the env."""
        return self._client is not None or bool(self._settings.anthropic_api_key)

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise LLMUnavailable(
                    "ANTHROPIC_API_KEY is not set. Run with --offline to use the "
                    "heuristic planner and extractive synthesiser instead."
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        return self._client

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
        """Call Claude and return a validated instance of `schema`."""
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
            response = await client.messages.parse(
                **kwargs, output_config={"effort": effort}
            )
        except TypeError:
            # Older SDKs reject output_config on the parse helper; effort is optional.
            response = await client.messages.parse(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic returned {exc.status_code}: {exc.message}") from exc
        except (TimeoutError, anthropic.APIConnectionError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"Model returned no parseable {schema.__name__}.")
        return parsed

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
