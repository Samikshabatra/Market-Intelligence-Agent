"""Provider selection and the two backends' contracts.

Neither backend is called over the network here: each is driven with a stub client
shaped like the real SDK response, so the translation layer is what gets tested.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from market_intelligence_agent.config import Settings
from market_intelligence_agent.llm import (
    AnthropicBackend,
    GeminiBackend,
    LLMClient,
    LLMError,
    LLMUnavailable,
    resolve_provider,
)


class Answer(BaseModel):
    subject: str
    score: int


# ------------------------------------------------------------------ provider choice


def test_auto_prefers_anthropic_when_both_keys_exist():
    settings = Settings(anthropic_api_key="sk-ant-x", gemini_api_key="g-x")
    assert resolve_provider(settings) == "anthropic"


def test_auto_falls_back_to_gemini_when_that_is_the_only_key():
    assert resolve_provider(Settings(gemini_api_key="g-x")) == "gemini"


def test_auto_reports_nothing_without_any_key():
    assert resolve_provider(Settings()) is None


def test_explicit_provider_without_its_key_is_unavailable():
    """Naming gemini but supplying only an Anthropic key must not silently use Claude."""
    settings = Settings(llm_provider="gemini", anthropic_api_key="sk-ant-x")
    assert resolve_provider(settings) is None


def test_explicit_provider_uses_its_own_key():
    settings = Settings(llm_provider="gemini", gemini_api_key="g-x")
    assert resolve_provider(settings) == "gemini"


def test_provider_none_disables_the_model_even_with_a_key():
    settings = Settings(llm_provider="none", gemini_api_key="g-x")
    assert resolve_provider(settings) is None


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMUnavailable):
        resolve_provider(Settings(llm_provider="llama"))


def test_client_reports_availability_and_provider():
    client = LLMClient(Settings(gemini_api_key="g-x"))
    assert client.available
    assert client.provider == "gemini"

    empty = LLMClient(Settings())
    assert not empty.available
    assert empty.provider is None


@pytest.mark.asyncio
async def test_calling_without_a_credential_explains_the_options():
    with pytest.raises(LLMUnavailable) as caught:
        await LLMClient(Settings()).structured(Answer, system="s", user="u")
    message = str(caught.value)
    assert "GEMINI_API_KEY" in message and "--no-llm" in message


# ------------------------------------------------------------------ gemini backend


class StubAsyncModels:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class StubGeminiClient:
    def __init__(self, response):
        self.aio = type("Aio", (), {"models": StubAsyncModels(response)})()


class StubResponse:
    def __init__(self, parsed):
        self.parsed = parsed


@pytest.mark.asyncio
async def test_gemini_returns_the_validated_model():
    expected = Answer(subject="Linear", score=7)
    client = StubGeminiClient(StubResponse(expected))
    backend = GeminiBackend(Settings(gemini_api_key="g-x"), client=client)

    result = await backend.structured(
        Answer, system="be terse", user="rate Linear", effort="low",
        max_tokens=500, timeout=None,
    )
    assert result == expected


@pytest.mark.asyncio
async def test_gemini_sends_the_schema_and_system_instruction():
    client = StubGeminiClient(StubResponse(Answer(subject="x", score=1)))
    settings = Settings(gemini_api_key="g-x", gemini_model="gemini-2.5-flash")
    await GeminiBackend(settings, client=client).structured(
        Answer, system="be terse", user="rate", effort="low", max_tokens=500, timeout=None,
    )
    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"].system_instruction == "be terse"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is Answer


@pytest.mark.asyncio
async def test_gemini_unparseable_response_raises_llm_error():
    client = StubGeminiClient(StubResponse(None))
    with pytest.raises(LLMError, match="no parseable"):
        await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )


@pytest.mark.asyncio
async def test_gemini_wrong_type_is_rejected_rather_than_passed_on():
    """A response that is not the requested model must not reach the pipeline."""
    client = StubGeminiClient(StubResponse({"subject": "Linear", "score": 7}))
    with pytest.raises(LLMError, match="expected Answer"):
        await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )


@pytest.mark.asyncio
async def test_gemini_api_errors_become_llm_errors():
    from google.genai import errors as genai_errors

    # 404 and 429 have their own guidance; anything else surfaces generically.
    failure = genai_errors.APIError(500, {"message": "internal error"})
    client = StubGeminiClient(failure)
    with pytest.raises(LLMError, match="Gemini request failed"):
        await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )


# ------------------------------------------------------------------ anthropic backend


class StubMessages:
    def __init__(self, response):
        self._response = response
        self.kwargs: dict = {}

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return self._response


class StubAnthropicClient:
    def __init__(self, response):
        self.messages = StubMessages(response)


@pytest.mark.asyncio
async def test_anthropic_returns_parsed_output():
    expected = Answer(subject="Jira", score=4)
    client = StubAnthropicClient(type("R", (), {"parsed_output": expected})())
    backend = AnthropicBackend(Settings(anthropic_api_key="sk-ant-x"), client=client)
    result = await backend.structured(
        Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
    )
    assert result == expected


@pytest.mark.asyncio
async def test_anthropic_missing_parsed_output_raises():
    client = StubAnthropicClient(type("R", (), {"parsed_output": None})())
    with pytest.raises(LLMError, match="no parseable"):
        await AnthropicBackend(Settings(anthropic_api_key="sk-ant-x"), client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )


# ------------------------------------------------------------------ pipeline wiring


@pytest.mark.asyncio
async def test_planner_uses_the_gemini_backend_when_that_is_configured():
    from market_intelligence_agent.planner import Planner

    class Recording(GeminiBackend):
        def __init__(self):
            self.used = False

        async def structured(self, schema, **kwargs):
            self.used = True
            return schema(
                subject="Linear",
                complexity="simple",
                sub_questions=[
                    {"id": "q1", "question": "What is Linear?", "search_query": "Linear"}
                ],
            )

    backend = Recording()
    settings = Settings(gemini_api_key="g-x")
    plan = await Planner(LLMClient(settings, backend=backend), settings).plan("What is Linear?")
    assert backend.used
    assert plan.sub_questions[0].search_query == "Linear"


@pytest.mark.asyncio
async def test_a_retired_model_id_explains_how_to_fix_it():
    """A 404 on the model id is the likeliest first-run failure; the message has to
    name the setting to change rather than just echo the status code."""
    from google.genai import errors as genai_errors

    failure = genai_errors.APIError(404, {"message": "model is no longer available"})
    client = StubGeminiClient(failure)
    settings = Settings(gemini_api_key="g-x", gemini_model="gemini-2.5-flash")
    with pytest.raises(LLMError, match="MIA_GEMINI_MODEL"):
        await GeminiBackend(settings, client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )


@pytest.mark.asyncio
async def test_gemini_reserves_headroom_for_reasoning_tokens():
    """Reasoning tokens come out of the same allowance as the answer. Passing the
    caller's ceiling straight through truncated the JSON and lost the response."""
    client = StubGeminiClient(StubResponse(Answer(subject="x", score=1)))
    await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
        Answer, system="s", user="u", effort="low", max_tokens=4000, timeout=None,
    )
    config = client.aio.models.calls[0]["config"]
    assert config.max_output_tokens > 4000


@pytest.mark.asyncio
async def test_gemini_maps_effort_onto_thinking_level():
    for effort, expected in (("low", "LOW"), ("medium", "MEDIUM"), ("max", "HIGH")):
        client = StubGeminiClient(StubResponse(Answer(subject="x", score=1)))
        await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
            Answer, system="s", user="u", effort=effort, max_tokens=100, timeout=None,
        )
        config = client.aio.models.calls[0]["config"]
        assert config.thinking_config.thinking_level == expected


@pytest.mark.asyncio
async def test_quota_exhaustion_is_reported_in_plain_terms():
    """The raw 429 is a wall of quota JSON. The message should say what happened and
    what the options are."""
    from google.genai import errors as genai_errors

    failure = genai_errors.APIError(429, {"message": "RESOURCE_EXHAUSTED quota"})
    client = StubGeminiClient(failure)
    with pytest.raises(LLMError, match="quota exhausted"):
        await GeminiBackend(Settings(gemini_api_key="g-x"), client=client).structured(
            Answer, system="s", user="u", effort="low", max_tokens=100, timeout=None,
        )
