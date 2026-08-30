"""Guard against committing a real credential.

`.env.example` is tracked, so a real key pasted into it goes straight to the remote.
This test exists because that happened once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Values that are obviously placeholders rather than credentials.
PLACEHOLDER = re.compile(r"^(|\.\.\.|<.*>|your-.*|.*\.\.\.)$")

# Provider key prefixes, with the length a real key of that kind exceeds.
KEY_SHAPES: tuple[tuple[str, int], ...] = (
    ("sk-ant-", 20),
    ("tvly-", 20),
    ("sk-", 20),
    ("ghp_", 20),
    ("AKIA", 16),
)


def env_example_values() -> list[tuple[str, str]]:
    path = REPO_ROOT / ".env.example"
    if not path.exists():
        # Collected at import time, so a missing template must not abort the whole
        # suite; test_env_example_exists reports it as a normal failure instead.
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


@pytest.mark.parametrize("name,value", env_example_values())
def test_env_example_holds_only_placeholders(name: str, value: str):
    if PLACEHOLDER.match(value):
        return
    for prefix, min_length in KEY_SHAPES:
        assert not (value.startswith(prefix) and len(value) > min_length), (
            f".env.example {name} looks like a real credential. Put it in .env "
            f"(untracked) instead - .env.example is committed and pushed."
        )


def test_env_example_exists():
    """The template must survive. Renaming it to .env (instead of copying) removes
    the documented list of settings and disables the credential guard above."""
    assert (REPO_ROOT / ".env.example").exists(), (
        ".env.example is missing - copy it to .env rather than renaming it."
    )


def test_env_is_not_tracked():
    """.env must stay untracked; .gitignore is the only thing standing between a real
    key and the remote."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignore]


def test_template_placeholders_are_not_mistaken_for_credentials(monkeypatch):
    """Copying .env.example to .env leaves "sk-ant-..." in place. A placeholder must
    read as no key at all, or the agent reports a model it cannot actually call."""
    from market_intelligence_agent.config import Settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-real-looking-key-000000000000")
    settings = Settings.from_env()
    assert settings.anthropic_api_key is None
    assert settings.tavily_api_key == "tvly-real-looking-key-000000000000"
