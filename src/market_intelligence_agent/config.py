"""Runtime configuration, resolved from the environment with spec-derived defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Domains that reliably produce low-signal or unattributable content for market
# research. Sources from these hosts are dropped before they reach the evidence store.
DEFAULT_DENYLIST: tuple[str, ...] = (
    "pinterest.com",
    "quora.com",
    "answers.com",
    "ezinearticles.com",
    "slideshare.net",
    "scribd.com",
    "coursehero.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
)

# Authority tiers feed the confidence scorer (see confidence.py). Higher is better.
DEFAULT_AUTHORITY_TIERS: dict[str, float] = {
    # Funding / company databases
    "crunchbase.com": 0.95,
    "sec.gov": 1.0,
    # Review platforms
    "g2.com": 0.9,
    "capterra.com": 0.85,
    "trustradius.com": 0.85,
    "gartner.com": 0.95,
    # News / trade press
    "techcrunch.com": 0.85,
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.95,
    "ft.com": 0.95,
    "theinformation.com": 0.9,
    "businessinsider.com": 0.7,
    # Professional / hiring signals
    "linkedin.com": 0.75,
    # Community
    "news.ycombinator.com": 0.5,
    "reddit.com": 0.45,
    "medium.com": 0.4,
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


@dataclass(slots=True)
class Settings:
    """All tunables for one agent run.

    The stage budgets mirror section 6 of the spec; the orchestrator treats them as
    soft deadlines and hard-caps the whole run at ``total_budget_seconds``.
    """

    # --- credentials -------------------------------------------------------
    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None

    # --- models ------------------------------------------------------------
    model: str = "claude-opus-5"
    planner_effort: str = "low"
    synthesizer_effort: str = "medium"

    # --- search ------------------------------------------------------------
    search_provider: str = "tavily"
    # Optional JSON corpus for the mock provider, so offline runs can model sparse
    # and conflicting queries instead of always returning rich synthetic evidence.
    mock_corpus_path: str | None = None
    results_per_subquestion: int = 5
    min_distinct_domains: int = 5
    max_sources: int = 24
    per_request_timeout_seconds: float = 8.0
    max_parallel_searches: int = 8
    denylist: tuple[str, ...] = DEFAULT_DENYLIST
    authority_tiers: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AUTHORITY_TIERS))
    default_authority: float = 0.6

    # --- planning ----------------------------------------------------------
    min_sub_questions: int = 3
    max_sub_questions: int = 8

    # --- confidence & fallback ---------------------------------------------
    confidence_threshold: float = 0.55
    max_fallback_rounds: int = 1
    fallback_enabled: bool = True
    recency_half_life_days: float = 540.0

    # --- latency budget (seconds) ------------------------------------------
    total_budget_seconds: float = 60.0
    planning_budget_seconds: float = 5.0
    search_budget_seconds: float = 25.0
    grounding_budget_seconds: float = 10.0
    fallback_budget_seconds: float = 15.0
    synthesis_budget_seconds: float = 5.0

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from a ``.env`` file / process environment, then apply overrides."""
        load_dotenv(override=False)
        settings = cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            model=os.getenv("MIA_MODEL", "claude-opus-5"),
            search_provider=os.getenv("MIA_SEARCH_PROVIDER", "tavily"),
            mock_corpus_path=os.getenv("MIA_MOCK_CORPUS"),
            total_budget_seconds=_env_float("MIA_TOTAL_BUDGET_SECONDS", 60.0),
            confidence_threshold=_env_float("MIA_CONFIDENCE_THRESHOLD", 0.55),
            min_distinct_domains=_env_int("MIA_MIN_DISTINCT_DOMAINS", 5),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(settings, key):
                raise AttributeError(f"Unknown setting: {key}")
            setattr(settings, key, value)
        return settings

    def authority_for(self, domain: str) -> float:
        """Authority weight for a domain, falling back to the neutral default."""
        domain = domain.lower().removeprefix("www.")
        if domain in self.authority_tiers:
            return self.authority_tiers[domain]
        # Match the registrable suffix so `blog.g2.com` inherits `g2.com`.
        for known, score in self.authority_tiers.items():
            if domain.endswith("." + known):
                return score
        return self.default_authority

    def is_denied(self, domain: str) -> bool:
        domain = domain.lower().removeprefix("www.")
        return any(domain == d or domain.endswith("." + d) for d in self.denylist)
