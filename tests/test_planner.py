from __future__ import annotations

import pytest

from market_intelligence_agent.config import Settings
from market_intelligence_agent.llm import LLMClient
from market_intelligence_agent.planner import (
    Planner,
    extract_subject,
    heuristic_plan,
    heuristic_replan,
)


def test_extract_subject_prefers_proper_nouns():
    assert extract_subject("How does Notion position against Coda?") == "Notion Coda"


def test_extract_subject_ignores_leading_question_words():
    assert "How" not in extract_subject("How is Ramp priced for mid-market teams?")


def test_heuristic_plan_covers_at_least_five_source_kinds(settings: Settings):
    plan = heuristic_plan("Compare Figma and Sketch in enterprise design", settings)
    assert len({sq.source_kind for sq in plan.sub_questions}) >= 5


def test_heuristic_plan_ids_are_stable_and_round_tagged(settings: Settings):
    plan = heuristic_plan("Ramp pricing", settings)
    assert [sq.id for sq in plan.sub_questions][:3] == ["r0q1", "r0q2", "r0q3"]
    assert plan.round_index == 0


def test_heuristic_plan_respects_max_sub_questions():
    tight = Settings(max_sub_questions=3)
    plan = heuristic_plan("Ramp pricing", tight)
    assert len(plan.sub_questions) == 3


def test_heuristic_replan_targets_missing_kinds(settings: Settings):
    first = heuristic_plan("Ramp pricing", settings)
    second = heuristic_replan("Ramp pricing", first, ["review_platform"], settings)
    assert second.round_index == 1
    assert {sq.source_kind for sq in second.sub_questions} == {"review_platform"}


@pytest.mark.asyncio
async def test_planner_falls_back_to_heuristics_without_credentials(settings: Settings):
    planner = Planner(LLMClient(settings), settings)
    plan = await planner.plan("How does Vanta compare to Drata on pricing?")
    assert plan.sub_questions
    assert plan.subject
