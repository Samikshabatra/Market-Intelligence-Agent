"""Render an AgentResult as Markdown.

The JSON in `to_spec_dict()` is the machine contract; this is the human one. Citations
render as numbered footnotes so every asserted sentence stays traceable to a URL.
"""

from __future__ import annotations

from .models import AgentResult, BriefSection

SECTION_TITLES: dict[str, str] = {
    "company_overview": "Company overview",
    "positioning": "Positioning",
    "pricing": "Pricing",
    "recent_moves": "Recent moves",
    "strengths": "Strengths",
    "weaknesses": "Weaknesses",
    "market_signals": "Market signals",
}

STATUS_LABEL: dict[str, str] = {
    "grounded": "grounded",
    "unverified": "UNVERIFIED",
    "conflicting": "CONFLICTING SOURCES",
    "insufficient_data": "INSUFFICIENT DATA",
}


def _citation_marks(section: BriefSection, index: dict[str, int]) -> str:
    marks = [f"[{index[c]}]" for c in section.citations if c in index]
    return " " + "".join(marks) if marks else ""


def render_markdown(result: AgentResult) -> str:
    """Full brief: header, sections with footnote markers, flags, sources, trace."""
    index = {record.source_id: n for n, record in enumerate(result.sources_used, start=1)}
    lines: list[str] = [
        f"# Competitor brief: {result.query}",
        "",
        f"*Generated {result.generated_at.isoformat(timespec='seconds')} - "
        f"{result.latency_ms / 1000:.1f}s - "
        f"{len(result.distinct_domains())} distinct domains - "
        f"{len(result.sources_used)} sources"
        + (f" - {result.fallback_rounds} fallback round(s)" if result.fallback_rounds else "")
        + "*",
        "",
    ]

    for name, section in result.brief.sections():
        title = SECTION_TITLES[name]
        status = STATUS_LABEL[section.status]
        lines.append(f"## {title}")
        if section.is_asserted():
            lines.append(f"{section.text}{_citation_marks(section, index)}")
            lines.append("")
            lines.append(f"*confidence {section.confidence:.2f}*")
        elif section.status == "conflicting":
            lines.append(f"**{status}** - sources disagree; both are cited below.")
            if section.text:
                lines.append("")
                lines.append(f"> {section.text}{_citation_marks(section, index)}")
        elif section.status == "unverified":
            lines.append(
                f"**{status}** (confidence {section.confidence:.2f}) - "
                "reported below but not corroborated well enough to assert."
            )
            if section.text:
                lines.append("")
                lines.append(f"> {section.text}{_citation_marks(section, index)}")
        else:
            lines.append(f"**{status}** - no retrieved source supports this section.")
        lines.append("")

    if result.unverified_flags:
        lines.append("## Flags")
        lines.extend(f"- {flag}" for flag in result.unverified_flags)
        lines.append("")

    lines.append("## Sources")
    for record in result.sources_used:
        published = record.published_at.date().isoformat() if record.published_at else "undated"
        lines.append(
            f"{index[record.source_id]}. [{record.domain}]({record.url}) - "
            f"{record.title or 'untitled'} ({record.source_kind}, published {published}, "
            f"retrieved {record.retrieved_at.date().isoformat()})"
        )
    lines.append("")

    lines.append("## Search plan")
    lines.extend(f"{n}. {step}" for n, step in enumerate(result.search_plan_trace, start=1))
    lines.append("")

    timings = result.timings
    lines.append("## Timing")
    lines.append(
        f"planning {timings.planning_ms:.0f}ms - search {timings.search_ms:.0f}ms - "
        f"grounding {timings.grounding_ms:.0f}ms - fallback {timings.fallback_ms:.0f}ms - "
        f"synthesis {timings.synthesis_ms:.0f}ms - total {result.latency_ms:.0f}ms"
        + ("  **over budget**" if result.budget_exceeded else "")
    )
    return "\n".join(lines)


def render_summary(result: AgentResult) -> str:
    """One-line status suitable for logs and the eval harness console output."""
    asserted = sum(1 for _, s in result.brief.sections() if s.is_asserted())
    return (
        f"{result.latency_ms / 1000:5.1f}s | {len(result.distinct_domains()):2d} domains | "
        f"{asserted}/7 sections asserted | {len(result.unverified_flags)} flags | "
        f"{result.query[:60]}"
    )
