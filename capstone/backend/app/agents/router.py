"""
Section Match Check node — the "hybrid router" from PHASE_0/PHASE_1.

The user picks a section manually in the UI (Phase 1 design), but we still
guardrail against an off-topic question landing in the wrong agent and
getting a bad answer. This node asks the LLM a narrow yes/no question:
"does this question actually belong in the section the user picked?"
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.state import GraphState

from datetime import date

# Plain-English description of each section, given to the LLM as context —
# this is the only place these descriptions live, so update here if sections change
SECTION_DESCRIPTIONS = {
    "world_cup": "Live matches, standings, schedules, and results for the FIFA World Cup — including which team won any given tournament.",
    "knowledge": "General football knowledge: tactics, concepts, and World Cup history.",
}


class SectionMatchResult(BaseModel):
    """Defines the exact shape we want the LLM's output in — no free text to parse."""
    matches: bool = Field(description="True if the question fits the given section.")
    reason: str = Field(description="One short sentence explaining the decision.")


_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)

# with_structured_output wraps the LLM so it returns a SectionMatchResult object
# directly, instead of us parsing free-text output ourselves
_router_llm = _llm.with_structured_output(SectionMatchResult)


async def check_section_match(state: GraphState) -> dict:
    section_description = SECTION_DESCRIPTIONS[state["section"]]

    result = await _router_llm.ainvoke(
        f"Today's date is {date.today().isoformat()}.\n"
        f"Section: {state['section']} — {section_description}\n"
        f"User question: {state['question']}\n"
        f"Does this question belong in this section?"
    )

    if result.matches:
        return {"section_valid": True}

    # Mismatch: write the explanation straight into "answer" so the graph
    # can end here with something useful to show the user
    return {
        "section_valid": False,
        "answer": (
            f"That question doesn't seem to fit the '{state['section']}' section "
            f"({result.reason}). Try picking a different section."
        ),
    }
