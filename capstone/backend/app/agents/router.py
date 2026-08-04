"""
Section classifier node — auto-detects which agent should handle a
question. Replaces the earlier manual-pick + validate design: instead of
checking whether a question fits a section the user already chose, this
node decides which section fits the question in the first place.
"""
from datetime import date
import re
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState

SECTION_DESCRIPTIONS = {
    "world_cup": (
        "Live matches, standings, schedules, and results for the FIFA World Cup — "
        "for any tournament year, past or present. A question fits this section "
        "regardless of whether that tournament's year is before or after today's "
        "date — never reject a question because of the tournament's year."
    ),
    "club_football": (
        "Club football competitions that run year-round: Premier League, La Liga, "
        "Bundesliga, Serie A, Ligue 1, and the Champions League — standings, "
        "results, and team form. Fits regardless of season year, past or present."
    ),
    "prediction": (
        "Predicting the outcome of a SPECIFIC match between two named teams "
        "(e.g. 'who will win Arsenal vs Chelsea?'). Does NOT cover predicting "
        "an entire tournament's winner or general 'who's the best team' "
        "questions — only a specific matchup between two named teams."
    ),
    "news": (
        "Recent news, transfer rumors, injury updates, or other current "
        "happenings about a specific club or player — 'what's the latest "
        "on X' style questions. Does NOT cover standings, schedules, or "
        "results (that's club_football/world_cup), and does NOT cover "
        "general tactics or history (that's knowledge)."
    ),
    "knowledge": "General football knowledge: tactics, concepts, and World Cup history.",
}

WORLD_CUP_PATTERN = r"world cup|\b(19|20)\d{2}\b"
LEAGUE_PATTERN = (
    r"premier league|\bepl\b|la liga|primera division|bundesliga|serie a|"
    r"ligue 1|ligue une|champions league|\bucl\b"
)
PREDICTION_PATTERN = r"who will win|who('s| is) going to win|who wins|\bpredict\b|likely to win|chances? of winning"
NEWS_PATTERN = r"transfer (news|rumou?r|talk)|injury update|latest on|what'?s happening with|\bnews\b.*\b(team|club|player|football|soccer)\b"

class SectionClassification(BaseModel):
    section: Literal["world_cup", "club_football", "knowledge", "prediction", "news", "none"] = Field(
        description="Which section best fits the question, or 'none' if it isn't about football at all."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)
_classifier_llm = _llm.with_structured_output(SectionClassification)

def _fast_path_classify(state: GraphState) -> str | None:
    """
    Deterministic shortcut for the common, unambiguous cases — skips the LLM
    entirely, both for speed and because the LLM has proven unreliable at
    specific narrow classification judgments (this one: distinguishing a
    prediction question from a factual club-football question, even when
    the section description gives this exact example). Only checks the
    CURRENT question, not history — see the earlier "Explain Total Football"
    misrouting bug for why history-scanning in the fast-path is unsafe.
    """
    text = state["question"]
    if re.search(PREDICTION_PATTERN, text, re.IGNORECASE):
        return "prediction"
    if re.search(NEWS_PATTERN, text, re.IGNORECASE):
        return "news"
    if re.search(WORLD_CUP_PATTERN, text, re.IGNORECASE):
        return "world_cup"
    if re.search(LEAGUE_PATTERN, text, re.IGNORECASE):
        return "club_football"
    return None

async def classify_section(state: GraphState) -> dict:
    fast_path = _fast_path_classify(state)
    # TODO: replace with logger.debug() — see Future work - Structured logging / observability from Phase_0.md
    # print(f"[DEBUG] question={state['question']!r} fast_path={fast_path!r}")
    if fast_path:
        return {"section": fast_path, "section_valid": True}

    section_list = "\n".join(f"- {name}: {desc}" for name, desc in SECTION_DESCRIPTIONS.items())
    system_prompt = (
        f"Today's date is {date.today().isoformat()}.\n"
        f"Available sections:\n{section_list}\n\n"
        "Decide which section best fits the user's LATEST message, based on "
        "TOPIC. Use 'none' only if the question isn't about football at all. "
        "Use the conversation so far for context — a short follow-up can "
        "still belong to whichever section the conversation is about, even "
        "if it looks vague on its own."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    result = await _classifier_llm.ainvoke(messages)

    if result.section == "none":
        answer = (
            "I can only help with football-related questions — try asking "
            "about the World Cup, club football, or general football knowledge."
        )
        return {"section_valid": False, "answer": answer, "messages": [AIMessage(content=answer)]}

    return {"section": result.section, "section_valid": True}