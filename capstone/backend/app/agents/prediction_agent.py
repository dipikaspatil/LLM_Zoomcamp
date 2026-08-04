"""
Prediction Agent node — predicts match outcomes using a deterministic
scoring formula (plain code, not a trained model), then has the LLM
explain the computed numbers. The LLM never invents the probabilities
itself — see PHASE_0.md's original design decision on this.
"""
import re
from datetime import date

import httpx
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState
from app.tools.football_api import get_standings

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

LEAGUE_KEYWORDS = {
    "PL": r"premier league|\bepl\b",
    "PD": r"la liga|primera division",
    "BL1": r"bundesliga",
    "SA": r"serie a",
    "FL1": r"ligue 1|ligue une",
    "CL": r"champions league|\bucl\b",
}

COMPETITION_NAMES = {
    "WC": "FIFA World Cup",
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}

WORLD_CUP_PATTERN = r"world cup|\b(19|20)\d{2}\b"


class MatchExtraction(BaseModel):
    team_a: str = Field(description="First team mentioned in the question")
    team_b: str = Field(description="Second team mentioned in the question")


_extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)
_match_extractor = _extractor_llm.with_structured_output(MatchExtraction)


def _detect_competition(state: GraphState) -> str:
    """Same league-detection pattern as Club Football Agent, plus a World Cup check."""
    text = state["question"]
    if re.search(WORLD_CUP_PATTERN, text, re.IGNORECASE):
        return "WC"
    for code, pattern in LEAGUE_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return "PL"


def _find_team_row(standings: dict, team_name: str) -> dict | None:
    """Fuzzy-matches a team name against the standings table (e.g. 'Arsenal' matches 'Arsenal FC')."""
    team_name_lower = team_name.lower()
    for group in standings.get("standings", []):
        table_type = group.get("type", "")
        if table_type and table_type != "TOTAL":
            continue
        for row in group.get("table", []):
            if team_name_lower in row["team"]["name"].lower():
                return row
    return None


def _predict_match(stats_a: dict, stats_b: dict) -> dict:
    """
    A simple, explainable deterministic formula — NOT a trained statistical
    model. Win probability is based on points-per-game (current-season
    form); draw probability increases the closer the two teams' form is.
    Good enough to demonstrate "code computes, LLM explains" honestly; not
    a genuinely validated predictive model.
    """
    form_a = stats_a["points"] / max(stats_a["playedGames"], 1)
    form_b = stats_b["points"] / max(stats_b["playedGames"], 1)

    closeness = 1 - abs(form_a - form_b) / 3
    draw_prob = max(0.15, min(0.35, 0.20 + 0.15 * closeness))

    remaining = 1 - draw_prob
    total_form = form_a + form_b
    if total_form == 0:
        win_a, win_b = remaining / 2, remaining / 2
    else:
        win_a = remaining * (form_a / total_form)
        win_b = remaining * (form_b / total_form)

    return {
        "win_a_pct": round(win_a * 100, 1),
        "win_b_pct": round(win_b * 100, 1),
        "draw_pct": round(draw_prob * 100, 1),
        "form_a": round(form_a, 2),
        "form_b": round(form_b, 2),
    }


async def prediction_agent_node(state: GraphState) -> dict:
    # TODO: replace with logger.debug() — see Future work - Structured logging / observability from Phase_0.md
    # print("=== PREDICTION AGENT REACHED ===", flush=True)
    today = date.today().isoformat()

    extraction = await _match_extractor.ainvoke(
        f"Question: {state['question']}\nWhich two teams is this question about?",
        config={"tags": ["extraction_only"]},
    )

    competition_code = _detect_competition(state)

    try:
        standings = await get_standings(competition_code=competition_code)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            answer = "I don't have current standings data for that competition to base a prediction on."
            return {"answer": answer, "messages": [AIMessage(content=answer)]}
        raise

    row_a = _find_team_row(standings, extraction.team_a)
    row_b = _find_team_row(standings, extraction.team_b)

    if not row_a or not row_b:
        missing = extraction.team_a if not row_a else extraction.team_b
        if not row_a and not row_b:
            missing = "both teams"
        answer = (
            f"I couldn't find current standings data for {missing} — they may "
            f"not be in this competition, or the season may not have enough "
            f"matches played yet."
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    prediction = _predict_match(row_a, row_b)
    team_a_name = row_a["team"]["name"]
    team_b_name = row_b["team"]["name"]

    competition_name = COMPETITION_NAMES.get(competition_code, competition_code)

    system_prompt = (
        f"Today's date is {today}.\n"
        "You are a football prediction assistant. A deterministic formula "
        "(not you) has already computed the win/draw/loss percentages below "
        "from real season stats. Your job is ONLY to explain why these "
        "numbers make sense, referencing the actual stats given. Do NOT "
        "invent or adjust the percentages yourself — present them exactly "
        "as given, and explain the reasoning (e.g. recent form, points per "
        f"game). Mention that this is based on {competition_name} standings "
        "in your answer.\n\n"
        f"Competition: {competition_name}\n"
        f"{team_a_name}: {prediction['win_a_pct']}% win — "
        f"{row_a['points']} pts in {row_a['playedGames']} games "
        f"({prediction['form_a']} pts/game)\n"
        f"{team_b_name}: {prediction['win_b_pct']}% win — "
        f"{row_b['points']} pts in {row_b['playedGames']} games "
        f"({prediction['form_b']} pts/game)\n"
        f"Draw: {prediction['draw_pct']}%\n\n"
        "Note: this is a simple heuristic based on current-season "
        "points-per-game, not a statistically validated prediction model — "
        "be honest that this is an estimate, not a certainty."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await _llm.ainvoke(messages)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}