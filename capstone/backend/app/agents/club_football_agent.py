"""
Club Football Agent node — answers questions about club competitions
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League)
using live football-data.org data. Reuses the same football_api.py tools
as the World Cup Agent, just with a different competition code.
"""
import asyncio
import re
from datetime import date

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState
from app.tools.football_api import get_standings, get_schedule

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

LEAGUE_NAMES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}

LEAGUE_KEYWORDS = {
    "PL": r"premier league|\bepl\b",
    "PD": r"la liga|primera division",
    "BL1": r"bundesliga",
    "SA": r"serie a",
    "FL1": r"ligue 1|ligue une",
    "CL": r"champions league|\bucl\b",
}

CURRENT_YEAR_PATTERN = r"\bcurrent year\b|\bthis year\b|\bcurrently\b|\bright now\b"


def _extract_league(state: GraphState) -> str:
    for code, pattern in LEAGUE_KEYWORDS.items():
        if re.search(pattern, state["question"], re.IGNORECASE):
            return code
    for message in reversed(state["messages"][:-1]):
        for code, pattern in LEAGUE_KEYWORDS.items():
            if re.search(pattern, message.content, re.IGNORECASE):
                return code
    return "PL"


def _extract_season(state: GraphState) -> str | None:
    if re.search(CURRENT_YEAR_PATTERN, state["question"], re.IGNORECASE):
        return None
    year_match = re.search(r"\b(19|20)\d{2}\b", state["question"])
    if year_match:
        return year_match.group(0)
    for message in reversed(state["messages"][:-1]):
        year_match = re.search(r"\b(19|20)\d{2}\b", message.content)
        if year_match:
            return year_match.group(0)
    return None


def _format_standings(standings: dict) -> str:
    """Formats a standings API response as a markdown table."""
    rows = []
    for group in standings.get("standings", []):
        table_type = group.get("type", "")
        if table_type and table_type != "TOTAL":
            continue
        for row in group.get("table", []):
            team = row["team"]["name"]
            rows.append(
                f"| {row['position']} | {team} | {row['points']} | "
                f"{row['won']} | {row['draw']} | {row['lost']} | "
                f"{row['goalsFor']}-{row['goalsAgainst']} |"
            )
    if not rows:
        return ""
    header = "| Pos | Team | Pts | W | D | L | GF-GA |\n|---|---|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


async def club_football_agent_node(state: GraphState) -> dict:
    competition_code = _extract_league(state)
    league_name = LEAGUE_NAMES[competition_code]
    season = _extract_season(state)
    today = date.today().isoformat()

    try:
        standings, schedule = await asyncio.gather(
            get_standings(competition_code=competition_code, season=season),
            get_schedule(competition_code=competition_code, season=season),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            answer = (
                f"I don't have data for {league_name}"
                + (f" in {season}" if season else "")
                + " — either it's outside the free API plan's coverage, or "
                  "that competition isn't included on the free tier."
            )
            return {"answer": answer, "messages": [AIMessage(content=answer)]}
        raise

    tournament_year = standings.get("filters", {}).get("season", season or "")

    all_zero = all(
        (row.get("playedGames") or 0) == 0
        for group in standings.get("standings", [])
        for row in group.get("table", [])
    )

    off_season_note = ""
    standings_block = ""

    if all_zero and not season:
        off_season_note = (
            f"Note: the {tournament_year} season has not started yet — no "
            f"matches have been played by any team so far.\n\n"
        )
        try:
            previous_season_year = str(int(tournament_year) - 1)
            previous_standings = await get_standings(
                competition_code=competition_code, season=previous_season_year
            )
            # Deliberately NOT including the empty current-season table here —
            # showing it alongside real previous-season data confused the LLM
            # into reporting the meaningless zeros instead of the useful data
            standings_block = (
                f"Most Recent Completed Season ({previous_season_year} final standings):\n"
                f"{_format_standings(previous_standings)}\n\n"
            )
        except (ValueError, httpx.HTTPStatusError):
            pass
    else:
        standings_block = f"Standings:\n{_format_standings(standings)}\n\n"

    header = (
        f"Today's date: {today}\n"
        f"Competition: {league_name} {tournament_year}\n\n"
    )

    match_lines = []
    for match in schedule.get("matches", []):
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        status = match["status"]
        if status == "FINISHED":
            score = match["score"]["fullTime"]
            match_lines.append(f"{home} {score['home']} - {score['away']} {away}")
        else:
            match_lines.append(f"{home} vs {away} ({status})")
    matches_text = "\n".join(match_lines)

    off_season_instruction = (
        "IMPORTANT: The user may ask about 'current' standings, but the new "
        "season hasn't started yet, so no current-season data exists. In this "
        "case, you MUST still answer using the previous season's final "
        "standings provided below — do NOT refuse to answer or say no data is "
        "available. Clearly state the new season hasn't started, then present "
        "the previous season's standings as the most relevant available answer.\n\n"
        if off_season_note else ""
    )

    system_prompt = (
        "You are a club football assistant. Answer the user's question using "
        "only the data below and the conversation so far. If the data doesn't "
        "contain the answer, say so honestly instead of guessing. Tables in "
        "the data below are already formatted as markdown — include them "
        "exactly as given, do not retype or reformat them, just add a short "
        "sentence of context.\n\n"
        f"{header}"
        f"{off_season_note}"
        f"{off_season_instruction}"
        f"{standings_block}"
        f"Recent/Scheduled Matches:\n{matches_text}"
    )


    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await _llm.ainvoke(messages)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}
