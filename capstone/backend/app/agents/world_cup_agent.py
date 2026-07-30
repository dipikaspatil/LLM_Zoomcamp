"""
World Cup Agent node — answers questions using live football-data.org data,
falling back to the RAG knowledge base for historical seasons the free API
plan doesn't cover.
"""
import asyncio
import re
from datetime import date

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState
from app.tools.football_api import get_standings, get_schedule, get_top_scorers
from app.tools.rag_search import search_rag

_llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)


CURRENT_YEAR_PATTERN = r"\bcurrent year\b|\bthis year\b|\bcurrently\b|\bright now\b"

def _extract_season(state: GraphState) -> str | None:
    """
    Looks for a 4-digit year in the current question first. If the current
    question explicitly says "current year"/"this year"/etc, that's treated
    as an override meaning "use live/current data" — checked before falling
    back to history, so a stale year mentioned earlier in the conversation
    can't override an explicit "now" in the current question.
    Otherwise checks the rest of the conversation, most recent first, for
    follow-ups that don't repeat the year but still refer to it implicitly.
    """
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

def _mentions_scorers(state: GraphState) -> bool:
    keywords = r"golden boot|top scorer|top scorers|most goals|leading scorer"
    if re.search(keywords, state["question"], re.IGNORECASE):
        return True
    for message in reversed(state["messages"][:-1]):
        if re.search(keywords, message.content, re.IGNORECASE):
            return True
    return False


async def world_cup_agent_node(state: GraphState) -> dict:
    season = _extract_season(state)
    include_scorers = _mentions_scorers(state)
    today = date.today().isoformat()

    try:
        tasks = [
            get_standings(season=season),
            get_schedule(season=season),
            get_schedule(stage="FINAL", season=season),
        ]
        if include_scorers:
            tasks.append(get_top_scorers(season=season))

        results = await asyncio.gather(*tasks)
        standings, schedule, final = results[0], results[1], results[2]
        scorers = results[3] if include_scorers else None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and season:
            chunks = await search_rag(state["question"], section="world_cup_history", top_k=5)

            if chunks:
                context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
                system_prompt = (
                    f"Today's date is {today}.\n"
                    "You are a World Cup football assistant. Live match data isn't "
                    "available for this season, so answer using the knowledge base "
                    "context below and the conversation so far. If the context "
                    "doesn't answer the question, say so honestly.\n\n"
                    f"Context:\n{context}"
                )
                messages = [SystemMessage(content=system_prompt)] + state["messages"]
                response = await _llm.ainvoke(messages)
                return {"answer": response.content, "messages": [AIMessage(content=response.content)]}

            answer = (
                f"I don't have live data or knowledge base coverage for the "
                f"{season} World Cup."
            )
            return {"answer": answer, "messages": [AIMessage(content=answer)]}
        raise

    competition_name = final.get("competition", {}).get("name", "FIFA World Cup")
    tournament_year = final.get("filters", {}).get("season", "")
    header = (
        f"Today's date: {today}\n"
        f"Tournament: {competition_name} {tournament_year} "
        f"(this is the current/most recent tournament's data)\n\n"
    )

    tournament_winner = None
    final_matches = final.get("matches", [])
    if final_matches and final_matches[0]["status"] == "FINISHED":
        final_match = final_matches[0]
        winner_code = final_match["score"].get("winner")
        if winner_code == "HOME_TEAM":
            tournament_winner = final_match["homeTeam"]["name"]
        elif winner_code == "AWAY_TEAM":
            tournament_winner = final_match["awayTeam"]["name"]

    winner_summary = f"Tournament Winner: {tournament_winner}\n\n" if tournament_winner else ""

    scorers_text = ""
    if scorers is not None:
        scorer_lines = []
        for entry in scorers.get("scorers", []):
            player = entry["player"]["name"]
            team = entry["team"]["name"]
            goals = entry.get("goals", 0)
            scorer_lines.append(f"{player} ({team}): {goals} goals")
        scorers_text = "Top Scorers:\n" + "\n".join(scorer_lines) + "\n\n"

    standings_rows = []
    for group in standings.get("standings", []):
        group_name = group.get("group") or group.get("stage", "")
        for row in group.get("table", []):
            team = row["team"]["name"]
            standings_rows.append(
                f"| {group_name} | {row['position']} | {team} | {row['points']} | "
                f"{row['won']} | {row['draw']} | {row['lost']} |"
            )
    if standings_rows:
        standings_text = (
            "| Group | Pos | Team | Pts | W | D | L |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(standings_rows)
        )
    else:
        standings_text = ""

    all_matches = schedule.get("matches", []) + final.get("matches", [])
    seen_ids = set()
    match_lines = []
    for match in all_matches:
        if match["id"] in seen_ids:
            continue
        seen_ids.add(match["id"])

        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        stage_name = match.get("stage", "")
        status = match["status"]

        if status == "FINISHED":
            score = match["score"]["fullTime"]
            winner_code = match["score"].get("winner")
            if winner_code == "HOME_TEAM":
                result = f"{home} won"
            elif winner_code == "AWAY_TEAM":
                result = f"{away} won"
            elif winner_code == "DRAW":
                result = "Draw"
            else:
                result = ""
            match_lines.append(
                f"{stage_name}: {home} {score['home']} - {score['away']} {away} ({result})"
            )
        else:
            match_lines.append(f"{stage_name}: {home} vs {away} ({status})")

    matches_text = "\n".join(match_lines)

    system_prompt = (
        "You are a World Cup football assistant. Answer the user's question "
        "using only the data provided below and the conversation so far. If "
        "the data doesn't contain the answer, say so honestly instead of "
        "guessing. Tables in the data below are already formatted as markdown "
        "— include them in your response exactly as given, do not retype or "
        "reformat them, just add a short sentence of context.\n\n"
        f"{header}"
        f"{winner_summary}"
        f"{scorers_text}"
        f"Standings:\n{standings_text}\n\n"
        f"Matches:\n{matches_text}"
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await _llm.ainvoke(messages)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}
