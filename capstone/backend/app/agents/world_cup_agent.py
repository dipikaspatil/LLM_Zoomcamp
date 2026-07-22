"""
World Cup Agent node — answers questions using live football-data.org data,
falling back to the RAG knowledge base for historical seasons the free API
plan doesn't cover.
"""
import asyncio
import re

import httpx
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.state import GraphState
from app.tools.football_api import get_standings, get_schedule
from app.tools.rag_search import search_rag

_llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)


async def world_cup_agent_node(state: GraphState) -> dict:
    # Detect a specific year in the question (e.g. "World Cup in 2022") so we
    # fetch that season's data instead of always defaulting to the current one
    year_match = re.search(r"\b(19|20)\d{2}\b", state["question"])
    season = year_match.group(0) if year_match else None

    try:
        standings, schedule, final = await asyncio.gather(
            get_standings(season=season),
            get_schedule(season=season),
            get_schedule(stage="FINAL", season=season),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and season:
            # Live API only covers the current tournament on the free plan —
            # fall back to the static knowledge base, which has historical
            # World Cup facts (past champions, notable finals)
            chunks = await search_rag(state["question"], section="world_cup_history", top_k=5)

            if chunks:
                context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
                prompt = (
                    "You are a World Cup football assistant. Live match data isn't "
                    "available for this season, so answer using the knowledge base "
                    "context below instead. If the context doesn't answer the "
                    "question, say so honestly.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {state['question']}"
                )
                response = await _llm.ainvoke(prompt)
                return {"answer": response.content}

            return {
                "answer": (
                    f"I don't have live data or knowledge base coverage for the "
                    f"{season} World Cup."
                )
            }
        raise

    # State the tournament and year explicitly
    competition_name = final.get("competition", {}).get("name", "FIFA World Cup")
    tournament_year = final.get("filters", {}).get("season", "")
    header = f"Tournament: {competition_name} {tournament_year}\n\n"

    # Surface the tournament winner as its own explicit fact
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

    # Standings summary
    standings_lines = []
    for group in standings.get("standings", []):
        group_name = group.get("group") or group.get("stage", "")
        for row in group.get("table", []):
            team = row["team"]["name"]
            standings_lines.append(
                f"{group_name} - #{row['position']} {team}: {row['points']} pts "
                f"({row['won']}W {row['draw']}D {row['lost']}L)"
            )
    standings_text = "\n".join(standings_lines)

    # Match results summary
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

    prompt = (
        "You are a World Cup football assistant. Answer the user's question "
        "using only the data provided below. If the data doesn't contain the "
        "answer, say so honestly instead of guessing.\n\n"
        f"{header}"
        f"{winner_summary}"
        f"Standings:\n{standings_text}\n\n"
        f"Matches:\n{matches_text}\n\n"
        f"Question: {state['question']}"
    )

    response = await _llm.ainvoke(prompt)
    return {"answer": response.content}
