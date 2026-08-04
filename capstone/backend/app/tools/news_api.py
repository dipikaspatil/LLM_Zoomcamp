"""
NewsAPI.org tool — fetches recent football news for a specific team.

The noise-filtering happens HERE, deterministically, at query time —
not as a separate LLM step. Two filters do the work:
  - qInTitle: only matches articles with the team name in the HEADLINE
    (not just anywhere in a 6000-word unrelated article)
  - domains: restricts results to a curated allowlist of sports outlets,
    so a team name that happens to match a politics story never
    reaches the LLM in the first place
"""
import httpx
from app.config import settings

BASE_URL = "https://newsapi.org/v2/everything"

# Curated allowlist — verified working via a live test query for "Arsenal"
SPORTS_DOMAINS = "skysports.com,bbc.co.uk,espn.com,goal.com,theguardian.com"


async def get_team_news(team: str, limit: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            BASE_URL,
            params={
                "qInTitle": team,
                "domains": SPORTS_DOMAINS,
                "sortBy": "publishedAt",
                "pageSize": str(limit),
                "language": "en",
                "apiKey": settings.NEWS_API_KEY,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
