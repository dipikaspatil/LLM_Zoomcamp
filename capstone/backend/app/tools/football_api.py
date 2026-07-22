# backend/app/tools/football_api.py
import httpx
from app.config import settings

BASE_URL = "https://api.football-data.org/v4"

async def _get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}{path}",
            headers={"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY},
            params=params,
        )

        remaining = response.headers.get("X-RequestsAvailable")
        reset_seconds = response.headers.get("X-RequestCounter-Reset")

        if response.status_code == 429:
            wait = int(reset_seconds) if reset_seconds else 60
            print(f"Rate limited by football-data.org, waiting {wait}s", flush=True)
            await asyncio.sleep(wait)
            return await _get(path, params)

        response.raise_for_status()

        if remaining is not None and int(remaining) <= 1 and reset_seconds:
            print(f"Only {remaining} requests left, pausing {reset_seconds}s", flush=True)
            await asyncio.sleep(int(reset_seconds))

        return response.json()


async def get_standings(competition_code: str = "WC", season: str | None = None) -> dict:
    params = {"season": season} if season else None
    return await _get(f"/competitions/{competition_code}/standings", params=params)


async def get_schedule(
    competition_code: str = "WC",
    date_from: str | None = None,
    date_to: str | None = None,
    stage: str | None = None,
    season: str | None = None,
) -> dict:
    params = {
        k: v for k, v in
        {"dateFrom": date_from, "dateTo": date_to, "stage": stage, "season": season}.items()
        if v
    }
    return await _get(f"/competitions/{competition_code}/matches", params=params)


async def get_live_matches(competition_code: str = "WC") -> dict:
    return await _get(f"/competitions/{competition_code}/matches", params={"status": "LIVE"})
