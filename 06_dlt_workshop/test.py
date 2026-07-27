import os
from datetime import UTC, datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import requests

headers = {
    "Authorization": f"Bearer {os.environ['LOGFIRE_READ_TOKEN']}",
    "Accept": "application/json",
}
body = {
    "sql": "SELECT * FROM records ORDER BY start_timestamp LIMIT 3",
    "min_timestamp": (datetime.now(tz=UTC) - timedelta(hours=48)).isoformat(),
}
resp = requests.post("https://logfire-us.pydantic.dev/v2/query", json=body, headers=headers)
resp.raise_for_status()
data = resp.json()
print(type(data))
print(list(data.keys()) if isinstance(data, dict) else data[:1])

print(data["schema"])
print(type(data["data"]), len(data["data"]))
print(data["data"][0] if data["data"] else "empty")