import os
from datetime import UTC, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

import dlt
import requests

LOGFIRE_READ_TOKEN = os.environ["LOGFIRE_READ_TOKEN"]
QUERY_URL = "https://logfire-us.pydantic.dev/v2/query"


@dlt.resource(name="spans", write_disposition="replace")
def logfire_spans(lookback_hours: int = 48):
    headers = {
        "Authorization": f"Bearer {LOGFIRE_READ_TOKEN}",
        "Accept": "application/json",
    }
    # records is Logfire's built-in virtual table name for the Query API 
    # So the query SELECT * FROM records ORDER BY start_timestamp reads: "give me every span/log row in my project, oldest first" 
    body = {
        "sql": "SELECT * FROM records ORDER BY start_timestamp",
        "min_timestamp": (datetime.now(tz=UTC) - timedelta(hours=lookback_hours)).isoformat(),
    }
    resp = requests.post(QUERY_URL, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    yield from data["data"]

def load():
    pipeline = dlt.pipeline(
        pipeline_name="logfire_ingest",   # was "agent_traces" — this was the collision
        destination="duckdb",
        dataset_name="agent_traces",       # keep this as-is
    )
    load_info = pipeline.run(logfire_spans())
    print(load_info)


if __name__ == "__main__":
    load()
