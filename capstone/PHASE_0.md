# ⚽ SoccerMind AI

**An Intelligent Multi-Agent Soccer Assistant**

> Analyze. Predict. Explain. Everything about football.

## What is this project?

SoccerMind AI is a chat assistant that answers football questions using real data and reasoning — not hardcoded answers. It uses several specialized agents, each an expert in one area (World Cup, tactics, predictions, etc.), plus a knowledge base built with RAG (Retrieval-Augmented Generation).

The World Cup is the first big use case, but the system is built to work for football all year round.

## Why this project

This project is built to show real, practical AI engineering skills:

- Multi-agent orchestration (LangGraph)
- RAG with real evaluation (not just a demo)
- A full working app: backend API + frontend UI, not just a notebook
- Explainable predictions (the AI shows its reasoning, not just a guess)

## How it works (Architecture)

```
                        User

                          │
                          ▼
              UI: User picks a section
        (World Cup / Knowledge / Tactics / etc.)

                          │
                          ▼
              Section Match Check (Router)
     (Checks if the question fits the chosen section.
      If not, asks the user to pick the right one.)

                          │
                          ▼
                    Matching Agent

                          │
                          ▼
                 Shared Tool Layer
         ┌────────────────┴────────────────┐
         │                                  │
   Football APIs                     Vector Database

         └────────────────┬────────────────┘
                          │
                          ▼
                     LLM Engine
                          │
                          ▼
                     Final Answer
```

**Note:** In Phase 1, the user picks the section manually (simple and predictable).
Later, the same router can be upgraded to auto-detect the right agent from a single text box, without picking a section first.

## The Agents

### 1. World Cup Agent 🌎 (Phase 1)
Live matches, standings, knockout bracket, match summaries, Golden Boot race.

Example questions:
- "Who won the World Cup?"
- "How did Spain do in the group stage?"
- "Who won the Golden Boot?"
- "Why did Spain beat Portugal?"
- "Who has the most goals?"

### 2. Knowledge Agent 📚 (Phase 1)
Powered by RAG. Answers questions using a knowledge base of football facts.

Phase 1 covers:
- Tactical concepts (done first)
- World Cup history (done first)

Marked TODO for later:
- FIFA Laws of the Game
- Club histories
- Famous matches
- Coaches

Example question:
- "Explain Total Football."

### 3. Club Football Agent 🏆 (Phase 2)
Works year-round. Covers Premier League, La Liga, Bundesliga, Serie A, Champions League, MLS, women's competitions.

Example question:
- "Compare Arsenal and Liverpool this season."

### 4. Tactical Analyst 🧠 (Phase 2)
Instead of just showing stats like "Possession: 61%", this agent explains *why* a team played the way it did.

Example answer style:
> "Spain controlled midfield by creating numerical superiority through inverted full-backs. Belgium struggled to progress the ball under pressure, leading to repeated turnovers in dangerous areas."

### 5. Prediction Agent 🔮 (Phase 3)
Predicts match outcomes, tournament winners, and the Golden Boot race — with reasons, not just a number. See "Prediction Approach" below for how this works without any ML model training.

Example question:
- "Spain vs Belgium — who is likely to win?"

### 6. Fantasy Football Assistant ⚡ (Phase 3)
Helps with fantasy football decisions using form, injuries, expected minutes, and fixture difficulty.

Example question:
- "Should I captain Haaland this week?"

### 7. Player Scout Agent 🔍 (Future Work)
Suggests players with a similar playing style. Not started — needs data we don't have access to yet.

## Shared Tool Layer

All agents reuse the same set of tools instead of duplicating logic:

- `get_live_matches()`
- `get_player_stats()`
- `get_team_stats()`
- `get_standings()`
- `get_schedule()`
- `get_news()`
- `search_rag()`
- `calculate_prediction_features()`

## Prediction Approach — No ML Training

We are **not** training a machine learning model for predictions. Instead, we compute the numbers with simple, reproducible logic, and let the LLM explain them in plain language.

```
User Question
      │
      ▼
Collect Statistics        (via Shared Tool Layer: team stats, standings, head-to-head)
      │
      ▼
Calculate Features         (form score, goal difference, head-to-head record — plain code)
      │
      ▼
Deterministic Scoring      (a simple weighted formula turns features into win/draw/loss %)
      │
      ▼
LLM Explanation             (LLM receives the computed numbers and explains them —
                             it does not invent the numbers itself)
      │
      ▼
Explainable Prediction
```

This keeps predictions reproducible and honest — the LLM's job is to explain, not to guess.

## Tech Stack

- Orchestration: LangGraph (agent routing and workflow)
- LLM: OpenAI GPT-4o-mini (default), GPT-4o (optional upgrade for Tactical Analyst + Prediction agent)
- Embeddings: OpenAI text-embedding-3-small
- Vector Database: Qdrant (chosen over Elasticsearch — lighter footprint, better fit for 8GB RAM dev machine; Elasticsearch noted as a future option if moved to a bigger machine)
- Backend API: FastAPI (streaming responses via SSE)
- Frontend: Next.js / React
- Football Data: football-data.org (free tier)
- Monitoring: Postgres logging + Grafana dashboard
- Containers: Docker Compose (OrbStack recommended over Docker Desktop for lower memory overhead on 8GB machines)
- Hosting: local-first for now (docker-compose up), cloud deployment decision deferred until Phase 1 works end-to-end

## Evaluation Plan

- Build a ground-truth set of ~30-50 real questions and correct answers
- Measure retrieval quality (hit rate, MRR) for the Knowledge Agent
- Measure answer quality using an LLM-as-judge on a sample of answers
- Keep results in a simple table so progress is easy to track

## Roadmap

### Phase 1 — MVP (Building Now)
- Backend: Section router with match-checking, World Cup Agent, Knowledge Agent (Tactical + World Cup History)
- Backend: FastAPI with streaming responses
- Frontend: Next.js chat UI with section picker
- Evaluation: ground-truth set, retrieval metrics, LLM-as-judge
- Docker Compose setup

### Phase 2
- Club Football Agent (stats, comparisons)
- Tactical Analyst Agent
- Conversation memory
- Upgrade router to also support auto-detecting intent (no section picker needed)
- Frontend: stats visualizations, conversation history

### Phase 3
- Prediction Agent (deterministic scoring + LLM explanation)
- Fantasy Football Assistant
- News Agent
- Frontend: prediction cards with confidence breakdown

### Phase 4
- Polish and refine explainable predictions
- Interactive tactical visualizations
- Personalized notifications

## Future Work (Not Scheduled Yet)

- Player Scout Agent (needs playing-style data we don't have yet)
- Voice interface
- Transfer impact prediction ("What if Barcelona signs Rodri?")
- Full natural-language auto-routing (no section picker at all)

## Data Sources (To Be Finalized)

- Knowledge base content: starting with tactical concepts and World Cup history, rest marked TODO

## Future work - Smart tool selection (LLM-driven tool calling)
Agents currently fetch all relevant data sources upfront every time (standings, schedule, scorers), regardless of whether the question actually needs them — wasteful once there are more data sources per agent. Future upgrade: let the LLM decide which specific tools to call based on the question, instead of always fetching everything proactively in code.

## **Tactical Analyst — descoped (2026-07-23):** 

Originally planned as a dedicated agent/section. Design review found the differentiation from Knowledge Agent was too thin once we verified (via direct API testing) that detailed per-match statistic (possession, shots) aren't available on any free tier for historical matches — without that, "tactical analysis" reduced to RAG-over-concepts with a different prompt persona, which didn't justify a separate agent/section/router entry. Folded the "explain why, not just what" framing directly into Knowledge Agent's prompt instead.

## Future work -  Structured logging / observability — replace debug prints with configurable logger

- Python's built-in logging module, one logger = logging.getLogger(__name__) per file, replacing each print(...) with logger.debug(...) at the same call sites — same information, just now filterable.
- One central config (e.g. app/logging_config.py, wired up in main.py's startup) that reads a LOG_LEVEL env var (default INFO, set to DEBUG locally when you need the verbose output) — this is exactly the runtime on/off switch you're describing, no code changes needed to toggle it.
- Each LangGraph node logs at natural points (fast-path decision, extraction result, fetch result) the same way your current debug prints do — just via logger.debug(f"...") instead.
- Optional later step, since you already have Postgres+Grafana in your stack and did the Logfire→DuckDB dlt workshop recently: structured (JSON) log output could eventually feed the same observability story instead of being a separate concern — worth keeping in mind as a natural connection point, not something to build now.

## **Fantasy Assistant — descoped (2026-07-31):**

Originally planned as a dedicated agent (Phase 3 scope, per Roadmap above). Descoped after the same "verify data availability before designing" discipline used for Tactical Analyst: Fantasy Assistant's spec fundamentally requires player-level data — injuries, expected minutes, individual player form — but football-data.org's free tier has never shown any sign of providing this; every tool built so far (`football_api.py`) only ever pulls team-level standings, results, and basic top-scorer stats, nothing at the player-status granularity a fantasy assistant needs to be useful. Without real player-level data, a Fantasy Assistant would either have to fabricate plausible-sounding injury/rotation info (directly against the project's core "never let the LLM invent facts" principle, see Prediction Agent design) or be so thin it wouldn't justify a dedicated agent — same shape of problem, same conclusion, as Tactical Analyst. No workaround pursued since no free-tier API surfaced player-level status data during Phase 3 research. If a suitable data source is found later, this could be revisited as future work rather than reopened as in-scope Phase 3 work.
