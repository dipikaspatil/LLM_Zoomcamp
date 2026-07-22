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
