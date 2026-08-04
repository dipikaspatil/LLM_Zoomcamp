"""
News Agent node — fetches recent, real news articles for a team via
NewsAPI (deterministic, filtered query), then has the LLM summarize
them. Same two-LLM-call shape as the Prediction Agent: an extraction
call (tagged so it doesn't leak into the SSE stream) and an
explanation call. The LLM never invents headlines — it only
summarizes the real articles handed to it.
"""
from datetime import date

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState
from app.tools.news_api import get_team_news

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)


class TeamExtraction(BaseModel):
    team: str = Field(description="The football club or national team the question is about")


_extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)
_team_extractor = _extractor_llm.with_structured_output(TeamExtraction)


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for article in articles:
        title = article["title"]
        source = article["source"]["name"]
        published = article["publishedAt"][:10]  # YYYY-MM-DD
        description = article.get("description") or ""
        url = article["url"]
        lines.append(f"- [{published}] {title} ({source})\n  {description}\n  {url}")
    return "\n".join(lines)


async def news_agent_node(state: GraphState) -> dict:
    today = date.today().isoformat()

    extraction = await _team_extractor.ainvoke(
        f"Question: {state['question']}\nWhich football team is this question about?",
        config={"tags": ["extraction_only"]},
    )

    articles = await get_team_news(extraction.team)
    # TODO: replace with logger.debug() — see Future work - Structured logging / observability from Phase_0.md
    # print(f"[DEBUG] team={extraction.team!r} articles={[a['title'] for a in articles]}")

    if not articles:
        answer = (
            f"I couldn't find any recent news headlines about {extraction.team} "
            "from my sources right now — try asking again later, or check a "
            "sports news site directly."
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    articles_text = _format_articles(articles)

    system_prompt = (
        f"Today's date is {today}.\n"
        "You are a football news assistant. Summarize the real news articles "
        "below for the user in a clear, conversational way. Only report what "
        "the articles actually say — do not add speculation or facts not "
        "present in the text. If articles disagree or seem outdated, note "
        "that. Include the source name so the user knows where each piece of "
        "news comes from. Do not include the raw URLs in your answer.\n\n"
        f"Recent articles about {extraction.team}:\n{articles_text}"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["question"])]
    response = await _llm.ainvoke(messages)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}
