"""
The actual LangGraph graph — wires the section classifier and all agents
into one runnable pipeline.
"""
from langgraph.graph import StateGraph, START, END

from app.agents.state import GraphState
from app.agents.router import classify_section
from app.agents.world_cup_agent import world_cup_agent_node
from app.agents.knowledge_agent import knowledge_agent_node
from app.agents.club_football_agent import club_football_agent_node
from app.agents.prediction_agent import prediction_agent_node
from app.agents.news_agent import news_agent_node

def route_after_classification(state: GraphState) -> str:
    if not state["section_valid"]:
        return "none"
    return state["section"]


def build_graph(checkpointer):
    """
    Takes the checkpointer as a parameter instead of creating one
    internally, so the graph-building logic is reusable regardless of
    which checkpointer backend is wired in (Postgres in production).
    """
    graph = StateGraph(GraphState)

    graph.add_node("classify_section", classify_section)
    graph.add_node("world_cup", world_cup_agent_node)
    graph.add_node("knowledge", knowledge_agent_node)
    graph.add_node("club_football", club_football_agent_node)
    graph.add_node("prediction", prediction_agent_node)
    graph.add_node("news", news_agent_node)
    graph.add_edge("news", END)

    graph.add_edge(START, "classify_section")

    graph.add_conditional_edges(
        "classify_section",
        route_after_classification,
        path_map={
            "world_cup": "world_cup",
            "club_football": "club_football",
            "prediction": "prediction",
            "news": "news",
            "knowledge": "knowledge",
            "none": END
        },
    )

    graph.add_edge("world_cup", END)
    graph.add_edge("knowledge", END)
    graph.add_edge("club_football", END)
    graph.add_edge("prediction", END)

    return graph.compile(checkpointer=checkpointer)
