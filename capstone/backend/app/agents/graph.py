"""
The actual LangGraph graph — wires the router and both agents into one
runnable pipeline.
"""
from langgraph.graph import StateGraph, START, END

from app.agents.state import GraphState
from app.agents.router import check_section_match
from app.agents.world_cup_agent import world_cup_agent_node
from app.agents.knowledge_agent import knowledge_agent_node


def route_after_check(state: GraphState) -> str:
    """
    Conditional edge function: LangGraph calls this right after the router
    node finishes. Whatever string it returns must be a key in the path_map
    passed to add_conditional_edges() below — that's how LangGraph decides
    which node runs next. Equivalent to the condition expression in a
    Kestra `switch` task.
    """
    if not state["section_valid"]:
        return "mismatch"       # router already wrote the mismatch message into "answer"
    return state["section"]     # "world_cup" or "knowledge" — routes straight to the matching agent


def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes: just associates a name with the function that implements it
    graph.add_node("check_section_match", check_section_match)
    graph.add_node("world_cup", world_cup_agent_node)
    graph.add_node("knowledge", knowledge_agent_node)

    # Every run starts at the router
    graph.add_edge(START, "check_section_match")

    # Branch based on route_after_check()'s return value
    graph.add_conditional_edges(
        "check_section_match",
        route_after_check,
        path_map={
            "mismatch": END,          # answer already set, nothing left to do
            "world_cup": "world_cup",
            "knowledge": "knowledge",
        },
    )

    # Both agents are terminal — once they produce an answer, the graph ends
    graph.add_edge("world_cup", END)
    graph.add_edge("knowledge", END)

    return graph.compile()  # validates the graph (all edges point to real nodes) and returns a runnable


# Compiled once at import time, reused across every request
soccermind_graph = build_graph()
