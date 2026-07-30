"""
The shared state that flows through every node in the graph.
"""
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str
    section: Optional[str]  # now determined by the classifier node, not sent by the frontend
    section_valid: Optional[bool]
    answer: Optional[str]
    messages: Annotated[list, add_messages]