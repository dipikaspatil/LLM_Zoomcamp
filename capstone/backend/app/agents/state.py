"""
The shared state that flows through every node in the graph.
Kestra equivalent: the flow's execution context/variables — except here
it's an explicit, typed Python structure instead of implicit YAML variables.
"""
from typing import TypedDict, Optional


class GraphState(TypedDict):
    question: str                    # the user's raw question
    section: str                     # section picked in the UI: "world_cup" or "knowledge"
    section_valid: Optional[bool]    # filled in by the router node
    answer: Optional[str]            # final answer text — set by the router (on mismatch) or an agent node
