"""Shared TypedDicts for agent node input/output contracts."""
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

"""
"whenever any node returns a new value for messages, don't just overwrite the field — call add_messages(old_value, new_value) and use that as the new state." 
The actual mechanics of combining (append vs. replace-by-id, as we saw in add_messages's code) are hidden inside the reducer function itself, supplied by the library.
"""
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # in-memory chat history for the current session
    user_id: str
    memories: list[str]  # long-term facts retrieved from mem0 that persist across sessions
    tool_call_count: int  # rounds through the tools node this turn — no reducer, plain overwrite


# Return types for each node — forces correct key names at the call site
class MemoryUpdate(TypedDict):
    memories: list[str]


class ModelUpdate(TypedDict):
    messages: list[BaseMessage]
    tool_call_count: int
