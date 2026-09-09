"""Unit tests for should_continue's tool-loop routing, including the
MAX_TOOL_ITERATIONS hard cap.

should_continue only reads state (doesn't touch the tools list itself), and
make_nodes([]) initializes quickly (memory client init only), so these run
without any real Anthropic API call.
"""
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import MAX_TOOL_ITERATIONS, make_nodes

_, _, _, should_continue, _ = make_nodes([])


def _state(last_message, tool_call_count=0):
    return {"messages": [HumanMessage(content="q"), last_message], "tool_call_count": tool_call_count}


def test_no_tool_calls_goes_to_save_memories():
    state = _state(AIMessage(content="here's your answer"))
    assert should_continue(state) == "save_memories"


def test_tool_call_under_cap_goes_to_tools():
    state = _state(
        AIMessage(content="", tool_calls=[{"name": "garmin_sleep", "args": {}, "id": "1"}]),
        tool_call_count=MAX_TOOL_ITERATIONS - 1,
    )
    assert should_continue(state) == "tools"


def test_tool_call_at_cap_forces_save_memories():
    state = _state(
        AIMessage(content="", tool_calls=[{"name": "garmin_sleep", "args": {}, "id": "1"}]),
        tool_call_count=MAX_TOOL_ITERATIONS,
    )
    assert should_continue(state) == "save_memories"


def test_tool_call_over_cap_forces_save_memories():
    state = _state(
        AIMessage(content="", tool_calls=[{"name": "garmin_sleep", "args": {}, "id": "1"}]),
        tool_call_count=MAX_TOOL_ITERATIONS + 3,
    )
    assert should_continue(state) == "save_memories"
