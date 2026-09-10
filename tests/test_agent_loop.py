"""Integration eval: does the actual compiled graph (agent/graph.py), not just
should_continue in isolation, actually stop at MAX_TOOL_ITERATIONS?

test_should_continue.py already proves should_continue's own logic is correct
in isolation. This proves the wiring holds end-to-end: a model that keeps
requesting the same tool forever must not make the real graph.ainvoke() call
hang or loop unboundedly.

ChatAnthropic.ainvoke is mocked to always request a tool call, so this is
deterministic and needs no real Anthropic API call — it isn't testing whether
Claude *would* loop, only that the graph enforces the cap if it does.
"""
import itertools
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.graph import create_graph
from agent.nodes import MAX_TOOL_ITERATIONS


@tool
def fake_tool(query: str) -> str:
    """A fake tool that always returns the same small result."""
    return "some data"


def _always_calls_tool(*_args, **_kwargs) -> AIMessage:
    # A fresh AIMessage every call — reusing one shared object would let
    # add_messages' dedup-by-id logic collapse repeated "new" responses into
    # updates of the same message instead of appending distinct ones.
    call_id = f"call_{next(_call_ids)}"
    return AIMessage(content="", tool_calls=[{"name": "fake_tool", "args": {"query": "x"}, "id": call_id}])


_call_ids = itertools.count() # gives you 0, then 1, then 2, then 3


async def test_graph_stops_at_max_tool_iterations():
    with (
        patch("agent.nodes.get_memory_client") as mock_mc,
        patch("agent.nodes.ChatAnthropic.ainvoke", new=AsyncMock(side_effect=_always_calls_tool)),
        patch("agent.nodes.route_tools", new=AsyncMock(return_value=[])),
    ):
        mock_mc.return_value.search_for_user.return_value = []
        graph = create_graph([fake_tool])

        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "test"}], "user_id": "test", "memories": []},
            config={"configurable": {"thread_id": "test-loop-cap"}},
        )

    assert result["tool_call_count"] <= MAX_TOOL_ITERATIONS

    # The model always requests fake_tool, so the cap should actually have
    # been hit, not just happen to be under it — otherwise this test would
    # pass even if the cap logic were silently broken.
    assert result["tool_call_count"] == MAX_TOOL_ITERATIONS


async def test_tool_call_count_resets_between_turns():
    """tool_call_count has no reducer, so without an explicit reset in
    retrieve_memories it would silently accumulate across the whole session
    instead of counting just the current turn — eventually locking out tool
    use permanently once the running total crosses MAX_TOOL_ITERATIONS.
    """
    call_num = itertools.count(1) # start at 1

    def one_tool_call_then_answer(*_a, **_k):
        # Odd calls request the tool, even calls answer plainly — so each
        # "turn" is exactly one tool round-trip.
        n = next(call_num)
        if n % 2 == 1:
            return AIMessage(content="", tool_calls=[{"name": "fake_tool", "args": {"query": "x"}, "id": f"call_{n}"}])
        return AIMessage(content="here is your answer")

    with (
        patch("agent.nodes.get_memory_client") as mock_mc,
        patch("agent.nodes.ChatAnthropic.ainvoke", new=AsyncMock(side_effect=one_tool_call_then_answer)),
        patch("agent.nodes.route_tools", new=AsyncMock(return_value=[])),
    ):
        mock_mc.return_value.search_for_user.return_value = []
        graph = create_graph([fake_tool])
        config = {"configurable": {"thread_id": "cross-turn-reset-test"}}

        r1 = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "turn 1"}], "user_id": "test", "memories": []}, config=config
        )
        r2 = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "turn 2"}], "user_id": "test", "memories": []}, config=config
        )

    assert r1["tool_call_count"] == 1
    assert r2["tool_call_count"] == 1  # must reset — not 2, which would mean it's accumulating across turns
