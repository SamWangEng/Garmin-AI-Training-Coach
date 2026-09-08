"""Unit tests for the history-trimming logic in agent/nodes.py's call_model.

trim_messages is a pure function (no LLM call), so these run without an
ANTHROPIC_API_KEY and are fast/deterministic — safe to run in CI.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, trim_messages

from agent.nodes import MAX_HISTORY_TOKENS


def _trim(messages):
    return trim_messages(
        messages,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
    )


def test_trim_no_op_when_under_budget():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello there")]
    assert _trim(messages) == messages


def test_trim_drops_oldest_messages_over_budget():
    # ~4 chars/token approximation — 40,000 chars is ~10k tokens, over the 8k cap
    old = HumanMessage(content="x" * 40_000)
    recent = HumanMessage(content="what's my VO2max?")
    trimmed = _trim([old, AIMessage(content="ok"), recent])

    assert old not in trimmed
    assert trimmed[-1] == recent


def test_trim_never_orphans_a_tool_result():
    """A pure token-count cut could land mid tool_call/tool_result pair.

    Sized so the naive budget cut alone would keep [ToolMessage, AIMessage,
    HumanMessage] — a dangling tool result up front. start_on="human" must
    trim further back until the result starts on a HumanMessage instead.
    """
    recent = HumanMessage(content="z" * 20_000)  # ~5,000 tokens
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "garmin_sleep", "args": {"pad": "a" * 20_000}, "id": "call_1"}]),
        ToolMessage(content="x" * 4_000, tool_call_id="call_1"),  # ~1,000 tokens
        AIMessage(content="y" * 4_000),  # ~1,000 tokens
        recent,
    ]
    trimmed = _trim(messages)

    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[0] is recent


def test_trim_keeps_full_history_when_short():
    messages = [HumanMessage(content=f"message {i}") for i in range(5)]
    assert _trim(messages) == messages
