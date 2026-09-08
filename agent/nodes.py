"""LangGraph node functions for the training coach agent."""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langgraph.prebuilt import ToolNode

from memory.client import get_memory_client
from agent.types import MemoryUpdate, ModelUpdate
from agent.router import route_tools, select_tools

# Cap on conversation history sent to the model per call. Tuned empirically —
# adjust based on observed [tokens] cost/latency, not a hard model limit.
MAX_HISTORY_TOKENS = 8000

SYSTEM_PROMPT = """\
You are a personal AI running coach with direct access to the user's Garmin Connect data.

You have Garmin MCP tools to fetch real workout history, heart rate, sleep, HRV, recovery scores,
training load, and more. You also have custom analysis tools for trends, PRs, and week comparisons.

When answering questions:
1. Query the appropriate Garmin tools to fetch data (garmin_today, garmin_activities, garmin_sleep, etc.). The database is kept up to date automatically — do NOT call garmin_sync unless the user explicitly asks to sync or says the data looks outdated.
2. Consider the full picture: training load, sleep quality, HRV, stress, and recovery scores.
3. Give specific, actionable coaching advice — not generic fitness tips.
4. Reference personal context (injuries, goals, past feelings) from memory when relevant.

Be direct and coach-like. Lead with the insight, then back it up with data.\
"""


def make_nodes(tools: list) -> tuple:
    """Return all node functions and the tool node, closed over the given tools."""
    memory_client = get_memory_client()

    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    tool_node = ToolNode(tools)

    async def retrieve_memories(state: dict) -> MemoryUpdate:
        user_id = state.get("user_id", "default")
        messages = state.get("messages", [])

        query = ""
        for m in reversed(messages):  # most recent message first
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                query = m.content
                break

        if not query:
            return MemoryUpdate(memories=[])

        try:
            memories = memory_client.search_for_user(query, user_id=user_id)
        except Exception as e:
            print(f"[memory] retrieve failed (non-critical): {e}")
            memories = []
        return MemoryUpdate(memories=memories)

    async def call_model(state: dict) -> ModelUpdate:
        memories = state.get("memories", [])

        # Static prompt is marked for caching — Anthropic reuses it across requests.
        # Dynamic memories are a separate block with no cache_control so they're
        # always sent fresh (they change per user and per conversation).
        system_blocks = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]
        if memories:
            system_blocks.append({
                "type": "text",
                "text": "\n\n## Personal context from previous conversations:\n"
                        + "\n".join(f"- {m}" for m in memories),
            })

        # Route: classify the user's intent and bind only the relevant tools.
        user_message = ""
        for m in reversed(list(state["messages"])):
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                user_message = m.content
                break

        categories = await route_tools(user_message) if user_message else list()
        routed_tools = select_tools(tools, categories) if categories else tools
        print(f"[router] categories={categories} tools_selected={len(routed_tools)}/{len(tools)}")

        routed_model = model.bind_tools(routed_tools) #attach to model (no network call, no tool execution)

        # Cap history sent to the model — state["messages"] keeps growing every turn
        # (that's the short-term memory), but resending all of it on every tool-call
        # round-trip gets expensive. start_on="human" avoids leaving a dangling
        # ToolMessage/tool_use fragment at the front after trimming.
        history = trim_messages(
            state["messages"],
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter="approximate",
            strategy="last",
            start_on="human",
        )
        if len(history) < len(state["messages"]):
            print(f"[trim] kept {len(history)}/{len(state['messages'])} messages (cap={MAX_HISTORY_TOKENS} tokens)")

        messages = [SystemMessage(content=system_blocks)] + history
        response = await routed_model.ainvoke(messages)

        # Token usage logging — input/output plus cache write/read for caching visibility.
        usage = response.usage_metadata
        if usage:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            print(
                f"[tokens] input={usage['input_tokens']} "
                f"output={usage['output_tokens']} "
                f"cache_write={cache_write} "
                f"cache_read={cache_read} "
                f"tools_bound={len(routed_tools)}"
            )
        #  It's this return value that then goes through the add_messages reducer we discussed, merging just the new AI response onto the real, persisted state["messages"].
        return ModelUpdate(messages=[response])

    async def save_memories(state: dict) -> dict:
        user_id = state.get("user_id", "default")
        messages = state.get("messages", [])

        to_save = []
        for m in messages[-4:]:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                to_save.append({"role": "user", "content": m.content})
            elif (
                isinstance(m, AIMessage)
                and isinstance(m.content, str)
                and m.content
                and not getattr(m, "tool_calls", None)
            ):
                to_save.append({"role": "assistant", "content": m.content})

        if to_save:
            try:
                memory_client.add(to_save, user_id=user_id)
            except Exception as e:
                print(f"[memory] save failed (non-critical): {e}")

        return {}

    def should_continue(state: dict) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "save_memories"

    return retrieve_memories, call_model, save_memories, should_continue, tool_node
