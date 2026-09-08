"""FastAPI backend for the Garmin Training Coach.

Startup sequence:
  1. Sync latest Garmin data (today + yesterday)
  2. Connect to Garmin MCP server (stdio subprocess via langchain-mcp-adapters)
  3. Load custom analysis tools
  4. Compile the LangGraph agent with all tools
  5. Serve /chat and /health endpoints

Run with:
    uvicorn api.main:app --reload
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from agent.graph import create_graph
from agent.tools import get_custom_tools

# ── MCP server config ─────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_GARMIN_DIR = _REPO_ROOT / "garmin-givemydata"
_PYTHON = _GARMIN_DIR / "venv" / "Scripts" / "python.exe" if sys.platform == "win32" else _GARMIN_DIR / "venv" / "bin" / "python"

# Make garmin_mcp and garmin_client importable from the main process.
# GARMIN_DATA_DIR must be set before garmin_mcp.db is first imported because
# it resolves DB_PATH at module load time.
os.environ.setdefault("GARMIN_DATA_DIR", str(_GARMIN_DIR))
if str(_GARMIN_DIR) not in sys.path:
    sys.path.insert(0, str(_GARMIN_DIR))

_GARMIN_MCP_URL = os.environ.get("GARMIN_MCP_URL")

if _GARMIN_MCP_URL:
    # Docker: MCP server runs in its own container, reachable over the network.
    GARMIN_MCP = {
        "url": _GARMIN_MCP_URL,
        "transport": "streamable_http",
    }
else:
    # Local dev: spawn the MCP server as a stdio subprocess (unchanged behavior).
    GARMIN_MCP = {
        "command": str(_PYTHON),
        "args": [str(_GARMIN_DIR / "run_mcp.py")],
        "transport": "stdio",
        "cwd": str(_GARMIN_DIR),
        "env": {"GARMIN_DATA_DIR": str(_GARMIN_DIR)},
    }


# ── Lifespan ──────────────────────────────────────────────────────────────────

async def _auto_sync():
    """Run incremental Garmin sync directly in-process.

    Imports incremental_sync() from garmin_mcp and runs it in a thread pool
    so the event loop is never blocked. Fired via asyncio.create_task —
    runs concurrently with MCP setup. Errors are logged, not raised.
    """
    print("[startup] syncing latest Garmin data...")
    try:
        from garmin_mcp.sync import incremental_sync
        result = await asyncio.wait_for(
            asyncio.to_thread(incremental_sync, skip_if_synced_today=True), timeout=300
        )
        if result.get("status") == "ok":
            print(f"[startup] sync complete — {result.get('total_upserted', 0)} records upserted")
        elif result.get("status") == "skipped":
            print(f"[startup] {result.get('message')}")
        else:
            print(f"[startup] sync failed: {result.get('message', result)}")
    except asyncio.TimeoutError:
        print("[startup] sync timed out after 5 minutes")
    except Exception as e:
        print(f"[startup] sync skipped ({type(e).__name__}: {e})")

"""
When FastAPI starts up, it runs the lifespan function in api/main.py:62. 
This is where the expensive setup happens — connecting to the MCP server, loading tools, compiling the LangGraph agent. 
You don't want to redo this on every request, so you store the result somewhere persistent.
"app.state" is that persistent storage. It's just a simple object where you can attach anything
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("ENABLE_AUTO_SYNC", "true").lower() == "false":
        print("[startup] auto-sync disabled (ENABLE_AUTO_SYNC=false)")
    else:
        asyncio.create_task(_auto_sync())  # fire and forget — runs concurrently with MCP setup

    custom_tools = get_custom_tools()

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        # v0.1.0+ — no longer a context manager; create client and await get_tools()
        mcp_client = MultiServerMCPClient({"garmin": GARMIN_MCP}) #Create the MCP client and tell it about the Garmin subprocess configuration.
        mcp_tools = await mcp_client.get_tools() # Actually launch the Garmin MCP subprocess and ask it "what tools do you have?"
        app.state.mcp_client = mcp_client  # Store the MCP client on app.state so it stays alive for the entire server lifetime. 
        print(f"[startup] loaded {len(mcp_tools)} Garmin MCP tools + {len(custom_tools)} custom tools")
        app.state.graph = create_graph(mcp_tools + custom_tools) #Combine all 44 Garmin MCP tools + 3 custom tools into one list and compile the LangGraph agent.
    except Exception as e:
        print(f"[startup] Garmin MCP unavailable ({e}) — running with custom tools only")
        app.state.graph = create_graph(custom_tools)

    yield #Everything before yield = setup. The server doesn't accept any requests until yield is reached.


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Garmin Training Coach API", version="0.1.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────
"""
Since every request currently defaults to "default", in practice this app treats every user as the same one person — 
there's no real multi-user separation yet unless someone starts passing a different user_id.
"""
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    response: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

# only returns once everything is done
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    graph = app.state.graph
    config = {"configurable": {"thread_id": request.user_id}}

    try:
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": request.message}],
                "user_id": request.user_id,
                "memories": [],
            },
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return the last non-tool-call AI message
    for msg in reversed(result.get("messages", [])):
        if (
            isinstance(msg, AIMessage)
            and isinstance(msg.content, str)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            return ChatResponse(response=msg.content)

    raise HTTPException(status_code=500, detail="Agent produced no response")

# GET is "give me something that already exists." POST is "here's my input, now produce something." 
# Since Claude's response only exists after you send a message, it's POST.
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    graph = app.state.graph
    config = {"configurable": {"thread_id": request.user_id}}

    async def generate():
        emitted_text = False
        try:
            async with asyncio.timeout(90):
                async for event in graph.astream_events(
                    {
                        # the langgraph abstract away the process of storing the user prompt into the state message
                        "messages": [{"role": "user", "content": request.message}],
                        "user_id": request.user_id,
                        "memories": [],
                    },
                    config=config,
                    version="v2",
                ):
                    kind = event["event"]
                    if kind == "on_tool_start" and not emitted_text:
                        # Emit a status line so the UI isn't blank during Garmin MCP calls.
                        # Skipped once real text is flowing to avoid injecting status mid-response.
                        tool_name = event.get("name", "tool")
                        yield f"*Fetching data via `{tool_name}`…*\n\n"
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content
                        if isinstance(content, str) and content:
                            emitted_text = True
                            yield content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        emitted_text = True
                                        yield text
        except asyncio.TimeoutError:
            print("[stream] timed out after 90s")
            yield "\n\n**Timed out** — the Garmin MCP tool took too long to respond."
        except Exception as e:
            print(f"[stream] error: {e}")
            yield f"\n\n**Error:** `{e}`"
    return StreamingResponse(generate(), media_type="text/plain")
