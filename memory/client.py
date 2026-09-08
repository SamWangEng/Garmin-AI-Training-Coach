"""Memory management with three tiers: mem0 hosted → mem0 local → JSON fallback.

The MemoryManager class provides a consistent interface regardless of backend.
"""
import json
import os
from pathlib import Path


# ── Fallback ──────────────────────────────────────────────────────────────────

class SimpleMemory:
    """Keyword-matched JSON file memory. Works with zero external dependencies."""

    def __init__(self, path: str = ".coach_memory.json"):
        self.path = Path(path)
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def add(self, messages: list[dict], user_id: str = "default") -> None:
        bucket = self._data.setdefault(user_id, [])
        for m in messages:
            content = m.get("content", "")
            if content and m.get("role") == "user":
                bucket.append(content)
        self._data[user_id] = bucket[-100:]  # keep last 100 entries
        self._save()

    def search(self, query: str, user_id: str = "default", limit: int = 5) -> dict:
        entries = self._data.get(user_id, [])
        query_words = set(query.lower().split())
        scored = [
            (len(query_words & set(e.lower().split())), e)
            for e in entries
        ]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return {"results": [{"memory": e} for _, e in scored[:limit]]}


# ── Unified manager ────────────────────────────────────────────────────────────

class MemoryManager:
    """Tries mem0 hosted → mem0 local (Anthropic LLM) → SimpleMemory fallback."""

    def __init__(self) -> None:
        self._client, self._mode = self._init()
        print(f"[memory] using backend: {self._mode}")

    def _init(self) -> tuple:
        # Tier 1: mem0 hosted
        # This is a thin HTTP client. Your app sends raw text to mem0's cloud service
        if os.getenv("MEM0_API_KEY"):
            try:
                from mem0 import MemoryClient
                client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
                return client, "mem0_hosted"
            except Exception as e:
                print(f"[memory] mem0 hosted init failed: {e}")

        # Tier 2: mem0 local with Anthropic LLM
        # The actual code in mem0/memory/main.py
        try:
            from mem0 import Memory
            config = {
                "llm": {
                    "provider": "anthropic",
                    "config": {
                        "model": "claude-haiku-4-5-20251001",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                }
            }
            client = Memory.from_config(config)
            return client, "mem0_local"
        except Exception as e:
            print(f"[memory] mem0 local init failed: {e}")

        # Tier 3: simple JSON fallback
        return SimpleMemory(), "simple"

    # Storing the memory
    def add(self, messages: list[dict], user_id: str = "default") -> None:
        try:
            self._client.add(messages, user_id=user_id)
        except Exception as e:
            print(f"[memory] add failed: {e}")

    # Retreiving the memory
    def search(self, query: str, user_id: str = "default", limit: int = 5) -> dict:
        try:
            if self._mode == "mem0_hosted":
                # mem0 v2 hosted API requires filters dict, not just user_id kwarg
                results = self._client.search(
                    query,
                    filters={"AND": [{"user_id": user_id}]},
                    limit=limit,
                )
            else:
                results = self._client.search(query, user_id=user_id, limit=limit)

            # MemoryClient returns a list directly; normalise to dict
            if isinstance(results, list):
                return {"results": [{"memory": r.get("memory", str(r))} for r in results[:limit]]}
            return results
        except Exception as e:
            print(f"[memory] search failed: {e}")
            return {"results": []}

    def search_for_user(self, query: str, user_id: str) -> list[str]:
        """Search memories and always return a plain list of strings."""
        results = self.search(query, user_id=user_id)
        if not isinstance(results, dict):
            return []
        return [r["memory"] for r in results.get("results", []) if isinstance(r, dict) and "memory" in r]

    @property
    def mode(self) -> str:
        return self._mode


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: MemoryManager | None = None


def get_memory_client() -> MemoryManager:
    global _instance
    if _instance is None:
        _instance = MemoryManager()
    return _instance
