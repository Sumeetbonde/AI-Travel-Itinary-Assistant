import threading
from typing import Optional, Dict, Any
import json
import os

# Thread-safe in-memory itinerary memory
_lock = threading.Lock()
_itinerary_memory: Optional[Dict[str, Any]] = None

# Default persistence file (relative to repo). You can change this if desired.
_persist_file = os.path.join("data", "itinerary_memory.json")


def _ensure_persist_dir():
    d = os.path.dirname(_persist_file)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


def _save_to_disk() -> None:
    try:
        _ensure_persist_dir()
        with _lock:
            if _itinerary_memory is None:
                # remove file if exists
                try:
                    if os.path.exists(_persist_file):
                        os.remove(_persist_file)
                except Exception:
                    pass
                return
            with open(_persist_file, "w", encoding="utf-8") as f:
                json.dump(_itinerary_memory, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal; don't raise from internal persistence
        return


def _load_from_disk() -> None:
    global _itinerary_memory
    try:
        if os.path.exists(_persist_file):
            with open(_persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _lock:
                _itinerary_memory = data
    except Exception:
        # ignore load errors
        _itinerary_memory = None


_load_from_disk()


def set_itinerary(itinerary: Dict[str, Any]) -> None:
    """Store a copy of the current itinerary in memory and persist to disk."""
    global _itinerary_memory
    with _lock:
        _itinerary_memory = dict(itinerary) if itinerary is not None else None
    _save_to_disk()


def get_itinerary() -> Optional[Dict[str, Any]]:
    """Return a shallow copy of the stored itinerary or None."""
    with _lock:
        return None if _itinerary_memory is None else dict(_itinerary_memory)


def clear_itinerary() -> None:
    """Clear stored itinerary memory (and delete persisted file)."""
    global _itinerary_memory
    with _lock:
        _itinerary_memory = None
    _save_to_disk()


def update_itinerary_sections(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply partial updates to stored itinerary sections and return the changed keys.

    Example: update_itinerary_sections({'hotels': {...}, 'budget': 40000})
    """
    global _itinerary_memory
    changed = {}
    with _lock:
        if _itinerary_memory is None:
            _itinerary_memory = {}
        for key, value in updates.items():
            _itinerary_memory[key] = value
            changed[key] = value
    _save_to_disk()
    return changed


def get_itinerary_memory_prompt() -> str:
    """Return a compact textual summary of the stored itinerary suitable for prepending to prompts.

    Returns empty string when no memory is stored.
    """
    it = get_itinerary()
    if not it:
        return ""
    lines = []
    for k, v in it.items():
        if isinstance(v, (str, int, float)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: [present]")
    return "Current Itinerary Memory:\n" + "\n".join(lines) + "\n\n"


def get_travel_copilot_instructions() -> str:
    """Return the travel-copilot rules and response-style instructions.

    These instructions are intended to be prepended to prompts sent to the LLM
    so the assistant consistently follows the user's requested copilot behavior.
    """
    return (
        "You are an AI Travel Assistant.\n\n"
        "The user already has a complete travel itinerary. Your job is to act as a travel copilot and answer follow-up questions about the generated plan.\n\n"
        "Rules:\n"
        "1. Always use the existing itinerary as the primary source of context.\n"
        "2. Modify only the sections relevant to the user's request instead of regenerating the entire itinerary.\n"
        "3. If the user asks for changes (budget, destinations, duration, hotels, transport, activities, food, weather, etc.), provide an updated version of only the affected sections.\n"
        "4. Preserve previously generated information unless the requested change impacts it.\n"
        "5. Show cost implications whenever a modification affects the budget.\n"
        "6. If information is unavailable, make reasonable travel recommendations and clearly mark them as [Suggested].\n"
        "7. Be conversational and concise for follow-up questions.\n"
        "8. When appropriate, provide actionable recommendations, alternatives, and trade-offs.\n"
        "9. Maintain a memory of the current itinerary throughout the conversation.\n"
        "10. Do not repeat the full itinerary unless explicitly requested.\n\n"
        "Response Style:\n"
        "- Show only the updated sections.\n"
        "- Explain the impact of the change (one short paragraph).\n"
        "- Provide revised cost estimates when applicable.\n"
        "- Keep responses structured and easy to scan.\n"
        "- Keep responses concise and conversational.\n"
        "- Mark any non-PDF recommendations as [Suggested].\n"
    )
