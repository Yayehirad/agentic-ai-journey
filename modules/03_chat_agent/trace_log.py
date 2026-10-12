"""Module 03: a tiny trace logger. Appends one JSON object per line (JSONL)."""
import json
from datetime import datetime
from pathlib import Path

# logs/trace.jsonl next to this file, wherever you run the command from
LOG_FILE = Path(__file__).parent / "logs" / "trace.jsonl"


def log_event(kind: str, data: dict) -> None:
    """Add one line to the trace log, e.g. kind="model_reply" or "tool_call"."""
    LOG_FILE.parent.mkdir(exist_ok=True)   # create logs/ the first time
    record = {"time": datetime.now().isoformat(timespec="seconds"),
              "kind": kind, **data}        # ** copies data's keys into record
    # "a" = append mode: add to the end, never erase earlier lines.
    # default=str turns anything JSON doesn't understand into text, so
    # logging can never crash the chat.
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
