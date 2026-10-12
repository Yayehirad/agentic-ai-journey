"""Module 05: the Module 03 chat agent, with a search_notes tool over my PDFs."""
import json
from datetime import datetime
from pathlib import Path

from ollama import chat

from search import search   # search.py next to this file

MODEL = "qwen3.5:9b"
MAX_STEPS = 5      # safety cap per turn: never loop forever
MAX_HISTORY = 20   # how many non-system messages the model gets to see
LOG_FILE = Path(__file__).parent / "logs" / "trace.jsonl"

# The rules for citing live here. The model only sees the labels
# "[file, page N]" in the search results, so it can copy them exactly.
SYSTEM_PROMPT = (
    "You are a study assistant that answers questions from the user's PDF notes. "
    "You do NOT know what is in the notes until you call search_notes, so for "
    "any question about a topic or fact, call search_notes before answering. "
    "Answer only from the search results, not from your own knowledge. "
    "After each fact, cite its source as (file name, page N), copying the "
    "[file, page N] label shown above that passage. Never invent a file name "
    "or page number. If the results don't contain the answer, say you "
    "couldn't find it in the notes. Keep answers short."
)


# ---------- The tool ----------
def search_notes(question: str) -> str:
    """Search the user's PDF notes for passages related to a question.

    Args:
      question: What to look for, for example "what is gradient descent"
    """
    hits = search(question)
    if not hits:
        return "No notes found. The user needs to run ingest.py first."
    # Put a "[file, page N]" label on every chunk, so the model can cite it
    return "\n\n".join(f"[{h['file']}, page {h['page']}]\n{h['text']}"
                       for h in hits)


TOOLS = {"search_notes": search_notes}


# ---------- Trace log (same idea as Module 03's trace_log.py) ----------
def log_event(kind: str, data: dict) -> None:
    """Add one JSON line to logs/trace.jsonl."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"),
              "kind": kind, **data}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------- Memory (unchanged from Module 03) ----------
def trim_history(messages: list) -> list:
    """Keep the system prompt plus only the most recent messages."""
    system, rest = messages[0], messages[1:]
    recent = rest[-MAX_HISTORY:]
    # Don't start in the middle of a tool exchange: begin at a user message
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    if not recent:
        last_user = max(i for i, m in enumerate(rest) if m["role"] == "user")
        recent = rest[last_user:]
    return [system] + recent


# ---------- The agent loop (unchanged from Module 03) ----------
def run_agent(messages: list, model: str = MODEL) -> str:
    for step in range(MAX_STEPS):
        messages[:] = trim_history(messages)

        response = chat(model=model, messages=messages,
                        tools=list(TOOLS.values()), think=False)
        messages.append(response.message)
        log_event("model_reply", response.message.model_dump(exclude_none=True))

        # No tool requested? Then this is the final answer. The model
        # DECIDES whether to search; we don't force it.
        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            fn = TOOLS.get(name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Error: {e}"
            # Search results are long, so only show the start on screen.
            # The full text still goes to the model and the log.
            print(f"[tool] {name}({args}) -> {str(result)[:150]}...")
            log_event("tool_call", {"name": name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Stopped: hit the step limit."


# ---------- The chat loop (unchanged from Module 03) ----------
def main() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Ask about your PDFs. Type 'quit' to stop.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_text.lower() == "quit":
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        print(f"Agent: {run_agent(messages)}")

    print("\nBye!")


if __name__ == "__main__":
    main()
