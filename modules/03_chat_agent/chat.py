"""Module 03: a terminal chat agent with memory, a guarded save tool and a trace log."""
from ollama import chat

# The tools live in tools.py, the logger in trace_log.py, both next to this file
from tools import get_current_time, calculate_gpa, read_note, save_note
from trace_log import log_event

MODEL = "qwen3.5:9b"
MAX_STEPS = 5      # safety cap per turn: never loop forever
MAX_HISTORY = 20   # how many non-system messages the model gets to see

SYSTEM_PROMPT = (
    "You are a helpful study assistant chatting in a terminal. "
    "You can tell the time, calculate a GPA, read the user's notes and save notes. "
    "Saving a note asks the user for permission; if they decline, accept it. "
    "Keep answers short."
)

# Map each tool's name to its function, so we can look it up
# when the model asks for a tool by name
TOOLS = {"get_current_time": get_current_time, "calculate_gpa": calculate_gpa,
         "read_note": read_note, "save_note": save_note}


# ---------- Memory ----------
def trim_history(messages: list) -> list:
    """Keep the system prompt plus only the most recent messages."""
    system, rest = messages[0], messages[1:]
    recent = rest[-MAX_HISTORY:]

    # A plain cut can start in the middle of a tool exchange, e.g. with a
    # "tool" result whose request was cut off. That confuses the model, so
    # drop messages from the front until we start at one of the user's.
    while recent and recent[0]["role"] != "user":
        recent.pop(0)

    # Rare case: this one turn alone is longer than MAX_HISTORY, so no user
    # message was left. Then keep everything from the latest user message.
    if not recent:
        last_user = max(i for i, m in enumerate(rest) if m["role"] == "user")
        recent = rest[last_user:]

    return [system] + recent


# ---------- The agent loop (one turn of the chat) ----------
def run_agent(messages: list, model: str = MODEL) -> str:
    # messages is the SHARED conversation. We change it in place, so the
    # chat loop below still has every reply and tool result next turn.
    for step in range(MAX_STEPS):
        # 0. Forget old messages. messages[:] = ... replaces the contents
        #    of the same list, instead of making a new list nobody else sees.
        messages[:] = trim_history(messages)

        # 1. Ask the model, and tell it which tools exist
        response = chat(model=model, messages=messages,
                        tools=list(TOOLS.values()), think=False)
        messages.append(response.message)          # remember what it said
        # model_dump turns ollama's reply object into a plain dict for JSON
        log_event("model_reply", response.message.model_dump(exclude_none=True))

        # 2. No tool requested? Then this is the final answer
        if not response.message.tool_calls:
            return response.message.content

        # 3. The model asked for tools. WE run them, not the model
        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            fn = TOOLS.get(name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {name}"
            except Exception as e:
                # Don't crash: tell the model what went wrong so it can recover
                result = f"Error: {e}"
            print(f"[tool] {name}({args}) -> {result}")
            log_event("tool_call", {"name": name, "args": args, "result": str(result)})

            # 4. Send the result back as a "tool" message, then loop again
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Stopped: hit the step limit."


# ---------- The chat loop ----------
def main() -> None:
    # The system prompt is message 0 and is never trimmed away
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Chat started. Type 'quit' to stop.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C (or a closed input) ends the chat politely, no traceback
            break

        if user_text.lower() == "quit":
            break
        if not user_text:
            continue   # empty line: just ask again

        messages.append({"role": "user", "content": user_text})
        answer = run_agent(messages)
        print(f"Agent: {answer}")

    print("\nBye!")


if __name__ == "__main__":
    main()
