"""Module 02: a hand-built agent loop with three tools."""
import sys
from ollama import chat

# The tools live in tools.py, next to this file
from tools import get_current_time, calculate_gpa, read_note

MODEL = "qwen3.5:9b"
MAX_STEPS = 5   # safety cap: never loop forever


# Map each tool's name to its function, so we can look it up
# when the model asks for a tool by name
TOOLS = {"get_current_time": get_current_time, "calculate_gpa": calculate_gpa,
         "read_note": read_note}


# ---------- The agent loop ----------
def run_agent(question: str, model: str = MODEL) -> tuple[str, list[str]]:
    messages = [{"role": "user", "content": question}]
    tools_used = []   # names of the tools the model called, in order

    for step in range(MAX_STEPS):
        # 1. Ask the model, and tell it which tools exist
        response = chat(model=model, messages=messages,
                        tools=list(TOOLS.values()), think=False)
        messages.append(response.message)          # remember what it said

        # 2. No tool requested? Then this is the final answer
        if not response.message.tool_calls:
            return response.message.content, tools_used

        # 3. The model asked for tools. WE run them, not the model
        for call in response.message.tool_calls:
            name = call.function.name
            tools_used.append(name)        # remember this tool for the caller
            args = call.function.arguments
            fn = TOOLS.get(name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {name}"
            except Exception as e:
                # Don't crash: tell the model what went wrong so it can recover
                result = f"Error: {e}"
            print(f"[tool] {name}({args}) -> {result}")

            # 4. Send the result back as a "tool" message, then loop again
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Stopped: hit the step limit.", tools_used


if __name__ == "__main__":
    # Use the words after the file name as the question, or a default
    question = " ".join(sys.argv[1:]) or \
        "What day is it, and what's my GPA if I got 4.0, 3.7 and 3.3?"
    answer, tools_used = run_agent(question)   # unpack the two values
    print(answer)