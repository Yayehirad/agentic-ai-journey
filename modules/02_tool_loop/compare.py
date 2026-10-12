"""Module 02: run the same questions on two models and compare them in a table."""
import time

from agent import run_agent

MODELS = ["qwen3.5:9b", "qwen3.5:4b"]

QUESTIONS = [
    "What day is it today?",
    "What time is it right now?",
    "My grades are 4.0, 3.3 and 3.7. What's my GPA?",
    "I got 3.0, 3.0, 4.0 and 3.0 this term. GPA please.",
    "What's in my note todo.txt?",          # this note doesn't exist on purpose
    "What is the capital of France?",       # needs no tool at all
]


def table_safe(text: str) -> str:
    """Make text safe for one Markdown table cell.

    A line break or a "|" would split the cell and break the table.
    """
    return text.replace("\n", " ").replace("|", "/").strip()


def main():
    rows = []   # one entry per run; we print them all at the end

    for model in MODELS:
        for number, question in enumerate(QUESTIONS, start=1):
            # perf_counter() is a precise stopwatch: read it before and after
            start = time.perf_counter()
            answer, tools_used = run_agent(question, model=model)
            seconds = time.perf_counter() - start

            rows.append((number, question, model, tools_used, answer, seconds))
            # Progress line, so a slow run doesn't look frozen
            print(f"{model}, question {number} done ({seconds:.1f}s)")

    # Build the Markdown table: a header row, a divider row, then the data
    print()
    print("| # | Question | Model | Tools called | Answer (first 60 chars) | Seconds |")
    print("|---|---|---|---|---|---|")
    for number, question, model, tools_used, answer, seconds in rows:
        tools = ", ".join(tools_used) or "—"     # "—" means no tool was used
        short = table_safe(answer)[:60]
        print(f"| {number} | {question} | {model} | {tools} | {short} | {seconds:.1f} |")


if __name__ == "__main__":
    main()
