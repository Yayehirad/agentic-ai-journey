"""Module 04: the same GPA tool from Module 02, but Pydantic AI runs the loop."""
import sys
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# /v1 is Ollama's OpenAI-compatible address, which Pydantic AI talks to
model = OllamaModel("qwen3.5:9b",
                    provider=OllamaProvider(base_url="http://localhost:11434/v1"))

# No output_type this time, so the final answer is plain text.
# We tell the model to use the tool because small models like to
# "do the maths in their head" and sometimes get it wrong.
agent = Agent(model, system_prompt=(
    "You help students with their grades. "
    "Always use the calculate_gpa tool for GPA maths; never compute it yourself."))


# @agent.tool_plain registers the function as a tool. Pydantic AI reads
# the type hints (list[float] -> float) and the docstring to build the
# tool description the model sees. In Module 02 we wrote that by hand.
# "plain" means the tool doesn't need the run context (we don't need it here).
@agent.tool_plain
def calculate_gpa(grades: list[float]) -> float:
    """Calculate a grade point average.

    Args:
      grades: A list of grade points, for example [4.0, 3.7, 3.3]
    """
    return round(sum(grades) / len(grades), 2)


if __name__ == "__main__":
    question = (" ".join(sys.argv[1:])
                or "My grades are 4.0, 3.7 and 3.3. What is my GPA?")

    # run_sync runs the whole loop for us: ask the model, run any tool it
    # asks for, send the result back, repeat until it gives a final answer.
    result = agent.run_sync(question)

    # The framework keeps every message from that loop. We look through the
    # model's replies (ModelResponse) for tool requests (ToolCallPart).
    print("Tools called:")
    called_any = False
    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    print(f"  - {part.tool_name}({part.args})")
                    called_any = True
    if not called_any:
        print("  (none - the model answered without tools)")

    print("\nAnswer:")
    print(result.output)
