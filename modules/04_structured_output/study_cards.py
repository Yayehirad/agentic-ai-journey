"""Module 04: Make the model return a fixed shape, a StudyCard."""
import sys
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# The shape we demand. Pydantic checks the model's answer against it.
class StudyCard(BaseModel):
    topic: str
    key_points: list[str] = Field(min_length=3, max_length=5)
    common_mistake: str
    confidence: float = Field(ge=0, le=1)   # 0 = guessing, 1 = certain


# /v1 is Ollama's OpenAI-compatible address, which Pydantic AI talks to
model = OllamaModel("qwen3.5:9b",
                    provider=OllamaProvider(base_url="http://localhost:11434/v1"))

agent = Agent(model, output_type=StudyCard,
              system_prompt="You create concise revision cards for students.")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Python list comprehensions"
    result = agent.run_sync(f"Make a study card about {topic}")
    card = result.output                      # a real StudyCard object
    print(card.model_dump_json(indent=2))