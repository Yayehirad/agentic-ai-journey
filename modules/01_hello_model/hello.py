"""Module 01: send one question to a local model and stream the answer."""
import sys
import ollama

MODEL = "qwen3.5:4b"   # the fast model; try "qwen3.5:9b" later


def check_ollama():
    """Stop with a friendly message if Ollama isn't running."""
    try:
        ollama.list()          # asks Ollama which models it has
    except Exception:
        print("Can't reach Ollama. Is the llama icon in your system tray?")
        print("Start Ollama from the Start menu, then run this again.")
        sys.exit(1)


def main():
    check_ollama()

    # The whole conversation is a list of messages. Each has a role.
    messages = [
        {"role": "system", "content": "You are a patient computer science tutor. "
                                      "Answer in 3 sentences or fewer."},
        {"role": "user", "content": "What is a variable in Python?"},
    ]

    print(f"Asking {MODEL}...\n")

    # stream=True gives us the answer in small pieces as it's written.
    # think=False hides the model's private reasoning for now.
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True, think=True):
        if chunk.message.thinking:                       # the model's reasoning
            print(chunk.message.thinking, end="", flush=True)
        if chunk.message.content:                        # the actual answer
            print(chunk.message.content, end="", flush=True)


if __name__ == "__main__":
    main()