# agentic-ai-journey

My step-by-step journey learning how AI agents work, built one small
module at a time in Python. Everything runs **locally on my own computer**
through [Ollama](https://ollama.com): no cloud APIs, no API keys, and my
notes never leave the machine.

## What the finished agent does

By Module 05 it is a **study assistant that answers questions from my own
lecture PDFs** and tells me where it found each answer.

A real run:

```
You: Why do agents that call tools work best at low temperature?
[tool] search_notes({'question': 'Why do agents that call tools work best at low temperature'}) -> [Lecture_01_Foundations_of_Agentic_AI.pdf, page 2] ...
Agent: According to the lecture notes, agents that call tools work best at
low temperature because a tool call must be exactly right. Higher
temperatures generate more varied text but also increase mistakes
(Lecture 01_Foundations_of_Agentic_AI.pdf, page 2).
```

The page is right, but the model copied the file name with a space instead of
an underscore. Small models make small slips like this.

It works in two steps:

1. **Ingest (once):** `ingest.py` reads my PDFs, cuts each page into small
   chunks, turns each chunk into an embedding (a list of numbers that
   captures its meaning) and stores them in a local ChromaDB database.
2. **Ask (every time):** `ask.py` is a chat agent. For each question the
   **model decides** whether it needs to search my notes. If it does, it
   calls the `search_notes` tool, gets the 4 closest chunks with their file
   name and page, and answers only from them, with a citation. If the notes
   don't contain the answer, it says so instead of guessing.

This pattern is called **RAG** (retrieval-augmented generation).

## What makes it an *agent*

A plain chatbot only predicts text. An agent runs a **loop**: the model
chooses an action, my code carries it out, and the result goes back to the
model, until it has a final answer.

```
question ─► model ─► wants a tool? ── no ──► final answer
              ▲            │ yes
              │            ▼
              └── result ◄─ my code runs the tool
```

The model never runs anything itself. It can only *ask*, and my code
decides what actually happens. That is why the tools have safety checks
(only read from one folder, ask me before writing a file) and the loop has
a step limit so it can never run forever.

## The modules

Each module adds one idea on top of the last.

| Module | What I built | What I learned |
|---|---|---|
| [01 Hello model](modules/01_hello_model/) | Send one question to a local model and stream the answer | Messages and roles (system, user, assistant); the model remembers nothing between calls |
| [02 Tool loop](modules/02_tool_loop/) | A hand-written agent loop with 3 tools: time, GPA, read a note. Compared a 9b and a 4b model | How tool calling works; the loop is about 30 lines; bigger models pick the right tool more often |
| [03 Chat agent](modules/03_chat_agent/) | A terminal chat with memory, a `save_note` tool that asks permission, and a trace log | Memory is just the message list; trimming it safely; human approval for actions that change things |
| [04 Structured output](modules/04_structured_output/) | Study cards in a fixed shape with Pydantic AI, and the GPA tool rebuilt with the framework | What a framework does for me, and what it hides ([COMPARISON.md](modules/04_structured_output/COMPARISON.md)) |
| [05 RAG](modules/05_rag/) | Ingest PDFs, search them, and a chat agent that searches and cites file and page | Embeddings, chunking, vector search, and stopping the model from inventing citations |

## Setup

Windows 11 with Git Bash. You need [Ollama](https://ollama.com) installed
and running.

```bash
# 1. The models (chat: big and fast, plus one for embeddings)
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# 2. A virtual environment with the libraries
python -m venv .venv
source .venv/Scripts/activate
pip install ollama "pydantic-ai-slim[openai]" pypdf chromadb
```

Activate the virtual environment (`source .venv/Scripts/activate`) in every
new terminal before running anything. You should see `(.venv)` in the prompt.

## Try the PDF study assistant

```bash
# Put PDFs in a folder, then store them (this wipes the old database first)
python modules/05_rag/ingest.py path/to/your/pdfs

# Optional: see the raw search results for a question
python modules/05_rag/search.py "what is prompt injection"

# Chat with the agent (type 'quit' to stop)
python modules/05_rag/ask.py
```

`ingest.py` takes `--chunk-size` (default 800 characters) if you want to
experiment with how the notes are cut up.

## Limits

- It runs on small local models, so it is slower and makes more mistakes
  than a big cloud model. Always check that a citation points to a real
  page.
- It can only read PDFs that contain real text. Scanned pages (pictures of
  text) are skipped.
- Personal notes, PDFs, the database and logs are kept out of git by
  `.gitignore`.
