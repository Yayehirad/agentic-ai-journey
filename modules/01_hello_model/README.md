# Module 01: Hello, model

`hello.py` sends one question to a local model through Ollama and prints
the answer as it streams in, piece by piece. First it checks that Ollama
is running. Then it sends a short conversation: a system instruction and
a user question.

## Run it

```bash
source .venv/Scripts/activate
python modules/01_hello_model/hello.py
```

## Roles

Each message in the conversation has a role:

- **system**: standing instructions from the developer, such as who the
  model is and how to answer.
- **user**: the person's question or request.
- **assistant**: the model's reply.

The model remembers nothing between calls. To continue a conversation,
add its reply as an `assistant` message, add your next `user` message,
and send the whole list again.
