## What the framework does for me

In Module 02 I wrote the agent loop myself in `agent.py`: send the messages,
check if the model asked for a tool, run it, add the result to the messages,
and repeat. In `gpa_agent.py`, one call to `agent.run_sync()` does all of that.
I don't write the tool description either, because `@agent.tool_plain` builds
it from the type hints and the docstring. With `output_type`, as in
`study_cards.py`, it also checks the model's answer against a Pydantic shape
and gives me back a real Python object instead of text.

## What it hides from me

I no longer see the messages and JSON that go to Ollama, or how many times the
model was called before it answered. To find out which tools ran, I had to
search through `result.all_messages()` after the run finished. I can't watch
it happen step by step. I also don't know what happens when a tool fails, for
example `calculate_gpa` with an empty list, unless I test it myself. The
framework also adds things I didn't ask for, like the startup banner.

## What I'd lose without it

Without it I'd write more code: the loop, the tool descriptions, and my own
checks that the answer has the right shape. Those are the parts where it's easy
to make mistakes. In return I'd keep full control and could see every step,
which is what made Module 02 good for learning. For small experiments I'd write
it by hand. For anything bigger, the framework saves time, but I need to know
what it is doing underneath.
