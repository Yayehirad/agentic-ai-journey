"""Module 02: the tools our agent can use. Each tool is a plain Python function."""
from datetime import datetime
from pathlib import Path

# The docstring (text in triple quotes) is what the model reads
# to decide when to use each tool. Keep them clear.

# The only folder read_note may read from: "notes" next to this file.
# Path(__file__).parent means "the folder this file lives in", so this
# works no matter which folder you run the command from.
NOTES_DIR = (Path(__file__).parent / "notes").resolve()
ALLOWED_TYPES = {".txt", ".md"}


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


def calculate_gpa(grades: list[float]) -> float:
    """Calculate a grade point average.

    Args:
      grades: A list of grade points, for example [4.0, 3.7, 3.3]
    """
    return round(sum(grades) / len(grades), 2)


def read_note(filename: str) -> str:
    """Read one of the user's notes and return its text.

    Args:
      filename: Just the note's file name, for example "ideas.txt"
    """
    # .resolve() turns the path into the real, full location on disk.
    # This undoes tricks like "../../secret.txt" or a full C:\ path,
    # so the check below sees where the file REALLY is.
    path = (NOTES_DIR / filename).resolve()

    # Safety check 1: the real location must be inside the notes folder.
    # Checking the text for ".." isn't enough; resolving is the reliable way.
    if not path.is_relative_to(NOTES_DIR):
        return "Refused: I can only read files inside the notes folder."

    # Safety check 2: only plain text and Markdown files.
    if path.suffix.lower() not in ALLOWED_TYPES:
        return "Refused: I can only read .txt or .md files."

    if not path.is_file():
        return f"No note called {filename!r} found."

    return path.read_text(encoding="utf-8")
