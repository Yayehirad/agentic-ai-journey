"""Module 05: read PDFs, cut them into chunks, embed them and store them in ChromaDB."""
import argparse
from pathlib import Path

import chromadb
from ollama import embed
from pypdf import PdfReader

# db/ next to this file, wherever you run the command from
DB_DIR = Path(__file__).parent / "db"
COLLECTION = "notes"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 800  # about 1-2 paragraphs: enough context, still one topic
OVERLAP = 100             # so a sentence cut at a boundary is whole in one chunk


def chunk_text(text: str, size: int) -> list[str]:
    """Cut text into pieces of `size` characters that overlap by OVERLAP."""
    step = size - OVERLAP     # e.g. size 800: each chunk starts 700 after the last
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break             # this chunk reached the end; stop
    return chunks


def ingest_pdf(pdf: Path, collection, size: int) -> int:
    """Store one PDF's chunks. Returns how many chunks were stored."""
    stored = 0
    # We chunk PAGE BY PAGE so every chunk has exactly one page number
    for page_num, page in enumerate(PdfReader(pdf).pages, start=1):
        # extract_text() can return None for pages with no text (e.g. scans).
        # " ".join(text.split()) squashes runs of spaces and newlines.
        text = " ".join((page.extract_text() or "").split())
        if not text:
            continue

        chunks = chunk_text(text, size)
        # nomic-embed-text was trained with this label in front of stored text
        vectors = embed(model=EMBED_MODEL,
                        input=[f"search_document: {c}" for c in chunks]).embeddings

        collection.add(
            # IDs like "notes.pdf-p3-c0", unique inside this run
            ids=[f"{pdf.name}-p{page_num}-c{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,       # the original text, to show later
            metadatas=[{"file": pdf.name, "page": page_num}] * len(chunks),
        )
        stored += len(chunks)
    return stored


def fresh_collection():
    """Delete the old collection (if any) and return a new, empty one."""
    # PersistentClient saves to disk, so the data is still there next run.
    client = chromadb.PersistentClient(path=str(DB_DIR))
    # Wipe first, so chunks of different sizes never get mixed together.
    # We ask Chroma to delete it instead of deleting the db/ folder, because
    # Windows refuses to delete files that are still open.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass    # first run: there was nothing to delete
    # embedding_function=None: we embed with Ollama ourselves, so Chroma
    # must not download and use its own embedding model.
    return client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def main() -> None:
    # argparse (built into Python) reads the command line for us and
    # makes a --help message automatically
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("folder", type=Path, help="folder with PDF files")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"characters per chunk (default {DEFAULT_CHUNK_SIZE})")
    args = parser.parse_args()   # --chunk-size becomes args.chunk_size

    if not args.folder.is_dir():
        parser.error(f"not a folder: {args.folder}")
    # With size <= OVERLAP the next chunk would never move forward
    if args.chunk_size <= OVERLAP:
        parser.error(f"--chunk-size must be bigger than the overlap ({OVERLAP})")

    collection = fresh_collection()
    print(f"Wiped old db. Chunk size {args.chunk_size}, overlap {OVERLAP}.")

    # .suffix.lower() also catches files named .PDF
    pdfs = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".pdf")
    pdf_count = chunk_count = 0
    for pdf in pdfs:
        try:
            n = ingest_pdf(pdf, collection, args.chunk_size)
        except Exception as e:
            # A broken or password-protected PDF shouldn't stop the others
            print(f"  Skipped {pdf.name}: {e}")
            continue
        if n == 0:
            print(f"  {pdf.name}: no text found (scanned images?), skipped")
            continue
        print(f"  {pdf.name}: {n} chunks")
        pdf_count += 1
        chunk_count += n

    print(f"Stored {pdf_count} PDFs, {chunk_count} chunks in {DB_DIR}")


if __name__ == "__main__":
    main()
