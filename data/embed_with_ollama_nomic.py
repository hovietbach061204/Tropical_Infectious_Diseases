# python
import datetime
import json
import re
import time
from pathlib import Path

import ollama


def parse_case95_chunks(text: str):
    """
    Parse chunks from Case95.txt which uses lines of '=' around 'CHUNK <n>' headers.
    Returns a list of dicts with chunk_index, section, and text.
    """
    # Split on: line of '=' + 'CHUNK <num>' + line of '='
    parts = re.split(r"^=+\s*\nCHUNK\s+(\d+)\s*\n=+\s*\n", text, flags=re.MULTILINE)
    # parts -> [prelude, id1, content1, id2, content2, ...]
    chunks = []
    now = datetime.datetime.now().isoformat()

    # Skip prelude at index 0; iterate over pairs (id, content)
    for i in range(1, len(parts), 2):
        idx_str = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        section = f"CHUNK {idx}"
        # Clean up content
        content = content.strip()
        if not content:
            continue

        chunks.append(
            {
                "case": "Case95",
                "type": "text_chunk",
                "section": section,
                "chunk_index": idx,
                "text": content,
                "timestamp": now,
            }
        )
    return chunks


def embed_chunks(chunks):
    """
    For each chunk, build 'search_document: <section>\\n<text>' and call ollama.embeddings.
    """
    for item in chunks:
        doc = f"{item['text']}"
        resp = ollama.embeddings(model="nomic-embed-text:latest", prompt=doc)
        item["embedding"] = resp["embedding"]
    return chunks


def main():
    # Resolve paths relative to this file
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "chunk_outputs" / "Case95.txt"
    out_dir = base_dir / "embedded_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "case95_embeddings_ollama_v15_v2.json"

    # Read Case95.txt as plain text
    try:
        text = input_path.read_text(encoding="utf-8")
        print(f"Read '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found.")
        return

    # Parse chunks
    chunks = parse_case95_chunks(text)
    if not chunks:
        print("No chunks parsed from input.")
        return
    print(f"Parsed {len(chunks)} chunks.")

    # Generate embeddings per chunk
    print("Generating embeddings with 'nomic-embed-text' (dim=768)...")
    start_time = time.time()
    chunks = embed_chunks(chunks)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Embeddings generated.")

    print(f"Embedding completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Average time per chunk: {elapsed_time / len(chunks):.2f} seconds")

    # Save JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Wrote embeddings to '{output_path}'.")


if __name__ == "__main__":
    main()
