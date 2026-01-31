import json
import time
from pathlib import Path
import re
import datetime
from typing import Optional

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()


def process_file(input_path: Path, model: SentenceTransformer, out_dir: Path) -> None:
    try:
        full_text = input_path.read_text(encoding='utf-8')
        print(f"Processing {input_path.name}...")
    except FileNotFoundError:
        print(f"Skipping missing file: {input_path}")
        return

    raw_chunks = re.split(
        r"============================================================\nCHUNK \d+\n============================================================\n",
        full_text
    )

    structured_chunks = []
    case_id = input_path.stem
    now = datetime.datetime.now().isoformat()

    for chunk_content in (chunk.strip() for chunk in raw_chunks[1:]):
        if not chunk_content:
            continue

        lines = chunk_content.split('\n')
        section_header = ""
        text_lines = []
        header_found = False

        for line in lines:
            stripped_line = line.strip()
            if not header_found and stripped_line:
                section_header = stripped_line
                header_found = True
            elif header_found:
                text_lines.append(line)

        text_content = '\n'.join(text_lines).strip()
        if section_header and text_content:
            structured_chunks.append({
                "case": case_id,
                "type": "text",
                "section": section_header,
                "text": text_content,
                "timestamp": now
            })

    if not structured_chunks:
        print(f"No valid chunks found in {input_path.name}, skipping.")
        return

    formatted_documents = [
        f"{item['section']}\n{item['text']}"
        for item in structured_chunks
    ]

    print("Generating embeddings...")
    start_time = time.time()
    embeddings_tensor = model.encode(formatted_documents, convert_to_tensor=True)
    elapsed_time = time.time() - start_time

    print(f"Embedding completed in {elapsed_time:.2f}s ({elapsed_time / 60:.2f}m)")
    print(f"Average time per chunk: {elapsed_time / len(structured_chunks):.2f}s")

    embeddings_list = embeddings_tensor.tolist()
    for idx, item in enumerate(structured_chunks):
        item['embedding'] = embeddings_list[idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{case_id}_embedding_google.json"
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(structured_chunks, f, indent=4)

    print(f"Saved embeddings to {output_path.name}.")

def extract_case_number(filename: str) -> Optional[int]:
    """Extract numeric case from filename, e.g., 'case96.pdf' -> 96."""
    m = re.search(r'Case(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    base_dir = Path(__file__).resolve().parent
    chunk_dir = base_dir / "chunk_outputs"
    out_dir = base_dir / "embedded_json"

    txt_files = sorted(chunk_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {chunk_dir}.")
        return

    start_case = 243

    print("Loading nomic-embed-text-v1.5 model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    entries = []
    for txt_file in txt_files:
        cn = extract_case_number(txt_file.name)
        if cn is not None:
            entries.append((cn, txt_file))

    entries.sort(key=lambda t: t[0])
    text_files = [t[1] for t in entries if t[0] >= start_case]
    for idx, file in enumerate(text_files):
        process_file(file, model, out_dir)


if __name__ == "__main__":
    main()
