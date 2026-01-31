"""
Hybrid Chunking with Docling
=============================

This script demonstrates Docling's HybridChunker for intelligent
document chunking that respects both document structure and
token limits.

What is Hybrid Chunking?
- Combines hierarchical document structure with token-aware splitting
- Respects semantic boundaries (paragraphs, sections, tables)
- Ensures chunks fit within token limits for embeddings
- Preserves metadata and document hierarchy

Why use it?
- Better for RAG systems than naive text splitting
- Maintains semantic coherence within chunks
- Optimized for embedding models with token limits
- Preserves document structure and context
"""
import re
from pathlib import Path
from typing import Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from transformers import AutoTokenizer


def chunk_document(file_path: str, max_tokens: int = 512):
    """Convert and chunk document using HybridChunker."""

    print(f"\n📄 Processing: {Path(file_path).name}")

    # Step 1: Convert document to DoclingDocument
    print("   Step 1: Converting document...")
    pipeline_options = PdfPipelineOptions(do_table_structure=True, enable_remote_services=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(file_path)
    doc = result.document

    # Step 2: Initialize tokenizer (using sentence-transformers model)
    print("   Step 2: Initializing tokenizer...")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Step 3: Create HybridChunker
    print(f"   Step 3: Creating chunker (max {max_tokens} tokens)...")
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=True  # Merge small adjacent chunks
    )

    # Step 4: Generate chunks
    print("   Step 4: Generating chunks...")
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    return chunks, tokenizer, chunker

def analyze_chunks(chunks, tokenizer):
    """Analyze and display chunk statistics."""

    print("\n" + "=" * 60)
    print("CHUNK ANALYSIS")
    print("=" * 60)

    total_tokens = 0
    chunk_sizes = []

    for i, chunk in enumerate(chunks):
        # Get text content
        text = chunk.text
        tokens = tokenizer.encode(text)
        token_count = len(tokens)

        total_tokens += token_count
        chunk_sizes.append(token_count)

        # Display first 3 chunks in detail
        # if i < 3:
        #     print(f"\n--- Chunk {i} ---")
        #     print(f"Tokens: {token_count}")
        #     print(f"Characters: {len(text)}")
        #     print(f"Preview: {text[:150]}...")
        #
        #     # Show metadata if available
        #     if hasattr(chunk, 'meta') and chunk.meta:
        #         print(f"Metadata: {chunk.meta}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per chunk: {total_tokens / len(chunks):.1f}")
    print(f"Min tokens: {min(chunk_sizes)}")
    print(f"Max tokens: {max(chunk_sizes)}")

    # Token distribution
    print(f"\nToken distribution:")
    ranges = [(0, 128), (128, 256), (256, 384), (384, 512)]
    for start, end in ranges:
        count = sum(1 for size in chunk_sizes if start <= size < end)
        print(f"  {start}-{end} tokens: {count} chunks")

def save_chunks(chunks, chunker, output_path: str):
    """Save chunks to file with separators, preserving context and headings."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"{'='*60}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'='*60}\n")

            # Use contextualize to preserve headings and metadata
            contextualized_text = chunker.contextualize(chunk=chunk)
            f.write(contextualized_text)
            f.write("\n\n")

    print(f"\n✓ Chunks saved to: {output_path}")
    print("   (with preserved headings and document context)")

def extract_case_number(filename: str) -> Optional[int]:
    """Extract numeric case from filename, e.g., 'case96.pdf' -> 96."""
    m = re.search(r'case(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None

def process_dataset(
    dataset_dir: str,
    output_dir: str = "output",
    max_tokens: int = 512,
    start_case: Optional[int] = None,
):
    """Process all PDF files in the dataset directory."""

    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    if not dataset_path.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return

    # Find all PDF files
    pdf_files = list(dataset_path.glob("*.pdf"))
    # Build list of (case_num, Path) and drop files without a case number
    entries = []
    for p in pdf_files:
        cn = extract_case_number(p.name)
        if cn is not None:
            entries.append((cn, p))

    # Sort numerically and apply start filter
    entries.sort(key=lambda t: t[0])
    if start_case is not None:
        entries = [t for t in entries if t[0] >= start_case]

    if not entries:
        print(f"✗ No matching PDF files found in {dataset_dir} for start_case={start_case}")
        return

    print("=" * 60)
    print("Hybrid Chunking with Docling - Batch Processing")
    print("=" * 60)
    print(f"\nDataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Max tokens per chunk: {max_tokens}")
    if start_case is not None:
        print(f"Starting from case: {start_case}")
    print(f"Total files to process: {len(entries)}\n")

    successful = 0
    failed = 0

    for idx, (case_num, pdf_file) in enumerate(entries, 1):
        try:
            print(f"\n[{idx}/{len(entries)}] Processing {pdf_file.name} (case {case_num})...")

            chunks, tokenizer, chunker = chunk_document(str(pdf_file), max_tokens)
            analyze_chunks(chunks, tokenizer)

            output_filename = f"Case{case_num}.txt"
            output_file_path = output_path / output_filename

            save_chunks(chunks, chunker, str(output_file_path))
            successful += 1

        except Exception as e:
            print(f"   ✗ Error processing {pdf_file.name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(entries)}")
    print("\n" + "=" * 60)
    print("KEY BENEFITS OF HYBRID CHUNKING")
    print("=" * 60)
    print("✓ Respects document structure (sections, paragraphs)")
    print("✓ Token-aware (fits embedding model limits)")
    print("✓ Semantic coherence (doesn't split mid-sentence)")
    print("✓ Metadata preservation (headings, document context)")
    print("✓ Ready for RAG (optimized chunk sizes)")


def main():
    print("=" * 60)
    print("Hybrid Chunking with Docling")
    print("=" * 60)
    #
    # # Document to process
    # pdf_path = "./Tropical_Dataset/case96.pdf"
    # max_tokens = 512  # Typical limit for embedding models
    #
    # print(f"\nInput: {pdf_path}")
    # print(f"Max tokens per chunk: {max_tokens}")
    #
    # try:
    #     # Generate chunks
    #     chunks, tokenizer, chunker = chunk_document(pdf_path, max_tokens)
    #
    #     # Analyze chunks
    #     analyze_chunks(chunks, tokenizer)
    #
    #     # Save chunks
    #     output_path = "output/output_chunks_v4.txt"
    #     save_chunks(chunks, chunker, output_path)
    #
    #     print("\n" + "=" * 60)
    #     print("KEY BENEFITS OF HYBRID CHUNKING")
    #     print("=" * 60)
    #     print("✓ Respects document structure (sections, paragraphs)")
    #     print("✓ Token-aware (fits embedding model limits)")
    #     print("✓ Semantic coherence (doesn't split mid-sentence)")
    #     print("✓ Metadata preservation (headings, document context)")
    #     print("✓ Ready for RAG (optimized chunk sizes)")
    #
    # except Exception as e:
    #     print(f"\n✗ Error: {e}")
    #     import traceback
    #     traceback.print_exc()

    dataset_dir = "../Tropical_Dataset"
    output_dir = "../chunk_outputs"
    max_tokens = 512
    start_case = 243  # start from this case (inclusive)

    process_dataset(dataset_dir, output_dir, max_tokens, start_case=start_case)


if __name__ == "__main__":
    main()