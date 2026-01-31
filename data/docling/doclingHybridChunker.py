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


import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    max_chunk_size: int = 2000  # Maximum chunk size
    min_chunk_size: int = 100  # Minimum chunk size
    use_semantic_splitting: bool = True  # Use HybridChunker (recommended)
    preserve_structure: bool = True  # Preserve document structure
    max_tokens: int = 512  # Maximum tokens for embedding models

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """Represents a document chunk with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None  # For embedder compatibility

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4

class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.

    This chunker uses Docling's built-in HybridChunker which:
    - Respects document structure (sections, paragraphs, tables)
    - Is token-aware (fits embedding model limits)
    - Preserves semantic coherence
    - Includes heading context in chunks
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # Initialize tokenizer for token-aware chunking
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )

        logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")

    def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format)
            title: Document title
            source: Document source
            metadata: Additional metadata
            docling_doc: Optional pre-converted DoclingDocument (for efficiency)

        Returns:
            List of document chunks with contextualized content
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {})
        }

        # If we don't have a DoclingDocument, we need to create one from markdown
        if docling_doc is None:
            # For markdown content, we need to convert it to DoclingDocument
            # This is a simplified version - in practice, content comes from
            # Docling's document converter in the ingestion pipeline
            logger.warning("No DoclingDocument provided, using simple chunking fallback")
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Use HybridChunker to chunk the DoclingDocument
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            # Convert Docling chunks to DocumentChunk objects
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self.chunker.contextualize(chunk=chunk)

                # Count actual tokens
                token_count = len(self.tokenizer.encode(contextualized_text))

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True  # Flag indicating contextualized chunk
                }

                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(DocumentChunk(
                    content=contextualized_text.strip(),
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata,
                    token_count=token_count
                ))

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
            self,
            content: str,
            base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple fallback chunking when HybridChunker can't be used.

        This is used when:
        - No DoclingDocument is provided
        - HybridChunker fails

        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple sliding window approach
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                # Try to end at sentence boundary
                chunk_end = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < len(content) and content[i] in '.!?\n':
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))

                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1  # Will update after
                    },
                    token_count=token_count
                ))

                chunk_index += 1

            # Move forward with overlap
            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks

def analyze_chunks(chunks: List[DocumentChunk], tokenizer):
    if not chunks:
        print("No chunks to analyze.")
        return
    token_counts = [c.token_count for c in chunks]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(chunks)
    print("\n" + "=" * 60)
    print("CHUNK ANALYSIS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens (approx): {total_tokens}")
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    print(f"Max tokens in a chunk: {max(token_counts)}")
    print(f"Min tokens in a chunk: {min(token_counts)}")
    print("Sample first chunk:")
    print("-" * 40)
    print(chunks[0].content[:400], "...")


def save_chunks(chunks: List[DocumentChunk], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"{'='*60}\nCHUNK {c.index}\n{'='*60}\n")
            f.write(c.content)
            f.write("\n\n")
    print(f"\n✓ Chunks saved to: {output_path}")

def main():
    print("=" * 60)
    print("Hybrid Chunking with Docling")
    print("=" * 60)

    # Document to process
    pdf_path = "../Tropical_Dataset/case147.pdf"
    max_tokens = 512  # Typical limit for embedding models

    print(f"\nInput: {pdf_path}")
    print(f"Max tokens per chunk: {max_tokens}")

    try:
        # Build Docling document
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True, enable_remote_services=True
        )
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        # pipeline_options.do_picture_description = True
        # pipeline_options.picture_description_options = smolvlm_picture_description
        # pipeline_options.generate_picture_images = True
        # pipeline_options.images_scale = 2.0

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        print("Converting PDF...")
        result = converter.convert(pdf_path)
        docling_doc = result.document
        markdown = docling_doc.export_to_markdown()


        # Chunk
        chunker = DoclingHybridChunker(ChunkingConfig(max_tokens=max_tokens))
        chunks = chunker.chunk_document(
            content=markdown,
            title="case147",
            source=pdf_path,
            docling_doc=docling_doc,
        )

        # Analyze
        analyze_chunks(chunks, chunker.tokenizer)

        # Save
        output_path = "output/Case147.txt"
        save_chunks(chunks, output_path)

        print("\n" + "=" * 60)
        print("KEY BENEFITS OF HYBRID CHUNKING")
        print("=" * 60)
        print("✓ Respects structure")
        print("✓ Token-aware")
        print("✓ Semantic coherence")
        print("✓ Metadata preserved")
        print("✓ RAG-ready")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()