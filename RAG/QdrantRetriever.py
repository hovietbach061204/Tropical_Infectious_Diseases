import asyncio
import os
from typing import Any, Dict, Optional, List
from langsmith import traceable
from qdrant_client import QdrantClient
from RAG.utils.embeddings import embed_text_query
from RAG.utils.queryVectorDB import search_vectors_v2

_client = None
_retriever = None

class QdrantRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "nomic_text_vectors",
        embedding_fn=embed_text_query,
        using_vector="nomic-embed-text",
        static_filters: Optional[Dict[str, Any]] = None,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn
        self._using_vector = using_vector
        self._static_filters = static_filters or {}

    @traceable
    async def query(self, query, k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[dict]:
        """
        query: str | PIL.Image | image path
        returns: [{id, score, context, payload}, ...]
        """
        merged_filters = {**self._static_filters, **(filters or {})}

        # Run sync Qdrant call in thread
        results = await asyncio.to_thread(
            search_vectors_v2,
            query,
            self._client,
            self._collection_name,
            self._embedding_fn,
            k,
            merged_filters or None,
            self._using_vector,
            False,  # verbose off
        )

        docs = []
        for r in results or []:
            payload = getattr(r, "payload", {}) or {}

            # CALL THE STANDARD FUNCTION
            std_context = standardize_context(payload)

            docs.append({
                "id": getattr(r, "id", None),
                "score": float(getattr(r, "score", 0.0)),
                "context": std_context, # Now strictly formatted
                "payload": payload,
            })
        return docs


def standardize_context(payload: dict) -> str:
    """
    Normalizes diverse payload structures into a single consistent string format
    for the LLM context window.
    """
    # 1. Extract Common Metadata
    section = payload.get("section", "General")

    # 2. Extract Core Content (The "Text")
    # Some chunks have 'text', others might have it elsewhere.
    raw_text = payload.get("text", "")

    # 3. Extract Structured Fields (if present)
    # We build a list of lines to keep it clean.
    structured_lines = []

    # Disease Name
    dx_short = payload.get("Disease Name Short")
    if dx_short:
        structured_lines.append(f"Disease: {dx_short}")

    # Final Diagnosis (often long, so we label it clearly)
    final_dx = payload.get("Final Diagnosis")
    if final_dx:
        structured_lines.append(f"Diagnosis Details: {final_dx}")

    # Vitals
    vitals = payload.get("Vitals")
    if vitals:
        structured_lines.append(f"Vitals: {vitals}")

    # 4. Construct the Final Block
    # Header
    header = f"[{section.upper()}]"

    # Body
    # If we have structured fields, put them first as they are high-density info.
    body_parts = []
    if structured_lines:
        body_parts.extend(structured_lines)

    # Add the raw text if it adds new info (check for near-duplicates if needed)
    if raw_text:
        # Optional: Truncate if text is just a repeat of Final Diagnosis
        if final_dx and final_dx in raw_text:
            pass # Skip redundant text
        else:
            body_parts.append(f"Content: {raw_text}")

    return f"{header}\n" + "\n".join(body_parts)


def get_retriever():
    """Lazy initialization - creates client only when needed."""
    global _client, _retriever

    if _retriever is None:
        _client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
        )
        _retriever = QdrantRetriever(client=_client)

    return _retriever