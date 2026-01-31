import os

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "docling"

doc_converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
# The :memory: mode is a Python imitation of Qdrant's APIs for prototyping and CI.
# For production deployments, use the Docker image: docker run -p 6333:6333 qdrant/qdrant
# client = QdrantClient(location="http://localhost:6333")

client.set_model("sentence-transformers/all-MiniLM-L6-v2")
client.set_sparse_model("Qdrant/bm25")

result = doc_converter.convert(
    "https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag"
)
documents, metadatas = [], []
for chunk in HybridChunker().chunk(result.document):
    documents.append(chunk.text)
    metadatas.append(chunk.meta.export_json_dict())

_ = client.add(
    collection_name=COLLECTION_NAME,
    documents=documents,
    metadata=metadatas,
    batch_size=64,
)

points = client.query(
    collection_name=COLLECTION_NAME,
    query_text="Can I split documents?",
    limit=10,
)

for i, point in enumerate(points):
    print(f"=== {i} ===")
    print(point.document)
    print()