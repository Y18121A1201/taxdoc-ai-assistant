import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
import faiss
import logging

logger = logging.getLogger(__name__)
client = OpenAI()

class EmbeddingStore:
    """
    Manages document embeddings and vector search using FAISS.
    Each session gets its own isolated index for privacy.
    """

    def __init__(self):
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self.chunks: Dict[str, List[str]] = {}

    def add_documents(self, doc_id: str, chunks: List[str]):
        """Generate embeddings and add to FAISS index."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embeddings = self._embed_texts(chunks)
        dimension = len(embeddings[0])

        # Inner product index (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        vectors = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        index.add(vectors)

        self.indexes[doc_id] = index
        self.chunks[doc_id] = chunks
        logger.info(f"Added {len(chunks)} embeddings to index for session {doc_id}")

    def search(self, doc_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using semantic similarity."""
        query_embedding = self._embed_texts([query])[0]
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.indexes[doc_id].search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[doc_id][idx],
                    "score": float(score),
                    "index": int(idx)
                })

        logger.info(f"Retrieved {len(results)} chunks | top score: {scores[0][0]:.3f}")
        return results

    def delete(self, doc_id: str):
        """Delete session data — privacy by design."""
        if doc_id in self.indexes:
            del self.indexes[doc_id]
        if doc_id in self.chunks:
            del self.chunks[doc_id]
        logger.info(f"Deleted embedding index for session {doc_id}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
