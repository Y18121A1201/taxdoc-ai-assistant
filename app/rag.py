import uuid
import logging
from typing import Dict, Any
from openai import OpenAI
from app.chunking import ChunkingStrategy
from app.embeddings import EmbeddingStore
import PyPDF2
import io

# Structured logging for auditability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

client = OpenAI()

SYSTEM_PROMPT = """You are a precise tax document assistant.
Answer questions ONLY using the provided document context.
Before answering, cite the exact section or paragraph you are drawing from.
If the context does not contain enough information to answer confidently, say so.
Never fabricate information not present in the context."""

class RAGPipeline:
    def __init__(self):
        self.chunker = ChunkingStrategy(
            strategy="sentence_boundary",
            chunk_size=512,
            overlap=50
        )
        self.store = EmbeddingStore()
        self.sessions: Dict[str, Any] = {}
        logger.info("RAG Pipeline initialized")

    def ingest_document(self, file_bytes: bytes, filename: str) -> str:
        """Ingest a PDF document and return a session ID."""
        doc_id = str(uuid.uuid4())
        logger.info(f"Ingesting document: {filename} | session: {doc_id}")

        # Extract text from PDF
        text = self._extract_text(file_bytes)
        logger.info(f"Extracted {len(text)} characters from {filename}")

        # Chunk the document
        chunks = self.chunker.chunk(text)
        logger.info(f"Created {len(chunks)} chunks using sentence_boundary strategy")

        # Generate embeddings and store
        self.store.add_documents(doc_id, chunks)
        self.sessions[doc_id] = {"filename": filename, "chunk_count": len(chunks)}

        return doc_id

    def answer_question(self, doc_id: str, question: str) -> Dict[str, Any]:
        """Retrieve context and generate a grounded answer."""
        logger.info(f"Question received | session: {doc_id} | question: {question}")

        # Retrieve relevant chunks
        retrieved = self.store.search(doc_id, question, k=5)
        logger.info(f"Retrieved {len(retrieved)} chunks for question")

        # Re-rank retrieved chunks
        reranked = self._rerank(question, retrieved)

        # Build context from top chunks
        context = "\n\n---\n\n".join([chunk["text"] for chunk in reranked[:3]])

        # Generate answer with citation requirement
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1  # Low temperature for factual accuracy
        )

        answer = response.choices[0].message.content
        logger.info(f"Answer generated | session: {doc_id} | tokens: {response.usage.total_tokens}")

        return {
            "response": answer,
            "sources": [chunk["text"][:200] for chunk in reranked[:3]],
            "confidence": self._estimate_confidence(answer, context)
        }

    def _extract_text(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() for page in reader.pages])

    def _rerank(self, question: str, chunks: list) -> list:
        """Simple re-ranking based on keyword overlap."""
        question_words = set(question.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            chunk["rerank_score"] = len(question_words & chunk_words) / len(question_words)
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate answer confidence based on citation presence."""
        citation_indicators = ["according to", "the document states", "as stated in", "section"]
        has_citation = any(indicator in answer.lower() for indicator in citation_indicators)
        return 0.9 if has_citation else 0.6

    def cleanup_session(self, doc_id: str):
        """Delete all session data — privacy by design."""
        self.store.delete(doc_id)
        if doc_id in self.sessions:
            del self.sessions[doc_id]
        logger.info(f"Session cleaned up: {doc_id}")
