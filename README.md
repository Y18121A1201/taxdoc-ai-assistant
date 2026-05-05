# TaxDoc AI Assistant
> M.S. Capstone Project — University of North Texas

An AI-powered document Q&A system that allows users to upload PDF tax documents and ask natural language questions about their content — built to explore practical applications of large language models and retrieval-augmented generation.

---

## The Research Question

How do you know when an AI system is wrong in a way that matters?

A hallucinated answer looks identical to a correct one. A model retrieving the wrong context responds with the same confidence as one that gets it right. This project was built around that question — systematically experimenting with RAG pipeline design to make the system behavior more observable, auditable, and reliable.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| LLM | OpenAI API (GPT-4o) |
| Embeddings | text-embedding-ada-002 |
| Vector Store | FAISS |
| Orchestration | LangChain |
| Database | PostgreSQL (metadata) |
| File Storage | AWS S3 |
| Frontend | React, TypeScript |

---

---

## Key Experiments and Findings

### 1. Chunking Strategy

Tested three approaches on 50 tax documents:

| Strategy | Retrieval Precision | Notes |
|---|---|---|
| Fixed size (512 tokens) | 0.71 | Baseline |
| Sentence boundary | 0.82 | Best for structured docs |
| Semantic chunking | 0.79 | Higher cost, marginal gain |

**Finding:** Sentence boundary chunking with 50-token overlap gave the best retrieval precision for tax document structure.

### 2. Chunk Size and Overlap

| Chunk Size | Overlap | Precision@3 | Recall@5 |
|---|---|---|---|
| 256 tokens | 25 | 0.74 | 0.81 |
| 512 tokens | 50 | 0.82 | 0.91 |
| 1024 tokens | 100 | 0.76 | 0.94 |

**Finding:** 512 tokens with 50-token overlap was the sweet spot.

### 3. Re-ranking

Adding cross-encoder re-ranking after initial retrieval improved precision by approximately 15% at the cost of ~200ms latency.

### 4. Hallucination Reduction

| Prompt Strategy | Hallucination Rate |
|---|---|
| Baseline (no instructions) | 34% |
| Explicit grounding instructions | 21% |
| Citation requirement | 14% |

**Finding:** Requiring the model to cite the exact document section before answering reduced hallucination rate by ~40%.

---

## Data Privacy Design

- No document content stored beyond the session
- AWS S3 for temporary file storage — auto-deleted after session ends
- All embedding and retrieval done in-memory
- PostgreSQL stores only metadata (filename, session ID, timestamps)

---

## Project Structure

    taxdoc-ai-assistant/
    ├── app/
    │   ├── main.py          # FastAPI application and endpoints
    │   ├── rag.py           # RAG pipeline — retrieval and generation
    │   ├── chunking.py      # Chunking strategies and experiments
    │   └── embeddings.py    # Embedding generation and vector store
    ├── requirements.txt
    └── README.md

---

## Setup and Installation

```bash
git clone https://github.com/YOUR-USERNAME/taxdoc-ai-assistant
cd taxdoc-ai-assistant
pip install -r requirements.txt
export OPENAI_API_KEY=your_openai_key
uvicorn app.main:app --reload
```

---

## What I Learned

Building this made one thing clear — the hard problem is not making the model answer questions. It is knowing when it is wrong.

A system that retrieves the wrong context and generates a confident, fluent, plausible-sounding wrong answer is more dangerous than a system that says it does not know. The instrumentation I built — structured logging, retrieval evaluation, citation requirements — was less about making the system work and more about making its failures visible.

That instinct is what led me toward AI safety research. The same challenge exists at the model level: not just building capable systems, but building systems whose behavior is observable, auditable, and aligned with what we actually want.

---

## Contact

**Niveditha Arkadu**
niveditha.arkadu@gmail.com
