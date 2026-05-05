from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.rag import RAGPipeline
import uvicorn

app = FastAPI(
    title="TaxDoc AI Assistant",
    description="AI-powered tax document Q&A using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF tax document for processing."""
    contents = await file.read()
    doc_id = rag.ingest_document(contents, file.filename)
    return {"doc_id": doc_id, "status": "processed"}

@app.post("/ask")
async def ask_question(doc_id: str, question: str):
    """Ask a natural language question about an uploaded document."""
    answer = rag.answer_question(doc_id, question)
    return {
        "question": question,
        "answer": answer["response"],
        "sources": answer["sources"],
        "confidence": answer["confidence"]
    }

@app.delete("/session/{doc_id}")
async def end_session(doc_id: str):
    """End session and delete document data."""
    rag.cleanup_session(doc_id)
    return {"status": "session ended, data deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
