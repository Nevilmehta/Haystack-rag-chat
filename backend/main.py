from fastapi import FastAPI, HTTPException
from backend.schemas import ChatRequest, ChatResponse   
from backend.rag_pipeline import RAGService
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="AI Second Brain API",
    description="Explainable RAG backend using FastAPI, Haystack and Ollama",
    version="0.1.0"
)

rag_service = RAGService()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Second Brain backend is running"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = rag_service.ask(request.question)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(error)}"
        )

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            rag_service.ask_stream(request.question),
            media_type="text/event-stream"
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream answer: {str(error)}"
        )