from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from backend.config import DATA_DIR
from backend.schemas import ChatRequest, ChatResponse   
from backend.rag_pipeline import RAGService
from fastapi.responses import StreamingResponse
from backend.security import verify_api_key, validate_question

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
def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    try:
        validate_question(request.question)
        result = rag_service.ask(request.question)
        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(error)}"
        )

@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    try:
        validate_question(request.question)

        history = [message.model_dump() for message in request.history]

        return StreamingResponse(
            rag_service.ask_stream(
                request.question,
                request.session_id,
                history,
            ),
            media_type="text/event-stream",
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream answer: {str(error)}",
        )

@app.post("/upload")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    allowed_extensions = [".pdf", ".txt"]
    file_extension = "." + file.filename.split(".")[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    session_dir = DATA_DIR / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / file.filename
    content = await file.read()

    with open(file_path, "wb") as output_file:
        output_file.write(content)

    rag_service.reload_session(session_id)

    return {
        "message": "File uploaded and indexed successfully.",
        "filename": file.filename,
        "session_id": session_id,
    } 