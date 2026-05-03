from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)

class SourceChunk(BaseModel):
    source: str
    chunk_id: int
    score: float | None
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: float
    response_time_seconds: float
