from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    history: list[ChatMessage] = []

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
    cached: bool = False
