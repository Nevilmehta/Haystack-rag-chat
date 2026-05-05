from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from backend.config import API_KEY, BLOCKED_PROMPT_PATTERNS, MAX_QUESTION_LENGTH

api_key_header = APIKeyHeader(
    name = "X-API-KEY",
    auto_error = False
)

def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing API Key"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Invalid API Key"
        )

    return api_key

def validate_question(question: str):
    cleaned_question = question.strip().lower()

    if len(cleaned_question) < 2:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Question is too short"
        )

    if len(cleaned_question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = f"Question is too long. Max length is {MAX_QUESTION_LENGTH} characters."
        )

    for pattern in BLOCKED_PROMPT_PATTERNS:
        if pattern in cleaned_question:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = "This question appears to contain unsafe prompt-injection instructions."
            )