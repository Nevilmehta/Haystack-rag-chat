from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

TOP_K_BM25 = 3
TOP_K_EMBEDDING = 3
TOP_K_RERANK = 2

API_KEY = os.getenv("API_KEY")
MAX_QUESTION_LENGTH = 1000

BLOCKED_PROMPT_PATTERNS = [
    "ignore previous instructions",
    "ignore the previous instructions",
    "reveal system prompt",
    "show system prompt",
    "developer message",
    "system message",
    "bypass",
    "jailbreak",
]