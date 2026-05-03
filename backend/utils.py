from pathlib import Path
from haystack import Document

def split_text(text: str, chunk_size: int = 350, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        
        start += chunk_size - overlap

    return chunks

def load_text_documents(data_dir: Path):
    documents = []

    for file_path in data_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        chunks = split_text(text)

        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    content = chunk,
                    meta = {
                        "source": file_path.name,
                        "chunk_id": index
                    }
                )
            )
        
    return documents

def calculate_confidence(scores: list[float | None]):
    valid_scores = [score for score in scores if score is not None]

    if not valid_scores:
        return 0.0

    avg_score = sum(valid_scores) / len(valid_scores)

    return round(min(max(avg_score, 0.0), 1.0), 2)