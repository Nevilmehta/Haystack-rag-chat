from pathlib import Path
from haystack import Document
from pypdf import PdfReader

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

def read_pdf(file_path: Path):
    reader = PdfReader(str(file_path))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text

def load_documents(data_dir: Path):
    documents = []

    for file_path in data_dir.glob("*"):
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")

        elif file_path.suffix.lower() == ".pdf":
            text = read_pdf(file_path)

        else:
            continue

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    content = chunk,
                    meta = {
                        "source": file_path.name,
                        "chunk_id": index,
                        "file_type": file_path.suffix.lower()
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