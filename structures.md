Project: AI Second Brain 

Upload PDF/TXT from Streamlit
        ↓
FastAPI /upload
        ↓
save file into data/
        ↓
extract text
        ↓
chunk text
        ↓
embed chunks
        ↓
write into Haystack DocumentStore
        ↓
/chat answers from uploaded files

Haystack’s OllamaGenerator is used after a PromptBuilder and connects to a local Ollama model by model name and URL.

Adding rag pipeline and basic backend for the first day 

For DAY 2:
for streaming responses:
make /chat/stream return the answer gradually instead of waiting for the full response. FastAPI supports this with StreamingResponse, and Haystack’s Ollama generator supports token streaming through a streaming_callback.

When user asks the same question again:
First time  → full RAG pipeline runs
Second time → cached response returns instantly

