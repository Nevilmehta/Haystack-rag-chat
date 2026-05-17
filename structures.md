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

backend and streamlit is working ,
and chat with streaming also working with responses
Adding conversational memory, because then your app feels:
ChatGPT for documents
instead of:
single-question RAG demo

🧠 Conversational Memory RAG
with:
previous messages
context-aware retrieval
follow-up question handling

Q1: Who is Nevil Mehta?
Q2: What frameworks does he use?

The second question should understand “he” from the previous chat.
