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

A more professional flow is:
one chat session = one temporary workspace

So when you upload 2 files in that chat:
only those 2 files are active
Old files should not interfere.

How real systems usually do it:
They don’t just dump everything into one global data/ folder.

They usually have:
user_id
  └── session_id / conversation_id
        ├── uploaded files
        ├── extracted chunks
        ├── embeddings
        └── chat history

So retrieval happens inside:
current session only
Not across every file ever uploaded.

Use:

data/sessions/<session_id>/
  uploaded PDFs for this chat only ✅

New chat opens
↓
session_id created
↓
uploaded files saved to data/sessions/<session_id>/
↓
RAG indexes only that session
↓
questions only use files from that chat
↓
New Chat = new clean session

