import requests
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

st.set_page_config(
    page_title = "AI Second Brain",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Second Brain")
st.caption("Explainable RAG assistant powered by FastAPI + Haystack + Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "API Key",
        value=API_KEY or "",
        type="password",
    )

    st.divider()

    st.subheader("About")
    st.write(
        "This assistant answers using your local notes and shows the retrieved sources."
    )

left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask your second Brain...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": api_key,
                        },
                        json={
                            "question": question,
                        },
                        timeout=120,
                    )

                    if response.status_code != 200:
                        st.error(response.text)
                    else:
                        data = response.json()

                        answer = data["answer"]
                        st.write(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        st.session_state.last_sources = data.get("sources", [])
                        st.session_state.last_confidence = data.get("confidence", 0.0)
                        st.session_state.last_cached = data.get("cached", False)
                        st.session_state.last_time = data.get(
                            "response_time_seconds", 0.0
                        )

                except Exception as error:
                    st.error(f"Request failed: {error}")

with right_col:
    st.subheader("Why this answer?")

    confidence = st.session_state.get("last_confidence", None)
    cached = st.session_state.get("last_cached", None)
    response_time = st.session_state.get("last_time", None)
    sources = st.session_state.get("last_sources", [])

    if confidence is not None:
        st.metric("confidence", confidence)

    if cached is not None:
        st.metric("cached", "Yes" if cached else "No")

    if response_time is not None:
        st.metric("Response Time", f"{response_time}s")

    st.divider()

    if sources:
        st.write("Retrieved Sources")

        for source in sources:
            with st.expander(
                f"{source['source']} | Chunk {source['chunk_id']} | Score {round(source['score'], 3) if source['score'] else 'N/A'}"
            ):
                st.write(source["content"])
    
    else:
        st.info("Ask a question to see retrieved sources.")