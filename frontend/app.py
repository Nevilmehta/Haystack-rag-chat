import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")


st.set_page_config(
    page_title="AI Second Brain",
    page_icon="🧠",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "last_cached" not in st.session_state:
    st.session_state.last_cached = False


left_col, chat_col, source_col = st.columns([1, 2.2, 1.2])


# LEFT PANEL
with left_col:
    st.title("🧠 AI Second Brain")
    st.caption("Upload documents and chat with your personal knowledge base.")

    api_key = st.text_input(
        "API Key",
        value=API_KEY,
        type="password",
    )

    st.divider()

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
    )

    if uploaded_file is not None:
        if st.button("Upload & Index", use_container_width=True):
            try:
                with st.spinner("Uploading and indexing document..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/upload",
                        headers={"X-API-Key": api_key},
                        files=files,
                        timeout=300,
                    )

                if response.status_code == 200:
                    st.success("Uploaded and indexed successfully.")
                else:
                    st.error(response.text)

            except Exception as error:
                st.error(f"Upload failed: {error}")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_confidence = 0.0
        st.session_state.last_cached = False
        st.rerun()


# MIDDLE CHAT PANEL
with chat_col:
    st.header("Chat")

    chat_box = st.container(height=520)

    with chat_box:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    question = st.chat_input("Ask something from your uploaded documents...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with chat_box:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                full_answer = ""

                try:
                    response = requests.post(
                        f"{API_URL}/chat/stream",
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": api_key,
                        },
                        json={"question": question},
                        stream=True,
                        timeout=300,
                    )

                    if response.status_code != 200:
                        st.error(response.text)
                    else:
                        for line in response.iter_lines():
                            if not line:
                                continue

                            decoded_line = line.decode("utf-8")

                            if not decoded_line.startswith("data: "):
                                continue

                            data_text = decoded_line.replace("data: ", "")

                            if data_text == "[DONE]":
                                break

                            event = json.loads(data_text)
                            event_type = event.get("type")
                            content = event.get("content")

                            if event_type == "token":
                                full_answer += content
                                answer_placeholder.markdown(full_answer)

                            elif event_type == "sources":
                                st.session_state.last_sources = content

                            elif event_type == "confidence":
                                st.session_state.last_confidence = content

                            elif event_type == "cache":
                                st.session_state.last_cached = content

                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_answer}
                        )

                        st.rerun()

                except Exception as error:
                    st.error(f"Request failed: {error}")


# RIGHT PANEL
with source_col:
    st.header("Why this answer?")

    st.metric("Confidence", st.session_state.last_confidence)
    st.metric("Cached", "Yes" if st.session_state.last_cached else "No")

    st.divider()

    sources = st.session_state.last_sources

    if sources:
        for source in sources:
            score = source.get("score")
            score_text = round(score, 3) if score else "N/A"

            with st.expander(
                f"{source.get('source')} | Chunk {source.get('chunk_id')} | Score {score_text}"
            ):
                st.write(source.get("content"))
    else:
        st.info("Ask a question to see retrieved sources.")