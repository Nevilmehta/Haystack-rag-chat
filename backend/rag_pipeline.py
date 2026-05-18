import time

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from backend.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    TOP_K_BM25,
    TOP_K_EMBEDDING,
    TOP_K_RERANK,
)
from backend.utils import load_documents, calculate_confidence
from backend.cache import InMemoryCache

from queue import Queue
from threading import Thread
from typing import Generator
import json

PROMPT_TEMPLATE = """
You are an AI Second Brain assistant.

STRICT RULES:
1. Use ONLY the provided context.
2. Do not invent facts.
3. Use conversation history only to understand follow-up references like "he", "she", "it", "that project", or "those skills".
4. If the answer is not in the context, say:
"I don't know based on the provided notes."
5. Keep the answer clear and useful.

CONVERSATION HISTORY:
{% for message in history %}
{{ message.role }}: {{ message.content }}
{% endfor %}

CONTEXT:
{% for doc in documents %}
[Source: {{ doc.meta.source }} | Chunk: {{ doc.meta.chunk_id }}]
{{ doc.content }}

{% endfor %}

QUESTION:
{{ question }}

ANSWER:
"""

class RAGService:
    def __init__(self):
        self.sessions = {}
        self.cache = InMemoryCache(ttl_seconds=3600)

    def _get_session_dir(self, session_id: str):
        session_dir = DATA_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _build_pipeline_for_session(self, session_id: str):
        session_dir = self._get_session_dir(session_id)

        document_store = InMemoryDocumentStore()

        documents = load_documents(session_dir)

        if documents:
            document_embedder = SentenceTransformersDocumentEmbedder(
                model=EMBEDDING_MODEL
            )

            document_embedder.warm_up()

            embedded_documents = document_embedder.run(documents)["documents"]

            document_store.write_documents(embedded_documents)

        text_embedder = SentenceTransformersTextEmbedder(
            model=EMBEDDING_MODEL
        )

        bm25_retriever = InMemoryBM25Retriever(
            document_store=document_store,
            top_k=TOP_K_BM25,
        )

        embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=TOP_K_EMBEDDING,
        )

        joiner = DocumentJoiner()

        ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=TOP_K_RERANK,
        )

        prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)

        generator = OllamaGenerator(
            model=OLLAMA_MODEL,
            url=OLLAMA_URL,
            generation_kwargs={
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )

        pipeline = Pipeline()

        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("embedding_retriever", embedding_retriever)
        pipeline.add_component("joiner", joiner)
        pipeline.add_component("ranker", ranker)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("generator", generator)

        pipeline.connect(
            "text_embedder.embedding",
            "embedding_retriever.query_embedding",
        )

        pipeline.connect(
            "bm25_retriever.documents",
            "joiner.documents",
        )

        pipeline.connect(
            "embedding_retriever.documents",
            "joiner.documents",
        )

        pipeline.connect(
            "joiner.documents",
            "ranker.documents",
        )

        pipeline.connect(
            "ranker.documents",
            "prompt_builder.documents",
        )

        pipeline.connect(
            "prompt_builder.prompt",
            "generator.prompt",
        )

        self.sessions[session_id] = {
            "document_store": document_store,
            "pipeline": pipeline,
        }

        return pipeline

    def reload_session(self, session_id: str):
        self.cache.clear()
        return self._build_pipeline_for_session(session_id)

    def get_pipeline(self, session_id: str):
        if session_id not in self.sessions:
            return self._build_pipeline_for_session(session_id)

        return self.sessions[session_id]["pipeline"]

    def ask_stream(
        self,
        question: str,
        session_id: str,
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:

        history = history or []

        pipeline = self.get_pipeline(session_id)

        cache_key = session_id + question + str(history)

        cached_result = self.cache.get(cache_key)

        if cached_result:
            yield f"data: {json.dumps({'type': 'token', 'content': cached_result['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'content': cached_result['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'content': cached_result['confidence']})}\n\n"
            yield f"data: {json.dumps({'type': 'cache', 'content': True})}\n\n"
            yield "data: [DONE]\n\n"
            return

        token_queue = Queue()

        sources_holder = {
            "sources": [],
            "confidence": 0.0,
        }

        full_answer_parts = []

        def streaming_callback(chunk):
            token = getattr(chunk, "content", "")

            if token is None or token == "":
                return

            full_answer_parts.append(token)
            token_queue.put(token)

        def run_pipeline():
            try:
                generator = OllamaGenerator(
                    model=OLLAMA_MODEL,
                    url=OLLAMA_URL,
                    generation_kwargs={
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                    streaming_callback=streaming_callback,
                )

                pipeline.components["generator"] = generator

                result = pipeline.run(
                    {
                        "text_embedder": {
                            "text": question
                        },

                        "bm25_retriever": {
                            "query": question
                        },

                        "ranker": {
                            "query": question
                        },

                        "prompt_builder": {
                            "question": question,
                            "history": history,
                        },
                    },
                    include_outputs_from={"ranker"},
                )

                docs = result.get("ranker", {}).get("documents", [])

                sources = [
                    {
                        "source": doc.meta.get("source", "unknown"),
                        "chunk_id": doc.meta.get("chunk_id", -1),
                        "score": doc.score,
                        "content": doc.content,
                    }
                    for doc in docs
                ]

                confidence = calculate_confidence(
                    [doc.score for doc in docs]
                )

                answer = "".join(full_answer_parts)

                sources_holder["sources"] = sources
                sources_holder["confidence"] = confidence

                self.cache.set(
                    cache_key,
                    {
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "response_time_seconds": 0.0,
                        "cached": False,
                    },
                )

            except Exception as error:
                token_queue.put(f"[ERROR] {str(error)}")

            finally:
                token_queue.put(None)

        thread = Thread(target=run_pipeline)
        thread.start()

        while True:
            token = token_queue.get()

            if token is None:
                break

            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'content': sources_holder['sources']})}\n\n"

        yield f"data: {json.dumps({'type': 'confidence', 'content': sources_holder['confidence']})}\n\n"

        yield f"data: {json.dumps({'type': 'cache', 'content': False})}\n\n"

        yield "data: [DONE]\n\n"
        cached_result = self.cache.get(question)

        history = history or []

        if cached_result:
            yield f"data: {json.dumps({'type': 'token', 'content': cached_result['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'content': cached_result['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'content': cached_result['confidence']})}\n\n"
            yield f"data: {json.dumps({'type': 'cache', 'content': True})}\n\n"
            yield "data: [DONE]\n\n"
            return

        token_queue = Queue()
        sources_holder = {
            "sources": [],
            "confidence": 0.0,
        }
        full_answer_parts = []

        def streaming_callback(chunk):
            token = getattr(chunk, "content", "")

            # Ignore final metadata-only streaming chunks
            if token is None or token == "":
                return

            full_answer_parts.append(token)
            token_queue.put(token)

        def run_pipeline():
            try:
                generator = OllamaGenerator(
                    model=OLLAMA_MODEL,
                    url=OLLAMA_URL,
                    generation_kwargs={
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                    streaming_callback=streaming_callback,
                )

                pipeline = Pipeline()

                text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)

                bm25_retriever = InMemoryBM25Retriever(
                    document_store=self.document_store,
                    top_k=TOP_K_BM25,
                )

                embedding_retriever = InMemoryEmbeddingRetriever(
                    document_store=self.document_store,
                    top_k=TOP_K_EMBEDDING,
                )

                joiner = DocumentJoiner()

                ranker = SentenceTransformersSimilarityRanker(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_k=TOP_K_RERANK,
                )

                prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)

                pipeline.add_component("text_embedder", text_embedder)
                pipeline.add_component("bm25_retriever", bm25_retriever)
                pipeline.add_component("embedding_retriever", embedding_retriever)
                pipeline.add_component("joiner", joiner)
                pipeline.add_component("ranker", ranker)
                pipeline.add_component("prompt_builder", prompt_builder)
                pipeline.add_component("generator", generator)

                pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
                pipeline.connect("bm25_retriever.documents", "joiner.documents")
                pipeline.connect("embedding_retriever.documents", "joiner.documents")
                pipeline.connect("joiner.documents", "ranker.documents")
                pipeline.connect("ranker.documents", "prompt_builder.documents")
                pipeline.connect("prompt_builder.prompt", "generator.prompt")

                result = pipeline.run(
                    {
                        "text_embedder": {"text": question},
                        "bm25_retriever": {"query": question},
                        "ranker": {"query": question},
                        "prompt_builder": {"question": question, "history": history},
                    },
                    include_outputs_from={"ranker"}
                )

                print("PIPELINE RESULT KEYS:", result.keys())
                print("FULL RESULT:", result)

                docs = result.get("ranker", {}).get("documents", [])

                sources = [
                    {
                        "source": doc.meta.get("source", "unknown"),
                        "chunk_id": doc.meta.get("chunk_id", -1),
                        "score": doc.score,
                        "content": doc.content,
                    }
                    for doc in docs
                ]

                confidence = calculate_confidence([doc.score for doc in docs])
                answer = "".join(full_answer_parts)

                sources_holder["sources"] = sources
                sources_holder["confidence"] = confidence

                self.cache.set(
                    question,
                    {
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "response_time_seconds": 0.0,
                        "cached": False,
                    },
                )

            except Exception as error:
                token_queue.put(f"[ERROR] {str(error)}")

            finally:
                token_queue.put(None)

        thread = Thread(target=run_pipeline)
        thread.start()

        while True:
            token = token_queue.get()

            if token is None:
                break

            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'content': sources_holder['sources']})}\n\n"
        yield f"data: {json.dumps({'type': 'confidence', 'content': sources_holder['confidence']})}\n\n"
        yield f"data: {json.dumps({'type': 'cache', 'content': False})}\n\n"
        yield "data: [DONE]\n\n"