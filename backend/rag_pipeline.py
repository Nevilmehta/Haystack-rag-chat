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
3. If the answer is not in the context, say:
"I don't know based on the provided notes."
4. Keep the answer clear and useful.

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
        self.document_store = InMemoryDocumentStore()
        self.pipeline = self._build_pipeline()
        self.cache = InMemoryCache(ttl_seconds=3600)

    def reload_documents(self):
        self.document_store = InMemoryDocumentStore()
        self.cache.clear()
        self.pipeline = self._build_pipeline()

    def _index_documents(self):
        documents = load_documents(DATA_DIR)

        if not documents:
            return 

        document_embedder = SentenceTransformersDocumentEmbedder(
            model = EMBEDDING_MODEL
        )
        document_embedder.warm_up()

        embedded_documents = document_embedder.run(documents)["documents"]
        self.document_store.write_documents(embedded_documents)

    def _build_pipeline(self):
        self._index_documents()

        text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)

        bm25_retriever = InMemoryBM25Retriever(
            document_store = self.document_store,
            top_k=TOP_K_EMBEDDING
        )

        embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=TOP_K_EMBEDDING,
        )

        joiner = DocumentJoiner()

        ranker = SentenceTransformersSimilarityRanker(
            model = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k = TOP_K_RERANK
        )

        prompt_builder = PromptBuilder(template=PROMPT_TEMPLATE)

        generator = OllamaGenerator(
            model = OLLAMA_MODEL,
            url = OLLAMA_URL,
            generation_kwargs = {
                "temperature": 0.1,
                "top_p": 0.9
            }
        )

        pipeline = Pipeline()

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

        return pipeline

    def ask(self, question: str):
        cached_result = self.cache.get(question)

        if cached_result:
            cached_result["cached"] = True
            cached_result["response_time_seconds"] = 0.0
            return cached_result

        start_time = time.time()

        result = self.pipeline.run(
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
                    "question": question
                }
            },
            include_outputs_from={"ranker", "generator"}
        )

        answer = result["generator"]["replies"][0]
        retrieved_docs = result["ranker"]["documents"]

        sources = []
        scores = []

        for doc in retrieved_docs:
            scores.append(doc.score)
            sources.append(
                {
                    "source": doc.meta.get("source", "unknown"),
                    "chunk_id": doc.meta.get("chunk_id", -1),
                    "score": doc.score,
                    "content": doc.content
                }
            )

        response = {
            "answer": answer,
            "sources": sources,
            "confidence": calculate_confidence(scores),
            "response_time_seconds": round(time.time() - start_time, 2),
            "cached": False
        }

        self.cache.set(question, response)

        return response

    def ask_stream(self, question: str) -> Generator[str, None, None]:
        cached_result = self.cache.get(question)

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
                        "prompt_builder": {"question": question},
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