from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

from .kb import LEGAL_DOCUMENTS


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for token, count in Counter(re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())).items():
                vec[hash(token) % self.dim] += count
            norm = np.linalg.norm(vec)
            vectors.append((vec / norm).tolist() if norm else vec.tolist())
        return vectors


class InMemoryCollection:
    def __init__(self, docs: List[Dict[str, str]], embeddings: List[List[float]]) -> None:
        self.docs = docs
        self.embeddings = np.array(embeddings, dtype=float)

    def query(self, query_embeddings, n_results: int = 3):
        query = np.array(query_embeddings[0], dtype=float)
        scores = self.embeddings @ query
        order = np.argsort(scores)[::-1][:n_results]
        return {
            "documents": [[self.docs[i]["text"] for i in order]],
            "metadatas": [[{"topic": self.docs[i]["topic"], "id": self.docs[i]["id"]} for i in order]],
            "distances": [[float(1 - scores[i]) for i in order]],
        }


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return HashingEmbedder()


def build_collection(embedder):
    texts = [doc["text"] for doc in LEGAL_DOCUMENTS]
    embeddings = embedder.encode(texts)
    try:
        import chromadb

        client = chromadb.Client()
        collection = client.create_collection("legal_document_assistant")
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[doc["id"] for doc in LEGAL_DOCUMENTS],
            metadatas=[{"topic": doc["topic"]} for doc in LEGAL_DOCUMENTS],
        )
        return collection
    except Exception:
        return InMemoryCollection(LEGAL_DOCUMENTS, embeddings)


def retrieve(question: str, embedder, collection, top_k: int = 3):
    property_terms = {
        "girl",
        "daughter",
        "father",
        "land",
        "property",
        "inheritance",
        "ancestral",
        "share",
        "succession",
    }
    question_terms = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", question.lower()))
    if property_terms & question_terms:
        property_doc = next(
            (doc for doc in LEGAL_DOCUMENTS if doc["topic"] == "Daughter Property Rights"),
            None,
        )
        if property_doc:
            context = f"[{property_doc['topic']}]\n{property_doc['text']}"
            return context, [property_doc["topic"]]

    query_embedding = embedder.encode([question])
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    context_parts = []
    sources = []
    for doc, meta in zip(docs, metas):
        topic = meta.get("topic", "Untitled")
        sources.append(topic)
        context_parts.append(f"[{topic}]\n{doc}")
    return "\n\n".join(context_parts), sources


def retrieval_smoke_test(embedder, collection) -> List[str]:
    _, sources = retrieve("What should I check in an NDA confidentiality clause?", embedder, collection)
    return sources
