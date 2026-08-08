"""
rag_engine.py
Core Retrieval-Augmented Generation (RAG) components for the Teacher Intelligence Agent.

Pipeline: JSON transcripts -> chunk -> embed (SentenceTransformers) -> FAISS index
        -> semantic search -> Groq LLM summarization.
"""
from pathlib import Path
from typing import List, Any, Optional, Dict
import os
import pickle

import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "faiss_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load tutoring transcripts (JSON) from a directory into LangChain Document objects.
    Each source file is a list of {"role": ..., "text": ..., "timestamp": ...} turns.
    """
    data_path = Path(data_dir).resolve()
    documents: List[Any] = []

    for json_file in data_path.glob("**/*.json"):
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[] | {role: .role, text: .text}",
                text_content=False,
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    return documents


class EmbeddingPipeline:
    """Splits transcripts into chunks and embeds them with a SentenceTransformer model."""

    def __init__(self, model_name: str = EMBED_MODEL, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings)


class FaissVectorStore:
    """Persistent FAISS vector store with metadata alongside each embedding."""

    def __init__(self, persist_dir: str = PERSIST_DIR, embedding_model: str = EMBED_MODEL,
                 chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.persist_dir = persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]) -> None:
        emb_pipe = EmbeddingPipeline(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n_vectors, dim).")
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Missing index or metadata at {self.persist_dir}. Build first.")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    def ensure_built(self, data_dir: str = DATA_DIR) -> None:
        """Load an existing index, or build one from the data directory if none exists."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.load()
        else:
            docs = load_all_documents(data_dir)
            self.build_from_documents(docs)


class RAGSearch:
    """Retrieves relevant transcript chunks and summarizes them with a Groq LLM."""

    def __init__(self, persist_dir: str = PERSIST_DIR, embedding_model: str = EMBED_MODEL,
                 llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.ensure_built()
        self.llm = ChatGroq(model=llm_model, temperature=0)

    def search_and_summarize(self, query: str, top_k: int = TOP_K) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        chunks = [r["metadata"]["text"] for r in results if r.get("metadata")]
        context = "\n\n".join(chunks) if chunks else "[No relevant transcript context found.]"
        prompt = (
            "You are a tutoring analytics assistant. Using only the transcript excerpts below, "
            f"answer the question concisely and concretely.\n\nExcerpts:\n{context}\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt).content
