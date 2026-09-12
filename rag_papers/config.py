"""Central configuration, driven entirely by environment variables.

Every tunable in the pipeline lives here so that experiments (chunk size,
embedding model, top-k, LLM backend) are reproducible from a .env file rather
than scattered literals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Paths
    papers_dir: Path = field(default_factory=lambda: _path("RAG_PAPERS_DIR", "data/papers"))
    index_dir: Path = field(default_factory=lambda: _path("RAG_INDEX_DIR", "data/index"))

    # Chunking
    chunk_tokens: int = field(default_factory=lambda: _int("RAG_CHUNK_TOKENS", 700))
    chunk_overlap: float = field(default_factory=lambda: _float("RAG_CHUNK_OVERLAP", 0.15))
    min_chunk_tokens: int = field(default_factory=lambda: _int("RAG_MIN_CHUNK_TOKENS", 50))

    # Embeddings
    embed_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embed_batch: int = field(default_factory=lambda: _int("RAG_EMBED_BATCH", 32))
    query_prefix: str = field(
        default_factory=lambda: os.getenv(
            "RAG_QUERY_PREFIX",
            "Represent this sentence for searching relevant passages:",
        )
    )

    # Store
    store: str = field(default_factory=lambda: os.getenv("RAG_STORE", "numpy").lower())

    # Retrieval
    retriever: str = field(
        default_factory=lambda: os.getenv("RAG_RETRIEVER", "hybrid").lower()
    )
    top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 5))
    rrf_k: int = field(default_factory=lambda: _int("RAG_RRF_K", 5))
    candidates: int = field(default_factory=lambda: _int("RAG_CANDIDATES", 20))
    min_score: float = field(default_factory=lambda: _float("RAG_MIN_SCORE", 0.35))

    # Generation
    llm_backend: str = field(default_factory=lambda: os.getenv("RAG_LLM_BACKEND", "ollama"))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("RAG_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("RAG_OLLAMA_MODEL", "llama3.2:3b")
    )
    temperature: float = field(default_factory=lambda: _float("RAG_LLM_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _int("RAG_LLM_MAX_TOKENS", 512))
    llm_timeout: int = field(default_factory=lambda: _int("RAG_LLM_TIMEOUT", 300))

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / "vectors.npz"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def validate(self) -> None:
        """Fail fast with actionable messages rather than deep stack traces."""
        if not self.papers_dir.exists():
            raise FileNotFoundError(
                f"Papers directory not found: {self.papers_dir}\n"
                f"Create it and add PDFs, or set RAG_PAPERS_DIR in .env"
            )
        if not (0.0 <= self.chunk_overlap < 0.9):
            raise ValueError(f"RAG_CHUNK_OVERLAP must be in [0, 0.9); got {self.chunk_overlap}")
        if self.chunk_tokens < 100:
            raise ValueError(f"RAG_CHUNK_TOKENS too small: {self.chunk_tokens}")
        if self.top_k > self.candidates:
            raise ValueError(
                f"RAG_TOP_K ({self.top_k}) cannot exceed RAG_CANDIDATES ({self.candidates})"
            )
        if self.retriever not in {"dense", "bm25", "hybrid", "rerank"}:
            raise ValueError(
                f"RAG_RETRIEVER must be dense|bm25|hybrid|rerank; got {self.retriever!r}"
            )
        if self.store not in {"numpy", "chroma"}:
            raise ValueError(f"RAG_STORE must be 'numpy' or 'chroma'; got {self.store!r}")
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
