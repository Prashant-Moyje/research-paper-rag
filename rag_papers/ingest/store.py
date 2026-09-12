"""Vector storage behind a small interface.

Design note (this is a deliberate, measured choice, not laziness):

This corpus is 264 chunks x 384 dimensions - a 0.4 MB matrix. A brute-force
cosine scan over it is *exact* and takes well under a millisecond, whereas an
approximate-nearest-neighbour index trades exactness for a speed-up that is
unmeasurable at this scale. ChromaDB would also pull in kubernetes, grpcio,
onnxruntime and opentelemetry - roughly 30 transitive packages - which is hard
to justify for searching 264 vectors.

So `NumpyStore` is the default: zero extra dependencies, exact results, and
trivially inspectable. `ChromaStore` is kept behind the same Protocol to show
the swap is a one-line config change (RAG_STORE=chroma) and to make the
trade-off concrete rather than theoretical. Use Chroma when the corpus outgrows
memory or needs server-side filtering and concurrent writers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from ..schema import Chunk

log = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Minimal contract every backend must satisfy."""

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...
    def __len__(self) -> int: ...


class NumpyStore:
    """Exact cosine search over an in-memory matrix, persisted as .npz + JSON."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_dir / "vectors.npz", vectors=self.vectors)
        (self.index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved %d chunks to %s", len(self.chunks), self.index_dir)

    def load(self) -> None:
        vec_path = self.index_dir / "vectors.npz"
        chunk_path = self.index_dir / "chunks.json"
        if not vec_path.exists() or not chunk_path.exists():
            raise FileNotFoundError(
                f"No index found in {self.index_dir}. Build it first:\n"
                f"    python scripts/ingest.py"
            )
        self.vectors = np.load(vec_path)["vectors"].astype(np.float32)
        self.chunks = [
            Chunk.from_dict(d)
            for d in json.loads(chunk_path.read_text(encoding="utf-8"))
        ]
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError(
                f"Corrupt index: {len(self.chunks)} chunks but {self.vectors.shape[0]} vectors. "
                f"Rebuild with: python scripts/ingest.py --force"
            )

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not len(self):
            return []
        # Vectors are L2-normalised at encode time, so a dot product is cosine.
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.chunks)


class ChromaStore:
    """Optional ChromaDB backend (RAG_STORE=chroma). Same interface."""

    def __init__(self, index_dir: Path, collection: str = "papers") -> None:
        self.index_dir = Path(index_dir) / "chroma"
        self.collection_name = collection
        self._client = None
        self._collection = None
        self._by_id: dict[str, Chunk] = {}

    def _connect(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "RAG_STORE=chroma requires chromadb.\n"
                    "Install it with:  pip install chromadb\n"
                    "(or set RAG_STORE=numpy to use the default exact store)"
                ) from exc
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.index_dir))
        return self._client

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        client = self._connect()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = client.create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "paper_title": c.paper_title,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in chunks
            ],
        )
        self._by_id = {c.chunk_id: c for c in chunks}

    def save(self) -> None:
        # PersistentClient writes through on add(); chunk metadata is mirrored
        # to JSON so the two backends stay interchangeable.
        (Path(self.index_dir).parent).mkdir(parents=True, exist_ok=True)
        (Path(self.index_dir).parent / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._by_id.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        client = self._connect()
        self._collection = client.get_collection(self.collection_name)
        path = Path(self.index_dir).parent / "chunks.json"
        self._by_id = {
            d["chunk_id"]: Chunk.from_dict(d)
            for d in json.loads(path.read_text(encoding="utf-8"))
        }

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()], n_results=k
        )
        out: list[tuple[Chunk, float]] = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            chunk = self._by_id.get(cid)
            if chunk is not None:
                out.append((chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    def __len__(self) -> int:
        return len(self._by_id)


def get_store(name: str, index_dir: Path) -> VectorStore:
    """Factory: resolve RAG_STORE to a backend."""
    name = (name or "numpy").lower()
    if name == "numpy":
        return NumpyStore(index_dir)
    if name == "chroma":
        return ChromaStore(index_dir)
    raise ValueError(f"Unknown store {name!r}; expected 'numpy' or 'chroma'")
