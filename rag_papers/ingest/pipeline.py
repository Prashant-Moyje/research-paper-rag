"""End-to-end ingestion: PDFs -> cleaned -> chunked -> embedded -> indexed."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..schema import Chunk
from .chunk import chunk_papers
from .clean import build_vocabulary
from .embed import Embedder
from .parse import parse_directory
from .store import get_store

log = logging.getLogger(__name__)


def _corpus_fingerprint(papers_dir: Path, cfg: Settings) -> str:
    """Hash inputs + settings so we can skip rebuilds when nothing changed."""
    h = hashlib.sha256()
    for pdf in sorted(papers_dir.glob("*.pdf")):
        stat = pdf.stat()
        h.update(pdf.name.encode())
        h.update(str(stat.st_size).encode())
    for key in (cfg.chunk_tokens, cfg.chunk_overlap, cfg.min_chunk_tokens, cfg.embed_model):
        h.update(str(key).encode())
    return h.hexdigest()[:16]


def build_index(cfg: Settings | None = None, force: bool = False) -> dict:
    """Build (or reuse) the vector index. Returns a manifest dict."""
    cfg = cfg or default_settings
    cfg.validate()

    fingerprint = _corpus_fingerprint(cfg.papers_dir, cfg)
    if not force and cfg.manifest_path.exists():
        try:
            existing = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") == fingerprint:
                log.info("Index is up to date (fingerprint %s). Use --force to rebuild.",
                         fingerprint)
                return existing
        except (json.JSONDecodeError, OSError):
            log.warning("Unreadable manifest; rebuilding.")

    t0 = time.perf_counter()

    log.info("[1/5] Parsing PDFs from %s", cfg.papers_dir)
    papers = parse_directory(cfg.papers_dir)

    log.info("[2/5] Building corpus vocabulary for hyphenation resolution")
    vocab = build_vocabulary([p.full_text for p in papers])

    log.info("[3/5] Loading embedding model %s", cfg.embed_model)
    embedder = Embedder(cfg.embed_model, cfg.query_prefix, cfg.embed_batch)
    if cfg.chunk_tokens > embedder.max_tokens:
        raise ValueError(
            f"RAG_CHUNK_TOKENS={cfg.chunk_tokens} exceeds the model limit of "
            f"{embedder.max_tokens}. Chunks would be silently truncated. "
            f"Lower RAG_CHUNK_TOKENS in .env."
        )

    log.info("[4/5] Chunking (section-aware, target=%d tokens)", cfg.chunk_tokens)
    chunks: list[Chunk] = chunk_papers(
        papers,
        target_tokens=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
        min_tokens=cfg.min_chunk_tokens,
        vocab=vocab,
        token_len=embedder.token_len,
    )
    if not chunks:
        raise RuntimeError("Chunking produced no chunks; check the PDFs and settings.")

    log.info("[5/5] Embedding %d chunks", len(chunks))
    vectors = embedder.embed_passages([c.text for c in chunks], show_progress=True)

    store = get_store(cfg.store, cfg.index_dir)
    store.build(chunks, vectors)
    store.save()

    elapsed = time.perf_counter() - t0
    manifest = {
        "fingerprint": fingerprint,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(elapsed, 2),
        "store": cfg.store,
        "embed_model": cfg.embed_model,
        "embed_dim": embedder.dim,
        "chunk_tokens": cfg.chunk_tokens,
        "chunk_overlap": cfg.chunk_overlap,
        "n_papers": len(papers),
        "n_chunks": len(chunks),
        "papers": [
            {
                "doc_id": p.doc_id,
                "title": p.title,
                "pages": p.n_pages,
                "chunks": sum(1 for c in chunks if c.doc_id == p.doc_id),
            }
            for p in papers
        ],
    }
    cfg.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Index built in %.1fs: %d chunks from %d papers", elapsed, len(chunks), len(papers))
    return manifest
