#!/usr/bin/env python
"""Build the vector index from the PDFs in data/papers/.

Usage:
    python scripts/ingest.py            # build (skips if already up to date)
    python scripts/ingest.py --force    # rebuild from scratch
    python scripts/ingest.py --stats    # show index statistics only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_papers.config import settings           # noqa: E402
from rag_papers.ingest.pipeline import build_index  # noqa: E402
from rag_papers.logging_utils import setup_logging  # noqa: E402


def _print_manifest(manifest: dict) -> None:
    print()
    print("=" * 66)
    print("  INDEX SUMMARY")
    print("=" * 66)
    print(f"  built at     : {manifest.get('built_at')}")
    print(f"  build time   : {manifest.get('build_seconds')}s")
    print(f"  store        : {manifest.get('store')}")
    print(f"  embed model  : {manifest.get('embed_model')} ({manifest.get('embed_dim')}-d)")
    print(f"  chunk target : {manifest.get('chunk_tokens')} tokens "
          f"/ {manifest.get('chunk_overlap')} overlap")
    print(f"  total chunks : {manifest.get('n_chunks')}")
    print("-" * 66)
    for paper in manifest.get("papers", []):
        print(f"  {paper['chunks']:4d} chunks  {paper['pages']:3d}p  {paper['title'][:44]}")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--stats", action="store_true", help="print existing index stats and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.stats:
        if not settings.manifest_path.exists():
            print(f"No index found at {settings.index_dir}. Run: python scripts/ingest.py")
            return 1
        _print_manifest(json.loads(settings.manifest_path.read_text(encoding="utf-8")))
        return 0

    try:
        manifest = build_index(settings, force=args.force)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nIngestion failed: {exc}\n", file=sys.stderr)
        return 1

    _print_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
