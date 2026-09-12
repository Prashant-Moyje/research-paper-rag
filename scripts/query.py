#!/usr/bin/env python
"""Ask the RAG system a question from the command line.

Usage:
    python scripts/query.py "What are the two sub-layers in each encoder layer?"
    python scripts/query.py --interactive
    python scripts/query.py --show-context "..."
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_papers.config import settings              # noqa: E402
from rag_papers.logging_utils import setup_logging  # noqa: E402
from rag_papers.pipeline import Answer, RAGPipeline  # noqa: E402

RULE = "=" * 74


def render(answer: Answer, show_context: bool = False) -> None:
    print()
    print(RULE)
    print(f"Q: {answer.question}")
    print(RULE)
    print()
    print(answer.text)
    print()

    if answer.answered and answer.citations and answer.citations.sources:
        print("-" * 74)
        print("SOURCES")
        for i, hit in zip(answer.citations.cited_indices, answer.citations.sources):
            print(f"  [{i}] {hit.chunk.citation}   (score {hit.score:.3f})")
    elif answer.answered:
        print("-" * 74)
        print("SOURCES (retrieved, but the model did not cite them inline)")
        for hit in answer.hits:
            print(f"  - {hit.chunk.citation}   (score {hit.score:.3f})")
    else:
        print("-" * 74)
        print(f"ABSTAINED ({answer.abstain_reason}); best retrieval score "
              f"{answer.top_score:.3f} vs threshold {settings.min_score}")
        if answer.hits:
            print("Closest passages considered:")
            for hit in answer.hits[:3]:
                print(f"  - {hit.chunk.citation}   (score {hit.score:.3f})")

    if answer.citations and answer.citations.invalid_indices:
        print(f"\n  ! model cited non-existent blocks: {answer.citations.invalid_indices}")

    if show_context:
        print("-" * 74)
        print("RETRIEVED CONTEXT")
        for i, hit in enumerate(answer.hits, 1):
            print(f"\n[{i}] {hit.chunk.citation}  (score {hit.score:.3f})")
            print(hit.chunk.body[:600] + ("..." if len(hit.chunk.body) > 600 else ""))

    print("-" * 74)
    cov = f"{answer.citations.citation_coverage:.0%}" if answer.citations else "n/a"
    print(
        f"retrieval {answer.retrieval_s:.2f}s | generation {answer.generation_s:.1f}s "
        f"| total {answer.total_s:.1f}s | {answer.completion_tokens} tok "
        f"| citation coverage {cov}"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="REPL mode")
    parser.add_argument("-k", type=int, default=None, help="number of chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="print retrieved passages")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.WARNING)

    try:
        pipeline = RAGPipeline(settings)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    ok, message = pipeline.health()
    if not ok:
        print(f"\nLLM unavailable: {message}\n", file=sys.stderr)
        return 1

    if args.interactive:
        print(f"RAG over {len(pipeline.retriever)} chunks. Ctrl-C or 'quit' to exit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"quit", "exit", "q"}:
                return 0
            if not q:
                continue
            try:
                render(pipeline.answer(q, k=args.k), args.show_context)
            except (ValueError, RuntimeError) as exc:
                print(f"Error: {exc}", file=sys.stderr)

    question = " ".join(args.question).strip()
    if not question:
        parser.error("provide a question, or use --interactive")

    try:
        render(pipeline.answer(question, k=args.k), args.show_context)
    except (ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
