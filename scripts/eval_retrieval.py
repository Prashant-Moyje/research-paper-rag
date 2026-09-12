#!/usr/bin/env python
"""Compare retrieval strategies on the gold set.

This is the experiment that decided the project's retrieval design. It is a
script rather than a notebook so the numbers in the README can be regenerated
on demand:

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 5 --json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_papers.config import settings                              # noqa: E402
from rag_papers.eval.goldset import ANSWERABLE                      # noqa: E402
from rag_papers.eval.retrieval_metrics import (                     # noqa: E402
    RetrievalReport,
    score_query,
)
from rag_papers.logging_utils import setup_logging                  # noqa: E402


def build_retrievers(include_rerank: bool):
    """Construct each strategy, sharing one loaded index."""
    from rag_papers.retrieve.bm25 import BM25Retriever
    from rag_papers.retrieve.dense import DenseRetriever
    from rag_papers.retrieve.hybrid import HybridRetriever

    dense = DenseRetriever(settings)
    lexical = BM25Retriever(settings, chunks=dense.store.chunks)
    hybrid = HybridRetriever(settings, dense=dense, lexical=lexical)

    configs = [("dense", dense), ("bm25", lexical), ("hybrid (RRF)", hybrid)]

    if include_rerank:
        from rag_papers.retrieve.rerank import RerankingRetriever

        configs.append(("hybrid + rerank", RerankingRetriever(hybrid, settings)))
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies.")
    parser.add_argument("--k", type=int, default=5, help="chunks retrieved per query")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also evaluate the cross-encoder reranker (slow; measured as a regression)",
    )
    parser.add_argument("--json", type=str, default=None, help="write results to a JSON file")
    args = parser.parse_args()

    setup_logging(logging.WARNING)

    try:
        configs = build_retrievers(args.rerank)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    reports: list[RetrievalReport] = []
    for name, retriever in configs:
        results = []
        for gold in ANSWERABLE:
            t0 = time.perf_counter()
            hits = retriever.retrieve(gold.question, k=args.k)
            results.append(score_query(gold, hits, time.perf_counter() - t0))
        reports.append(RetrievalReport(name, results))

    print()
    print(f"Gold set: {len(ANSWERABLE)} answerable questions | k={args.k}")
    print()
    print(RetrievalReport.header())
    for report in reports:
        print(report.summary_row())

    print()
    print("Rank of first relevant chunk per question (lower is better, MISS = not found)")
    print(f"{'qid':<5}" + "".join(f"{r.name[:15]:>17}" for r in reports))
    for i, gold in enumerate(ANSWERABLE):
        row = f"{gold.qid:<5}"
        for report in reports:
            res = report.results[i]
            row += f"{(str(res.first_rank) if res.hit else 'MISS'):>17}"
        print(row)

    best = max(reports, key=lambda r: (r.mrr, r.hit_rate))
    print()
    print(f"Best by MRR: {best.name}  (Hit@{args.k} {best.hit_rate:.1%}, MRR {best.mrr:.3f})")

    if args.json:
        payload = {
            "k": args.k,
            "n_questions": len(ANSWERABLE),
            "reports": [
                {
                    "name": r.name,
                    "hit_rate": r.hit_rate,
                    "p_at_1": r.p_at_1,
                    "mrr": r.mrr,
                    "recall": r.recall,
                    "mean_latency_ms": r.mean_latency * 1000,
                    "per_question": [
                        {
                            "qid": q.qid,
                            "hit": q.hit,
                            "first_rank": q.first_rank,
                            "recall": q.recall,
                            "sections_found": q.sections_found,
                        }
                        for q in r.results
                    ],
                }
                for r in reports
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
