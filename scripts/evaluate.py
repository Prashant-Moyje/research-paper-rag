#!/usr/bin/env python
"""Full end-to-end evaluation: retrieval, generation, citations, abstention.

Runs every gold question through the real pipeline and reports metrics that are
reproducible (no LLM judge), separating retrieval failures from generation
failures.

    python scripts/evaluate.py
    python scripts/evaluate.py --json docs/evaluation_results.json
    python scripts/evaluate.py --samples-only        # the 5 required questions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_papers.config import settings                                   # noqa: E402
from rag_papers.eval.answer_metrics import (                             # noqa: E402
    AnswerEvaluator,
    EvaluationReport,
)
from rag_papers.eval.goldset import GOLD, SAMPLES                        # noqa: E402
from rag_papers.eval.retrieval_metrics import is_relevant                # noqa: E402
from rag_papers.logging_utils import setup_logging                       # noqa: E402
from rag_papers.pipeline import RAGPipeline                              # noqa: E402

BAR = "=" * 78


def _pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system end to end.")
    parser.add_argument("--json", type=str, default=None, help="write full results to JSON")
    parser.add_argument("--samples-only", action="store_true",
                        help="evaluate only the 5 required sample questions")
    parser.add_argument("-k", type=int, default=None, help="chunks retrieved per query")
    args = parser.parse_args()

    setup_logging(logging.WARNING)

    questions = SAMPLES if args.samples_only else GOLD
    answerable_qids = {g.qid for g in questions if g.answerable}
    unanswerable_qids = {g.qid for g in questions if not g.answerable}

    try:
        pipeline = RAGPipeline(settings)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    ok, message = pipeline.health()
    if not ok:
        print(f"\nLLM unavailable: {message}\n", file=sys.stderr)
        return 1

    evaluator = AnswerEvaluator(pipeline.retriever.dense.embedder
                                if hasattr(pipeline.retriever, "dense")
                                else pipeline.retriever.embedder)

    print()
    print(f"Evaluating {len(questions)} questions "
          f"({len(answerable_qids)} answerable, {len(unanswerable_qids)} out-of-corpus)")
    print(f"retriever={settings.retriever}  model={settings.ollama_model}  "
          f"k={args.k or settings.top_k}  min_score={settings.min_score}")
    print()

    scores = []
    t_start = time.perf_counter()
    for i, gold in enumerate(questions, start=1):
        answer = pipeline.answer(gold.question, k=args.k)
        hit = any(is_relevant(h, gold) for h in answer.hits) if gold.answerable else False
        score = evaluator.score(gold, answer, retrieval_hit=hit)
        scores.append((gold, answer, score))
        flag = "ok " if score.failure_mode is None else score.failure_mode[:4].upper()
        print(f"  [{i:2d}/{len(questions)}] {gold.qid:<4} {flag:<5} "
              f"correct={_pct(score.correctness):>6} ground={_pct(score.groundedness):>6} "
              f"{answer.generation_s:5.1f}s  {gold.question[:40]}")
    wall = time.perf_counter() - t_start

    report = EvaluationReport([s for _, _, s in scores])
    summary = report.summary(answerable_qids, unanswerable_qids)

    print()
    print(BAR)
    print("  EVALUATION SUMMARY")
    print(BAR)
    print(f"  Retrieval hit rate     {_pct(summary['retrieval_hit_rate']):>8}"
          "   gold section present in retrieved context")
    print(f"  Answer rate            {_pct(summary['answer_rate']):>8}"
          "   answerable questions actually answered")
    print(f"  Correctness            {_pct(summary['correctness']):>8}"
          "   required facts present in the answer")
    print(f"  Groundedness           {_pct(summary['groundedness']):>8}"
          "   answer sentences supported by context")
    print("  " + "-" * 74)
    print(f"  Citation validity      {_pct(summary['citation_validity']):>8}"
          "   answers with zero fabricated block ids")
    print(f"  Citation coverage      {_pct(summary['citation_coverage']):>8}"
          "   factual sentences carrying a citation")
    print(f"  Citation precision     {_pct(summary['citation_precision']):>8}"
          "   cited block genuinely supports the claim")
    print("  " + "-" * 74)
    print(f"  Abstention accuracy    {_pct(summary['abstention_accuracy']):>8}"
          "   out-of-corpus questions correctly refused")
    print(f"  False abstention       {_pct(summary['false_abstention']):>8}"
          "   answerable questions wrongly refused")
    print(f"  Hallucinations         {summary['hallucinations']:>8}"
          "   out-of-corpus questions answered anyway")
    print("  " + "-" * 74)
    print(f"  Retrieval failures     {summary['retrieval_failures']:>8}"
          "   gold section never retrieved")
    print(f"  Generation failures    {summary['generation_failures']:>8}"
          "   context was right, answer was not")
    print("  " + "-" * 74)
    print(f"  Mean retrieval          {summary['mean_retrieval_s'] * 1000:7.1f}ms")
    print(f"  Mean generation         {summary['mean_generation_s']:7.1f}s")
    print(f"  Mean answer length      {summary['mean_completion_tokens']:7.0f} tokens")
    print(f"  Total wall time         {wall:7.1f}s")
    print(BAR)

    failures = [(g, a, s) for g, a, s in scores if s.failure_mode]
    if failures:
        print()
        print("  FAILURE DETAIL")
        print("  " + "-" * 74)
        for gold, _answer, score in failures:
            print(f"  {gold.qid} [{score.failure_mode}] {gold.question[:58]}")
            if score.missing_facts:
                print(f"        missing facts: {score.missing_facts}")
            if not score.retrieval_hit and gold.answerable:
                print(f"        gold sections never retrieved: {gold.gold_sections}")
            for sent in score.unsupported_sentences[:2]:
                print(f"        unsupported: {sent[:66]}...")
        print()

    if args.json:
        payload = {
            "config": {
                "retriever": settings.retriever,
                "model": settings.ollama_model,
                "embed_model": settings.embed_model,
                "top_k": args.k or settings.top_k,
                "min_score": settings.min_score,
            },
            "summary": summary,
            "wall_seconds": round(wall, 1),
            "questions": [
                {
                    **asdict(score),
                    "question": gold.question,
                    "answer": answer.text,
                    "sources": [h.chunk.citation for h in answer.hits],
                }
                for gold, answer, score in scores
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"  Wrote {args.json}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
