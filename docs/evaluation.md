# Evaluation

All numbers below are reproducible with:

```bash
python scripts/eval_retrieval.py     # retrieval strategies
python scripts/evaluate.py           # full end-to-end
```

Configuration: `hybrid` retrieval (RRF, K=5), `bge-small-en-v1.5` embeddings,
`llama3.2:3b` via Ollama, `k=5`, `min_score=0.58`, temperature 0.

---

## 1. Methodology

### Why no LLM-as-judge

The system runs fully locally, so an LLM judge would mean grading a 3B model's
output with a 3B model — weak, and not reproducible between runs. Every metric
here is therefore **deterministic**: the same run produces the same score. That
is a stronger claim than a judged score, not a weaker one, though it comes at
the cost of measuring proxies rather than meaning (see *Limitations*).

### The gold set

18 questions in `rag_papers/eval/goldset.py`:

- **5 sample questions** (the project's required benchmark)
- **9 additional in-corpus questions** spanning all three papers
- **4 out-of-corpus questions** that must be refused

Each answerable question carries `gold_sections` — the section(s) that
genuinely contain the answer. These were established by *reading the parsed
text before writing the question*, so retrieval is scored against ground truth
rather than impression. An automated check confirms every gold label exists in
the index.

### Metrics

| Metric | Definition |
|---|---|
| **Retrieval hit rate** | A chunk from a gold section appeared in the top *k* |
| **MRR** | 1 / rank of the first relevant chunk |
| **Correctness** | Share of required facts present in the answer |
| **Groundedness** | Share of answer sentences supported by retrieved context (cosine ≥ 0.60) |
| **Citation validity** | Answer contains zero fabricated block ids |
| **Citation coverage** | Share of factual sentences carrying a citation |
| **Citation precision** | The *specific* block cited actually supports that sentence |
| **Abstention accuracy** | Out-of-corpus questions correctly refused |
| **False abstention** | Answerable questions wrongly refused |

Retrieval and generation failures are attributed separately: if the gold
section never reached the context it is a **retrieval** failure; if the context
was correct but the answer was not, it is a **generation** failure.

---

## 2. Retrieval results

14 answerable questions, k=5:

| Retriever | Hit@5 | P@1 | MRR | Recall | Latency |
|---|---|---|---|---|---|
| dense only | 78.6% | 64.3% | 0.681 | 75.0% | 22 ms |
| BM25 only | 92.9% | 50.0% | 0.693 | 87.5% | **0.8 ms** |
| **hybrid RRF (adopted)** | **92.9%** | **78.6%** | **0.839** | **89.3%** | 23 ms |
| hybrid + cross-encoder rerank | 85.7% | 57.1% | 0.660 | 82.1% | 2 098 ms |

Hybrid improves on the dense baseline by **+14.3 pp Hit@5** and **+23 % MRR**
for roughly 1 ms of extra latency.

### Two defaults that measurement overturned

**Cross-encoder reranking made retrieval worse.** `ms-marco-MiniLM-L-6-v2` is
trained on short web queries against web prose; these are math-dense academic
passages. It lost 7 pp of Hit@5, broke two previously-correct questions, and
cost 100× the latency. Rejected, but kept selectable (`RAG_RETRIEVER=rerank`)
so the negative result stays reproducible.

**RRF's published K=60 is wrong for this corpus.** With K=60 the
"agreement bonus" dominates absolutely: a chunk ranked #1 by one retriever
scores 1/61, while any chunk appearing in *both* at ranks 2–3 scores
1/62 + 1/63 and always wins. That buried `2.2 Retriever: DPR`, which BM25 ranks
first.

| RRF K | Hit@5 | MRR | Misses |
|---|---|---|---|
| 60 (literature default) | 85.7% | 0.821 | R1, R2 |
| **5 (adopted)** | **92.9%** | **0.839** | R2 |

---

## 3. End-to-end results

18 questions (14 answerable, 4 out-of-corpus):

| Metric | Result |
|---|---|
| Retrieval hit rate | 92.9% |
| Answer rate | 100.0% |
| **Correctness** | **92.9%** |
| **Groundedness** | **100.0%** |
| Citation validity | 100.0% |
| Citation coverage | 69.5% |
| Citation precision | 78.6% |
| **Abstention accuracy** | **100.0%** |
| False abstention | 0.0% |
| **Hallucinations** | **0** |
| Retrieval failures | 1 |
| Generation failures | 1 |

Latency (CPU, 16 cores): retrieval **25 ms**, generation **6.7 s** mean warm
(≈15 s cold), answers ≈86 tokens. Retrieval is ~0.4 % of total time — the local
LLM dominates end to end.

### Abstention behaviour

All four out-of-corpus questions were refused, and the two gates did
**different** work — which is the argument for having both:

| Question | Top score | Gate that fired |
|---|---|---|
| Sourdough bread recipe | 0.452 | **Gate 1** (retrieval) — refused with no LLM call |
| Mixtral hyperparameters | 0.673 | Gate 2 (model declined) |
| LoRA fine-tuning cost | 0.766 | Gate 2 (model declined) |
| BERT SQuAD 2.0 accuracy | 0.672 | Gate 2 (model declined) |

Gate 1 alone would have passed the three plausible ML questions straight to the
model, since they score well above the 0.58 threshold. Gate 2 alone would have
spent an LLM call on the sourdough question. Each covers the other's blind spot.

---

## 4. Failure analysis

### R2 — retrieval failure (unresolved)

> *"Which generator model does RAG use and how large is it?"* → `2.3 Generator: BART`

Dense ranks the gold chunk **31st**, BM25 **5th**, fused **13th**. The RAG paper
contains many chunks discussing "generation" that score well in *both*
retrievers and crowd out the one short section that actually answers the
question; its body is dominated by math notation (`pθ(yi|x,z,y1:i-1)`) which
dilutes its embedding. Confirmed that the gold label is right — §2.1 mentions
neither "BART" nor "400M". Reranking was the obvious fix and empirically made
things worse. The system still produced a partially correct answer (it named
BART from a neighbouring chunk) but could not supply the parameter count.

### T1 — generation failure

> *"Which optimizer was used to train the Transformer and with what settings?"*

Retrieval was perfect (§5.3 at rank 1). The answer gives Adam, β₁, β₂ and ε but
omits the warmup schedule (4000 steps), which the same chunk contains. A
genuine incompleteness typical of a 3B model summarising a dense passage.

### Two bugs this evaluation surfaced

**Bibliography markers collided with context-block numbering.** The generator
copied `[26]` verbatim out of "DPR [26]" and emitted it as a citation when only
5 blocks existed. The validator caught it, but the fix was to strip the papers'
own inline references at ingestion — they are dangling pointers anyway, since
the bibliography is already removed. Corpus-wide: 238 numeric, 186
alphanumeric. Citation validity went **92.3% → 100%**.

**Keyword correctness punished correct answers.** Two of three apparent
"generation failures" were metric artifacts:

- S3 answered with the actual formulas `sin(...)` / `cos(...)`; the required
  facts were spelled "sine"/"cosine", so a *more* precise answer scored 0%.
- T3 answered that self-attention is "faster than recurrent layers when n < d",
  one of several valid reasons §4 gives; the keyword list allowed only
  "complexity".

Both were widened to accept alternatives. This is the central weakness of
keyword-based correctness scoring and is stated here rather than hidden: it
measures lexical overlap with an expected answer, not semantic correctness.

---

## 5. Limitations

- **Groundedness is a similarity proxy, not entailment.** It detects claims with
  no basis in the retrieved context, but cannot catch a claim that *inverts* the
  context's meaning while reusing its vocabulary. A proper NLI model would.
- **Correctness is lexical.** See above.
- **Citation coverage of 69.5% is genuinely imperfect** — roughly three in ten
  factual sentences carry no inline citation, usually introductory or linking
  sentences. The UI mitigates this by always showing retrieved evidence.
- **Citation precision of 78.6%** means about one cited block in five does not
  strongly support the specific sentence citing it, typically when adjacent
  chunks from the same section are interchangeable.
- **18 questions is a small sample.** A single question is worth 7 percentage
  points, so differences below ~10 pp should not be over-read.
- **Latency varies with model warmth** (6.7 s warm vs ≈15 s cold); figures are
  from a warm model on CPU.
- **Figures and tables are not extracted**, so questions answerable only by a
  figure cannot be answered.
