# Research Paper Q&A — a Retrieval-Augmented Generation system

Ask natural-language questions about three foundational AI papers and get
answers grounded in the papers themselves, with citations down to the section
and page — and an explicit refusal when the papers don't cover the question.

Runs **fully locally**. No API keys, no cloud calls, no cost.

```
Q: What are the two sub-layers in each encoder layer of the Transformer model?

A: The two sub-layers in each encoder layer of the Transformer model are:
   1. A multi-head self-attention mechanism,
   2. A simple, position-wise fully connected feed-forward network [1].

   [1] Attention Is All You Need § 3.1 Encoder and Decoder Stacks (p.3)  — similarity 0.785

   retrieval 40 ms | generation 4.1 s
```

```
Q: What learning rate did the Mixtral 8x7B paper use?

A: Not enough evidence in the indexed papers to answer this.
   Why: passages were retrieved (best similarity 0.673), but the model judged
   them insufficient. → 0 hallucinations across the evaluation set.
```

---

## Results

Measured on 18 gold questions (14 answerable + 4 out-of-corpus), reproducible
with `python scripts/evaluate.py`:

| Metric | Result |
|---|---|
| Retrieval hit rate | **92.9%** |
| Answer correctness | **92.9%** |
| Groundedness | **100%** |
| Citation validity | **100%** |
| Abstention accuracy | **100%** |
| Hallucinations | **0** |
| False abstentions | **0** |
| Retrieval latency | 25 ms |
| Generation latency | 6.7 s (CPU) |

Retrieval strategies, compared on the same gold set:

| Retriever | Hit@5 | P@1 | MRR | Latency |
|---|---|---|---|---|
| dense only | 78.6% | 64.3% | 0.681 | 22 ms |
| BM25 only | 92.9% | 50.0% | 0.693 | 0.8 ms |
| **hybrid RRF** ← adopted | **92.9%** | **78.6%** | **0.839** | 23 ms |
| hybrid + cross-encoder rerank | 85.7% | 57.1% | 0.660 | 2 098 ms |

Full methodology, failure analysis and limitations: **[docs/evaluation.md](docs/evaluation.md)**.

---

## Corpus

| Paper | arXiv | Pages | Chunks |
|---|---|---|---|
| Attention Is All You Need | 1706.03762 | 15 | 34 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2005.11401 | 19 | 56 |
| Language Models are Few-Shot Learners (GPT-3) | 2005.14165 | 75 | 194 |

---

## Quick start

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) running locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the local model (~2 GB)
ollama pull llama3.2:3b

# 3. Configure (defaults work as-is)
cp .env.example .env

# 4. Build the index from the PDFs in data/papers/  (~60s, one time)
python scripts/ingest.py

# 5. Ask something
python scripts/query.py "How is scaled dot-product attention computed?"
```

Then launch the web UI:

```bash
streamlit run app.py
```

> **If Ollama returns `"<model>" does not support chat`** — restart the Ollama
> server. A freshly pulled model can be rejected until the server is restarted,
> even though `ollama list` shows it.

### Other commands

```bash
python scripts/query.py --interactive          # REPL
python scripts/query.py --show-context "..."   # print retrieved passages
python scripts/ingest.py --stats               # index statistics
python scripts/ingest.py --force               # rebuild the index
python scripts/eval_retrieval.py               # compare retrieval strategies
python scripts/evaluate.py                     # full end-to-end evaluation
pytest -q                                      # 60 tests
```

---

## How it works

```
PDFs ─► parse ─► normalise ─► section-aware chunk ─► embed ─► vector store
                                                  └─────────► BM25 index

query ─┬─► dense  ─┐
       └─► BM25   ─┴─► RRF fusion ─► Gate 1: score ≥ 0.58?
                                        │ no  → abstain (no LLM call)
                                        │ yes → prompt with numbered blocks
                                               └─► LLM ─► Gate 2: declared insufficient?
                                                            │ yes → abstain
                                                            │ no  → validate citations ─► answer
```

Full diagram and per-component rationale: **[docs/architecture.md](docs/architecture.md)**.

### Design decisions worth noting

**Section-aware chunking.** Every benchmark answer lives inside one *named*
section, so chunks never cross a heading, and each chunk embeds a
`Paper › Section` breadcrumb. Queries in this domain echo section titles
("positional encoding", "multi-head attention"), which makes the heading
high-signal for retrieval — and exact metadata for citations.

**Hybrid retrieval.** Dense embeddings blur rare surface forms; this corpus is
full of them (`DPR`, `BART`, `Adam`, `LAMBADA`). Dense retrieval alone never
surfaced `2.2 Retriever: DPR`, whose own heading contains the answer. BM25 ranks
it first. RRF fuses ranks, so neither needs score normalisation.

**Two abstention gates.** A cheap similarity floor rejects off-topic questions
before any LLM call; a prompt-level sentinel catches plausible-but-uncovered
questions that score highly. Measured, each catches what the other misses.

**Citations are validated, not trusted.** The model emits `[n]`; a validator
confirms each id exists, strips fabricated ones, and resolves survivors to
paper/section/page.

**An exact NumPy index instead of a vector database.** 284 × 384 floats is
0.4 MB — a brute-force cosine scan is *exact* and sub-millisecond, while
ChromaDB would add ~30 transitive packages (kubernetes, grpcio, onnxruntime,
opentelemetry) to search 284 vectors. A `ChromaStore` sits behind the same
interface (`RAG_STORE=chroma`) for when the corpus outgrows memory.

---

## Configuration

Everything is environment-driven (`.env`). Notable knobs:

| Variable | Default | Notes |
|---|---|---|
| `RAG_RETRIEVER` | `hybrid` | `dense` \| `bm25` \| `hybrid` \| `rerank` |
| `RAG_RRF_K` | `5` | Tuned by measurement; the literature default of 60 is worse here |
| `RAG_TOP_K` | `5` | Chunks passed to the LLM |
| `RAG_MIN_SCORE` | `0.58` | Abstention threshold, calibrated from the score distribution |
| `RAG_CHUNK_TOKENS` | `400` | Must stay under the embedding model's 512 limit |
| `RAG_OLLAMA_MODEL` | `llama3.2:3b` | Swap to a larger local model for quality |
| `RAG_STORE` | `numpy` | `numpy` \| `chroma` |

---

## Known limitations

- **One retrieval failure in the gold set** (`2.3 Generator: BART`): dense ranks
  it 31st and BM25 5th, so fusion lands it at 13. Many chunks discussing
  "generation" out-compete the one short section that answers the question.
- **Citation coverage is 69.5%** — roughly three in ten factual sentences carry
  no inline citation. The UI always shows retrieved evidence to compensate.
- **Groundedness is a similarity proxy, not entailment.** It catches
  unsupported claims but cannot catch a claim that *inverts* the context while
  reusing its vocabulary.
- **Correctness scoring is lexical**, so it measures overlap with an expected
  answer rather than semantic correctness.
- **Figures and tables are not extracted**, so anything answerable only by a
  figure is out of reach.
- **18 questions is a small sample** — one question is worth ~7 percentage
  points; differences under ~10 pp shouldn't be over-read.
- **A 3B model summarising dense passages drops details** — e.g. it reports
  Adam's β₁/β₂/ε but omits the warmup schedule from the same chunk.

## Possible improvements

1. **Fix the R2 retrieval miss** with query-side expansion (expand "generator
   model" to likely entity names) or a section-title-boosted index.
2. **Replace the groundedness proxy with a real NLI model** for entailment-level
   faithfulness checking.
3. **Sentence-level attribution** — cite the specific sentence, not the chunk,
   to lift citation precision above 78.6%.
4. **Try a reranker trained on scientific text** (e.g. SPECTER-style) rather
   than the MS MARCO cross-encoder that failed here.
5. **Add tables and figure captions** via a layout-aware parser.
6. **Multi-hop questions** that need synthesis across two papers.
7. **Streaming generation** so answers appear as they're produced.

---

## Project layout

```
rag_papers/        ingest · retrieve · generate · eval  (see docs/architecture.md)
scripts/           ingest.py · query.py · evaluate.py · eval_retrieval.py
tests/             60 tests
docs/              architecture.md · evaluation.md · sample_qa.md · portfolio.md
data/papers/       source PDFs
data/index/        generated index (gitignored)
app.py             Streamlit UI
```

## Tests

```bash
pytest -q     # 60 passed
```

---

## Licence and sources

Source code is MIT licensed — see [LICENSE](LICENSE).

The PDFs in `data/papers/` are arXiv preprints included so the pipeline is
reproducible end to end. They remain the copyright of their authors and are not
covered by the MIT licence:

- Vaswani et al., *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Brown et al., *Language Models are Few-Shot Learners* — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

Tests cover normalisation rules (each one tied to a measured artifact), chunking
invariants (including the oversized-table guard), citation validation, RRF
fusion, and the evaluation metrics themselves.
