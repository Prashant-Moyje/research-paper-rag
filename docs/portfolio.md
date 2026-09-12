# Portfolio material

## Resume bullet (concise)

> **Retrieval-Augmented Generation system for research-paper Q&A** — Built an
> end-to-end local RAG pipeline over three foundational AI papers (PDF parsing →
> section-aware chunking → embeddings → hybrid retrieval → grounded generation
> with validated citations). Improved retrieval from 78.6% to 92.9% Hit@5 by
> adding BM25 + Reciprocal Rank Fusion, achieving 100% groundedness, 100%
> citation validity, and **zero hallucinations** on an 18-question gold set with
> deterministic evaluation. Python, sentence-transformers, Ollama, Streamlit.

## Resume bullet (shorter)

> Built a local RAG question-answering system over AI research papers; raised
> retrieval Hit@5 from 78.6% → 92.9% with hybrid dense+BM25 retrieval, and
> achieved 0 hallucinations / 100% citation validity via a two-stage abstention
> mechanism and programmatic citation validation.

## LinkedIn / project description

> **Research Paper Q&A — Retrieval-Augmented Generation**
>
> An end-to-end RAG system that answers questions about *Attention Is All You
> Need*, the original *RAG* paper, and *GPT-3* — with every claim traced to a
> specific paper, section and page, and an explicit refusal when the papers
> don't cover the question.
>
> Runs entirely locally (no API keys): pypdf → section-aware chunking →
> bge-small embeddings → hybrid dense + BM25 retrieval with Reciprocal Rank
> Fusion → llama3.2 via Ollama → validated citations.
>
> What I'm most pleased with is that every design decision is backed by a
> measurement rather than a convention. Two widely-used defaults turned out to
> be wrong for this corpus and I have the numbers to show it: the standard
> "retrieve-then-rerank" cross-encoder step *degraded* retrieval by 7 points
> while costing 100× the latency, and RRF's published K=60 constant cost another
> 7 points versus K=5. I kept both experiments in the repo as reproducible
> negative results.
>
> Results: 92.9% retrieval Hit@5, 92.9% answer correctness, 100% groundedness,
> 100% citation validity, 0 hallucinations across an 18-question gold set.

## One-line summary

> Local RAG system for AI-paper Q&A with section-level citations, measured
> hybrid retrieval (+14 pp Hit@5), and a two-gate abstention mechanism that
> produced zero hallucinations.

---

# Interview talking points

## The strongest stories to tell

### 1. "I measured instead of assuming — and two standard practices lost"

The most interesting result is a **negative** one. The conventional RAG recipe
is retrieve → rerank with a cross-encoder. I implemented it and it made things
*worse*: Hit@5 fell 92.9% → 85.7%, P@1 fell 78.6% → 57.1%, and latency rose
from 22 ms to 2 098 ms. It also broke two questions that previously worked.

The likely cause is domain mismatch — `ms-marco-MiniLM` is trained on short web
queries against web prose, while this corpus is math-dense academic text with
section breadcrumbs. I kept the code behind a config flag so the result stays
reproducible rather than deleting the evidence.

Similarly, RRF's published damping constant K=60 (tuned for TREC-scale corpora)
cost 7 points of Hit@5 here. With K=60, a chunk ranked #1 by one retriever
scores 1/61, while any chunk appearing in *both* retrievers at ranks 2–3 scores
1/62 + 1/63 and always wins — so a confident single-retriever hit can never
surface. That buried `2.2 Retriever: DPR`, which BM25 ranks first. K=5 fixed it.

**The point:** published defaults encode assumptions about corpus scale and
domain. Both were wrong here, and only measurement revealed it.

### 2. "A one-character bug that silently breaks keyword search"

Searching `fine-tuning` in the GPT-3 paper returns **zero** hits. The text
contains `ﬁne-tuning` with U+FB01, the `fi` ligature — pypdf emits it because
that's how the PDF encodes the glyph. After NFKC normalisation: 27 hits.

This matters because BM25 would silently fail on one of the corpus's most
important terms, with no error anywhere. Corpus-wide: 444 ligatures in GPT-3,
114 in RAG, 0 in the Transformer paper.

**The point:** data-quality bugs in RAG are usually silent. Nothing crashes;
retrieval just quietly gets worse.

### 3. "Hyphenation where any fixed rule is wrong half the time"

PDFs break words across lines with hyphens, and the hyphen is ambiguous:
`knowl-edge` should join into "knowledge", but `position-wise` must keep its
hyphen. I counted 90 instances in this corpus, split roughly 50/50 — so joining
always or keeping always is wrong about half the time.

It mattered concretely: joining `fine-tuning` → "finetuning" and `few-shot` →
"fewshot" would have broken the GPT-3 benchmark questions.

I resolved it against **corpus vocabulary**: if the joined form appears
elsewhere in the corpus, join; if the hyphenated form appears, keep the hyphen;
otherwise default to keeping it (technical compounds dominate, and BM25 splits
on hyphens anyway). 11/12 correct on inspection.

### 4. "Two abstention gates, because each catches what the other misses"

Everyone says "make RAG refuse when it doesn't know". The interesting question
is *where* to put that check. I have both, and data showing each is necessary:

| Question | Top similarity | Caught by |
|---|---|---|
| Sourdough bread recipe | 0.452 | Gate 1 (similarity floor) — no LLM call |
| Mixtral hyperparameters | 0.673 | Gate 2 (model declines) |
| LoRA fine-tuning cost | 0.766 | Gate 2 |
| BERT SQuAD 2.0 accuracy | 0.672 | Gate 2 |

Gate 1 alone would pass all three plausible ML questions straight through — they
score far above any usable threshold. Gate 2 alone would waste an LLM call on
sourdough. The threshold itself (0.58) was calibrated from a measured score
distribution: in-corpus 0.83–0.85, near-miss 0.61–0.77, gibberish 0.56,
off-topic 0.50. My initial guess of 0.35 would never have fired.

### 5. "The evaluation caught my own metric being wrong"

Two of three apparent "generation failures" were measurement artifacts:

- **S3** answered with the actual formulas `sin(pos/10000^…)` / `cos(…)`. My
  required facts were the *words* "sine"/"cosine", so a **more precise** answer
  scored 0%.
- **T3** answered "self-attention layers are faster than recurrent layers when
  n < d" — one of several valid reasons §4 gives. My keyword list allowed only
  "complexity".

I widened both to accept alternatives and documented this as the central
weakness of keyword-based correctness scoring rather than quietly patching it.

**The point:** when a metric says your system failed, check the metric first.

### 6. "Citations that are validated, not trusted"

An LLM can emit a citation that looks perfect and refers to nothing. Mine did:
it copied `[26]` verbatim out of the passage "DPR [26]" and emitted it as a
context-block citation when only 5 blocks existed.

Two layers of defence:
1. A validator parses every `[n]`, checks it's in range, strips fabricated ones,
   and resolves survivors to paper/section/page.
2. The root cause was fixed at ingestion — the papers' own bibliography markers
   (238 numeric + 186 alphanumeric) collide with my numbering scheme and are
   dangling pointers anyway, since the bibliography is already stripped.

Citation validity went 92.3% → 100%.

---

## Likely interview questions

### Architecture

**Q: Why section-aware chunking rather than fixed-size?**
Every benchmark answer lives inside one *named* section. A fixed-size splitter
cuts blindly and would sever the two-sub-layer explanation in §3.1, leaving
neither half independently answerable. Splitting on headings keeps explanations
whole, and the section label doubles as exact citation metadata. I also embed a
`Paper › Section` breadcrumb inside each chunk, because queries here echo
section titles.

**Q: Why 400-token chunks?**
Because `bge-small-en-v1.5` truncates at 512 tokens **silently**. I originally
configured 700 and would have lost ~30% of every chunk with no error. The
ingest pipeline now raises if `chunk_tokens > model limit`, and chunk sizes are
enforced against the real tokenizer, not a word-count estimate.

**Q: Why not use a vector database?**
284 chunks × 384 dims is a 0.4 MB matrix. A brute-force cosine scan is *exact*
and sub-millisecond; ANN indexing trades exactness for a speed-up that is
unmeasurable at this scale. ChromaDB would add ~30 transitive packages
(kubernetes, grpcio, onnxruntime, opentelemetry) to search 284 vectors. I kept
a `ChromaStore` behind the same Protocol so the swap is one config line — the
right answer changes when the corpus outgrows memory or needs concurrent
writers.

**Q: Why hybrid retrieval?**
Dense embeddings capture meaning but blur rare surface forms, and this corpus is
full of them — `DPR`, `BART`, `Adam`, `LAMBADA`. Dense retrieval *never*
surfaced `2.2 Retriever: DPR` even though the section heading contains the
answer; BM25 ranks it first. Measured: +14.3 pp Hit@5, +23% MRR, ~1 ms cost.

**Q: Why RRF instead of weighted score blending?**
Dense cosine lives on roughly [0.4, 0.9]; BM25 is unbounded and corpus-dependent.
Combining raw scores needs a normalisation constant that is itself a tuned
parameter and drifts as the corpus changes. RRF ignores magnitudes and fuses
ranks, so it needs no such constant — only K, which I tuned by measurement.

### Evaluation

**Q: How do you know retrieval is actually working?**
A gold set where each question is labelled with the section(s) that genuinely
contain the answer — established by reading the parsed text *before* writing the
question. A hit means "a chunk from a gold section reached the context", not
"looked relevant". An automated check confirms every gold label exists in the
index, so the gold set can't silently drift from the corpus.

**Q: How do you separate retrieval failures from generation failures?**
If the gold section never reached the context, it's a retrieval failure. If the
context was right but the answer was wrong, it's generation. In the final run:
1 retrieval failure (R2), 1 generation failure (T1). Without that split, you
can't tell whether to fix the retriever or the prompt.

**Q: Why no LLM-as-judge?**
The system is fully local, so a judge would mean grading a 3B model's output
with a 3B model — weak and unreproducible. Every metric here is deterministic,
so the same run produces the same score. The trade-off is that I measure proxies
(lexical overlap for correctness, embedding similarity for groundedness) rather
than meaning, which I state explicitly in the limitations.

**Q: What does "100% groundedness" actually mean — and not mean?**
Every answer sentence has a semantically similar sentence in the retrieved
context (cosine ≥ 0.60). It does **not** mean the answers are logically entailed.
The metric would not catch a claim that *inverts* the context's meaning while
reusing its vocabulary — "the encoder has three sub-layers" would score as
grounded. A proper NLI model is the fix, and it's on the improvements list.

### Failure and honesty

**Q: What doesn't work?**
One retrieval miss in the gold set. "Which generator model does RAG use and how
large is it?" should retrieve §2.3 `Generator: BART`. Dense ranks it 31st, BM25
5th, fused 13th. The RAG paper has many chunks discussing "generation" that
score well in *both* retrievers and crowd out the one short section that
answers it; its body is dominated by math notation that dilutes the embedding.
I verified the gold label is correct (§2.1 mentions neither BART nor 400M).
Reranking was the obvious fix and empirically made things worse. It's
documented, not hidden.

**Q: How would you fix it?**
Query-side expansion — expand "generator model" toward likely entity names
before retrieval — or a section-title-boosted index that weights heading matches
more heavily. I'd also try a reranker trained on scientific text rather than
MS MARCO web passages.

**Q: What would you do differently at 10,000 documents?**
The exact NumPy scan stops being free, so a real ANN index (FAISS/Chroma)
earns its place — the Protocol is already there. BM25 would move to a proper
inverted index. Chunk metadata filtering (by paper/year/section) becomes
valuable for narrowing before search. And the gold set would need to be much
larger — 18 questions gives ~7 points of resolution per question, which is too
coarse to detect small regressions.

### Production concerns

**Q: How is this reproducible?**
Pinned dependencies; all tunables in environment variables rather than scattered
literals; a corpus fingerprint (file sizes + chunking/embedding settings) so the
index rebuilds only when inputs actually change; temperature 0; local models so
there's no silent provider-side version drift. Evaluation is two commands.

**Q: What breaks first in production?**
Ingestion, on a PDF unlike these three — a scanned paper (no text layer; the
parser raises a clear error saying OCR is required rather than indexing
garbage), or a two-column layout, where pypdf interleaves columns and would
produce incoherent chunks. My section-header regex also assumes numbered
headings; a paper using unnumbered headings falls back to whole-document
windows, which degrades citation precision.

**Q: How would you monitor this live?**
Log retrieval top-score distribution (drift signals corpus/query mismatch),
abstention rate (a spike means the corpus no longer covers what users ask),
citation validity rate (a drop means the model is fabricating), and per-stage
latency. The abstention rate is the most useful single number — it's a direct
proxy for "are we being asked things we can actually answer".
