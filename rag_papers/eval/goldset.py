"""Evaluation gold set.

Every entry was grounded by reading the referenced section in the parsed text
before the question was written, so `gold_sections` is ground truth rather than
a guess. That is what makes retrieval scoring objective here: a hit is "did a
retrieved chunk come from a section that genuinely contains the answer",
not "did it look relevant".

`must_include` holds short factual tokens that a correct answer should mention.
They give a coarse but objective correctness signal. A fact may offer
alternatives separated by "|", satisfied if any one appears.

Alternatives are not a convenience - without them the metric punishes correct
answers. Two cases found during evaluation:

  * S3 asks about positional encoding. The paper writes the formulas as
    "sin(...)" / "cos(...)", and a correct answer quoting them contains neither
    the word "sine" nor "cosine". Requiring the prose spelling scored a
    precise, correct answer as 0%.
  * T3 asks why self-attention was chosen. Section 4 gives several valid
    reasons (computational complexity, parallelisation, path length, and being
    faster when n < d); requiring one specific term would fail an answer that
    correctly cites another.

This is the central weakness of keyword-based correctness scoring, and it is
worth stating plainly: it measures lexical overlap with an expected answer, not
semantic correctness.

The five questions marked `sample=True` are the project's required benchmark
questions; the rest broaden coverage so a change in retrieval is measured on
more than five data points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRANSFORMER = "1706.03762v7"
RAG = "2005.11401v4"
GPT3 = "2005.14165v4"


@dataclass
class GoldQuestion:
    qid: str
    question: str
    doc_id: str | None                       # None for out-of-corpus
    gold_sections: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    answerable: bool = True
    sample: bool = False                     # one of the 5 required questions


GOLD: list[GoldQuestion] = [
    # ---------------- the five required sample questions ----------------
    GoldQuestion(
        qid="S1",
        question="What are the main components of a RAG model, and how do they interact?",
        doc_id=RAG,
        gold_sections=["2 Methods", "2.1 Models", "2.2 Retriever: DPR", "2.3 Generator: BART"],
        must_include=["retriever", "generator"],
        sample=True,
    ),
    GoldQuestion(
        qid="S2",
        question="What are the two sub-layers in each encoder layer of the Transformer model?",
        doc_id=TRANSFORMER,
        gold_sections=["3.1 Encoder and Decoder Stacks"],
        must_include=["self-attention", "feed-forward"],
        sample=True,
    ),
    GoldQuestion(
        qid="S3",
        question=(
            "Explain how positional encoding is implemented in Transformers and why it "
            "is necessary."
        ),
        doc_id=TRANSFORMER,
        gold_sections=["3.5 Positional Encoding"],
        must_include=["sine|sin(", "cosine|cos("],
        sample=True,
    ),
    GoldQuestion(
        qid="S4",
        question=(
            "Describe the concept of multi-head attention in the Transformer architecture. "
            "Why is it beneficial?"
        ),
        doc_id=TRANSFORMER,
        gold_sections=["3.2.2 Multi-Head Attention"],
        must_include=["subspaces"],
        sample=True,
    ),
    GoldQuestion(
        qid="S5",
        question="What is few-shot learning, and how does GPT-3 implement it during inference?",
        doc_id=GPT3,
        gold_sections=["2 Approach"],
        must_include=["demonstrations", "no weight updates"],
        sample=True,
    ),
    # ---------------- additional in-corpus questions ----------------
    GoldQuestion(
        qid="T1",
        question="Which optimizer was used to train the Transformer and with what settings?",
        doc_id=TRANSFORMER,
        gold_sections=["5.3 Optimizer"],
        must_include=["Adam", "warmup|4000"],
    ),
    GoldQuestion(
        qid="T2",
        question="How is scaled dot-product attention computed, and why is it scaled?",
        doc_id=TRANSFORMER,
        gold_sections=["3.2.1 Scaled Dot-Product Attention"],
        must_include=["softmax"],
    ),
    GoldQuestion(
        qid="T3",
        question="Why did the authors choose self-attention over recurrent layers?",
        doc_id=TRANSFORMER,
        gold_sections=["4 Why Self-Attention"],
        must_include=["complexity|parallel|path length|faster"],
    ),
    GoldQuestion(
        qid="R1",
        question="What retriever does RAG use and what architecture does it follow?",
        doc_id=RAG,
        gold_sections=["2.2 Retriever: DPR"],
        must_include=["DPR", "bi-encoder"],
    ),
    GoldQuestion(
        qid="R2",
        question="Which generator model does RAG use and how large is it?",
        doc_id=RAG,
        gold_sections=["2.3 Generator: BART"],
        must_include=["BART", "400M"],
    ),
    GoldQuestion(
        qid="R3",
        question="What is the difference between RAG-Sequence and RAG-Token?",
        doc_id=RAG,
        gold_sections=["2.1 Models"],
        must_include=["RAG-Sequence", "RAG-Token"],
    ),
    GoldQuestion(
        qid="G1",
        question="What architecture does GPT-3 use and how does it differ from GPT-2?",
        doc_id=GPT3,
        gold_sections=["2.1 Model and Architectures"],
        must_include=["sparse attention"],
    ),
    GoldQuestion(
        qid="G2",
        question="What data was GPT-3 trained on and how was it filtered?",
        doc_id=GPT3,
        gold_sections=["2.2 Training Dataset"],
        must_include=["Common Crawl"],
    ),
    GoldQuestion(
        qid="G3",
        question="What limitations do the authors acknowledge about GPT-3?",
        doc_id=GPT3,
        gold_sections=["5 Limitations"],
        must_include=["repeat"],
    ),
    # ---------------- out-of-corpus (must abstain) ----------------
    GoldQuestion(
        qid="X1",
        question="What learning rate and batch size did the Mixtral 8x7B paper use?",
        doc_id=None,
        answerable=False,
    ),
    GoldQuestion(
        qid="X2",
        question="How does LoRA reduce the cost of fine-tuning large language models?",
        doc_id=None,
        answerable=False,
    ),
    GoldQuestion(
        qid="X3",
        question="What is the best recipe for sourdough bread?",
        doc_id=None,
        answerable=False,
    ),
    GoldQuestion(
        qid="X4",
        question="What accuracy did BERT achieve on the SQuAD 2.0 benchmark?",
        doc_id=None,
        answerable=False,
    ),
]

ANSWERABLE = [q for q in GOLD if q.answerable]
UNANSWERABLE = [q for q in GOLD if not q.answerable]
SAMPLES = [q for q in GOLD if q.sample]
