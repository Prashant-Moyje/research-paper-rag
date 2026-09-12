# Sample questions and answers

Verbatim output from `python scripts/evaluate.py`. Nothing here is hand-written or edited.

Configuration: `hybrid` retrieval, `bge-small-en-v1.5` embeddings, `llama3.2:3b`, k=5, min_score=0.58.

---

## The five required benchmark questions

### 1. What are the main components of a RAG model, and how do they interact?

> The main components of a RAG model are a retriever pη(z|x) and a generator pθ(yi|x,z,y 1:i-1) [1]. The retriever returns (top-K truncated) distributions over text passages given a queryx, and the generator produces a current token based on a context of the previousi- 1 tokensy1:i-1, the original inputx, and a retrieved passagez [1].

**Sources retrieved**

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2 Methods (pp.2-3)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 6 Discussion (pp.9-10)
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.1 Models (p.3)
4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 6 Discussion (pp.17-18)
5. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 4.5 Additional Results (pp.7-8)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 14.8s`

### 2. What are the two sub-layers in each encoder layer of the Transformer model?

> The two sub-layers in each encoder layer of the Transformer model are: 
1. A multi-head self-attention mechanism, 
2. A simple, position-wise fully connected feed-forward network [1].

**Sources retrieved**

1. Attention Is All You Need § 3.1 Encoder and Decoder Stacks (p.3)
2. Attention Is All You Need § 3.3 Position-wise Feed-Forward Networks (p.5)
3. Attention Is All You Need § 3 Model Architecture (pp.2-3)
4. Attention Is All You Need § 3.2.3 Applications of Attention in our Model (p.5)
5. Attention Is All You Need § 7 Conclusion (p.10)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 4.0s`

### 3. Explain how positional encoding is implemented in Transformers and why it is necessary.

> The positional encoding is implemented as PE (pos,2i) = sin(pos/100002i/dmodel) and PE (pos,2i+1) = cos(pos/100002i/dmodel), where pos is the position and i is the dimension [1]. This is necessary because the model contains no recurrence and no convolution, and therefore needs to inject information about the relative or absolute position of the tokens in the sequence [1].

**Sources retrieved**

1. Attention Is All You Need § 3.5 Positional Encoding (p.6)
2. Attention Is All You Need § Abstract (p.1)
3. Attention Is All You Need § 6.2 Model Variations (pp.8-9)
4. Attention Is All You Need § 3 Model Architecture (pp.2-3)
5. Attention Is All You Need § 6.2 Model Variations (p.9)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 5.5s`

### 4. Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?

> Multi-head attention in the Transformer architecture involves linearly projecting the queries, keys, and values in different dimensions, and then performing the attention function in parallel on each of these projected versions. This results in a concatenated output, which is then projected again to obtain the final values. [1]

This concept is beneficial because it allows the model to jointly attend to information from different representation subspaces at different positions, which is inhibited by single-head attention. By using multiple attention heads, the model can capture more nuanced relationships between different parts of the input sequence. [1]

**Sources retrieved**

1. Attention Is All You Need § 3.2.2 Multi-Head Attention (pp.4-5)
2. Attention Is All You Need § 2 Background (p.2)
3. Attention Is All You Need § 3.2.3 Applications of Attention in our Model (p.5)
4. Attention Is All You Need § 3.2 Attention (pp.3-4)
5. Attention Is All You Need § 6.2 Model Variations (p.9)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 6.3s`

### 5. What is few-shot learning, and how does GPT-3 implement it during inference?

> Few-shot learning is the term used in this work to refer to the setting where the model is given a few demonstrations of the task at inference time as conditioning, but no weight updates are allowed [2]. In this setting, the model is given K examples of context and completion, and then one final example of context, with the model expected to provide the completion [2].

**Sources retrieved**

1. Language Models are Few-Shot Learners (GPT-3) § 5 Limitations (p.34)
2. Language Models are Few-Shot Learners (GPT-3) § 2 Approach (p.6)
3. Language Models are Few-Shot Learners (GPT-3) § 5 Limitations (p.34)
4. Language Models are Few-Shot Learners (GPT-3) § 1 Introduction (p.5)
5. Language Models are Few-Shot Learners (GPT-3) § 1 Introduction (pp.4-5)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 5.1s`

---

## Additional in-corpus questions

### 6. Which optimizer was used to train the Transformer and with what settings?

> The Adam optimizer was used to train the Transformer [1]. The settings were β1 = 0.9, β2 = 0.98, and ε = 10^-9 [1].

**Sources retrieved**

1. Attention Is All You Need § 5.3 Optimizer (p.7)
2. Language Models are Few-Shot Learners (GPT-3) § 2.3 Training Process (p.9)
3. Attention Is All You Need § 7 Conclusion (p.10)
4. Attention Is All You Need § 6.1 Machine Translation (p.8)
5. Attention Is All You Need § 6.2 Model Variations (pp.8-9)

`correctness: partial (50%)` · `groundedness: 100%` · `citations valid: True` · `generation: 3.9s`

> **Known gap:** the answer omits `['warmup|4000']`.

### 7. How is scaled dot-product attention computed, and why is it scaled?

> The scaled dot-product attention is computed as:
Attention(Q,K,V ) = softmax(QK T / √dk ) * V (1)

It is scaled by 1 / √dk to counteract the effect of large dot products for large values of dk, which can push the softmax function into regions with extremely small gradients.

**Sources retrieved**

1. Attention Is All You Need § 3.2.1 Scaled Dot-Product Attention (p.4)
2. Attention Is All You Need § 3.2 Attention (pp.3-4)
3. Attention Is All You Need § Abstract (p.1)
4. Attention Is All You Need § 3.2.2 Multi-Head Attention (pp.4-5)
5. Attention Is All You Need § 3.2.3 Applications of Attention in our Model (p.5)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 4.6s`

### 8. Why did the authors choose self-attention over recurrent layers?

> According to [1], self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece and byte-pair representations.

**Sources retrieved**

1. Attention Is All You Need § 4 Why Self-Attention (pp.6-7)
2. Attention Is All You Need § 4 Why Self-Attention (pp.6-7)
3. Attention Is All You Need § 3.2.3 Applications of Attention in our Model (p.5)
4. Attention Is All You Need § 3.4 Embeddings and Softmax (pp.5-6)
5. Attention Is All You Need § 2 Background (p.2)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 4.6s`

### 9. What retriever does RAG use and what architecture does it follow?

> The retriever used by RAG is DPR, which follows a bi-encoder architecture [4]. Specifically, DPR uses a BERTBASE document encoder and a query encoder, both based on BERTBASE, to calculate the top-k documents with the highest prior probability [4].

**Sources retrieved**

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2 Methods (pp.2-3)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 4.1 Open-domain Question Answering (pp.5-6)
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 4.5 Additional Results (pp.7-8)
4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.2 Retriever: DPR (p.3)
5. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.1 Models (p.3)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 12.6s`

### 10. Which generator model does RAG use and how large is it?

> The generator model used in RAG is BART [4].

**Sources retrieved**

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.1 Models (p.3)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2 Methods (pp.2-3)
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.5 Decoding (p.4)
4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 1 Introduction (p.2)
5. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 6 Discussion (pp.17-18)

`correctness: partial (50%)` · `groundedness: 100%` · `citations valid: True` · `generation: 3.1s`

> **Known gap:** the answer omits `['400M']`.

### 11. What is the difference between RAG-Sequence and RAG-Token?

> The difference between RAG-Sequence and RAG-Token lies in how they treat the retrieved document. In RAG-Sequence, the retrieved document is treated as a single latent variable that is marginalized to get the seq2seq probability p(y|x). In contrast, RAG-Token treats each target token as a different latent document, allowing the generator to choose content from several documents when producing an answer.

**Sources retrieved**

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.1 Models (p.3)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 4.5 Additional Results (p.8)
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2.5 Decoding (p.4)
4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 2 Methods (pp.2-3)
5. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks § 4.5 Additional Results (p.8)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 5.7s`

### 12. What architecture does GPT-3 use and how does it differ from GPT-2?

> GPT-3 uses the same model and architecture as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein [1]. The main difference is that GPT-3 uses alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer [1].

**Sources retrieved**

1. Language Models are Few-Shot Learners (GPT-3) § 2.1 Model and Architectures (p.8)
2. Language Models are Few-Shot Learners (GPT-3) § 3.9.1 Arithmetic (p.22)
3. Language Models are Few-Shot Learners (GPT-3) § 3.3 Translation (p.14)
4. Language Models are Few-Shot Learners (GPT-3) § 6.3 Energy Usage (p.39)
5. Language Models are Few-Shot Learners (GPT-3) § 8 Conclusion (pp.45-46)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 4.9s`

### 13. What data was GPT-3 trained on and how was it filtered?

> GPT-3 was trained on a dataset that includes a mix of Common Crawl (filtered) and WebText2, with the weight in training mix being 60% Common Crawl and 22% WebText2 [1]. The dataset was intentionally not made proportional to the size of the datasets, and some datasets were seen up to 3.4 times during training while others were seen less than once [1].

**Sources retrieved**

1. Language Models are Few-Shot Learners (GPT-3) § 2.2 Training Dataset (p.9)
2. Language Models are Few-Shot Learners (GPT-3) § 3.5 Common Sense Reasoning (p.18)
3. Language Models are Few-Shot Learners (GPT-3) § 3.3 Translation (p.14)
4. Language Models are Few-Shot Learners (GPT-3) § 1 Introduction (pp.5-6)
5. Language Models are Few-Shot Learners (GPT-3) § 8 Conclusion (p.44)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 5.5s`

### 14. What limitations do the authors acknowledge about GPT-3?

> The authors acknowledge several limitations about GPT-3, including:

1. Notable weaknesses in text synthesis, such as repeating themselves semantically at the document level, losing coherence over long passages, and containing non-sequitur sentences or paragraphs [1].
2. Difficulty with "common sense physics" and questions of the type "If I put cheese into the fridge, will it melt?" [1].
3. Notable gaps in in-context learning performance on their suite of benchmarks, particularly in one-shot and few-shot settings [1].
4. Poor performance on certain "comparison" tasks, such as determining if two words are used the same way in a sentence, or if one sentence implies another (WIC and ANLI respectively) [1].
5. Limited performance on a subset of reading comprehension tasks [1].
6. Structural and algorithmic limitations that could account for some of the issues above [2].
7. High cost and inconvenience to perform inference on, which may present a challenge for practical applicability of models of this scale [2].
8. Decisions that are not easily interpretable, and it is not necessarily well-calibrated in its predictions on novel inputs [2].
9. Retaining biases of the data it has been trained on, which may lead to stereotyped or prejudiced content [2].
10. Ambiguity about whether few-shot learning actually learns new tasks "from scratch" at inference time, or if it simply recognizes and identifies tasks that it has learned during training [3].

**Sources retrieved**

1. Language Models are Few-Shot Learners (GPT-3) § 5 Limitations (p.33)
2. Language Models are Few-Shot Learners (GPT-3) § 5 Limitations (p.34)
3. Language Models are Few-Shot Learners (GPT-3) § 5 Limitations (p.34)
4. Language Models are Few-Shot Learners (GPT-3) § 1 Introduction (p.6)
5. Language Models are Few-Shot Learners (GPT-3) § 8 Conclusion (pp.40-41)

`correctness: correct` · `groundedness: 100%` · `citations valid: True` · `generation: 13.4s`

---

## Out-of-corpus questions — the system must refuse

These have no answer in the three indexed papers. A system that answers them is hallucinating.

### What learning rate and batch size did the Mixtral 8x7B paper use?

> The retrieved passages do not contain enough information to answer this question.

`refused correctly` · `generation: 2.6s`

### How does LoRA reduce the cost of fine-tuning large language models?

> The retrieved passages do not contain enough information to answer this question.

`refused correctly` · `generation: 2.6s`

### What is the best recipe for sourdough bread?

> I could not find sufficient evidence in the indexed papers to answer this question.

`refused correctly` · `generation: 0.0s`

### What accuracy did BERT achieve on the SQuAD 2.0 benchmark?

> The retrieved passages do not contain enough information to answer this question.

`refused correctly` · `generation: 2.8s`

---

## Aggregate results

| Metric | Result |
|---|---|
| Retrieval hit rate | 92.9% |
| Correctness | 92.9% |
| Groundedness | 100.0% |
| Citation validity | 100.0% |
| Citation coverage | 69.5% |
| Citation precision | 78.6% |
| Abstention accuracy | 100.0% |
| Hallucinations | 0 |
| Mean retrieval latency | 25 ms |
| Mean generation latency | 6.7 s |
