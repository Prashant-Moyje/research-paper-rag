"""Streamlit interface for the RAG paper-QA system.

Run with:  streamlit run app.py

Design intent: the evidence is not an optional extra. A RAG answer is only
trustworthy to the extent the user can check it, so retrieved passages, their
scores, and the exact section/page each claim came from are always one click
away — and when the system abstains it says *which* gate fired and shows the
closest passages it rejected, rather than a bare "I don't know".
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_papers.config import settings                  # noqa: E402
from rag_papers.logging_utils import setup_logging      # noqa: E402
from rag_papers.pipeline import Answer, RAGPipeline     # noqa: E402

setup_logging(logging.WARNING)

st.set_page_config(page_title="Paper RAG", page_icon="📄", layout="wide")

SAMPLE_QUESTIONS = [
    "What are the main components of a RAG model, and how do they interact?",
    "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Explain how positional encoding is implemented in Transformers and why it is necessary.",
    "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
    "What is few-shot learning, and how does GPT-3 implement it during inference?",
    "What learning rate did the Mixtral 8x7B paper use?  (not in corpus — should abstain)",
]


@st.cache_resource(show_spinner="Loading index and embedding model…")
def load_pipeline() -> tuple[RAGPipeline | None, str]:
    """Build the pipeline once and reuse it across reruns."""
    try:
        pipeline = RAGPipeline(settings)
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - surface any startup problem in the UI
        return None, f"Failed to initialise pipeline: {exc}"
    return pipeline, ""


def render_sources(answer: Answer) -> None:
    """Show which passages were cited, and let the user read them."""
    cited_ranks = set(answer.citations.cited_indices) if answer.citations else set()

    if cited_ranks:
        st.markdown("**Cited sources**")
        for i, hit in zip(answer.citations.cited_indices, answer.citations.sources):
            st.markdown(f"- `[{i}]` {hit.chunk.citation} — similarity `{hit.score:.3f}`")
    else:
        st.info(
            "The model did not attach inline citations to this answer. "
            "The passages it was given are listed below."
        )

    with st.expander(f"Retrieved evidence ({len(answer.hits)} passages)", expanded=False):
        for i, hit in enumerate(answer.hits, start=1):
            used = "cited" if i in cited_ranks else "retrieved, not cited"
            st.markdown(
                f"**[{i}] {hit.chunk.paper_title}**  \n"
                f"§ {hit.chunk.section or '—'} · "
                f"p.{hit.chunk.page_start}"
                + (f"–{hit.chunk.page_end}" if hit.chunk.page_end != hit.chunk.page_start else "")
                + f" · similarity `{hit.score:.3f}` · _{used}_ · matched by `{hit.source}`"
            )
            st.text(hit.chunk.body[:1500] + ("…" if len(hit.chunk.body) > 1500 else ""))
            if i < len(answer.hits):
                st.divider()


def render_answer(answer: Answer) -> None:
    if not answer.answered:
        st.warning("**Not enough evidence in the indexed papers to answer this.**")
        st.markdown(answer.text)

        reason = {
            "low_retrieval_score": (
                f"No passage cleared the confidence threshold "
                f"(best similarity `{answer.top_score:.3f}` < `{settings.min_score}`). "
                "The question was rejected before calling the language model."
            ),
            "model_declined": (
                f"Passages were retrieved (best similarity `{answer.top_score:.3f}`), "
                "but the model judged them insufficient to answer."
            ),
        }.get(answer.abstain_reason, answer.abstain_reason or "")
        st.caption(f"Why: {reason}")

        if answer.hits:
            with st.expander("Closest passages considered (rejected)", expanded=False):
                for i, hit in enumerate(answer.hits, start=1):
                    st.markdown(f"**[{i}]** {hit.chunk.citation} — `{hit.score:.3f}`")
                    st.text(hit.chunk.body[:600] + "…")
        return

    st.markdown(answer.text)

    if answer.citations and answer.citations.invalid_indices:
        st.warning(
            f"The model referenced non-existent sources "
            f"`{answer.citations.invalid_indices}`; these were removed automatically."
        )

    st.divider()
    render_sources(answer)


def main() -> None:
    st.title("📄 Research Paper Q&A")
    st.caption(
        "Retrieval-Augmented Generation over *Attention Is All You Need*, "
        "*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, "
        "and *Language Models are Few-Shot Learners* (GPT-3)."
    )

    pipeline, error = load_pipeline()

    with st.sidebar:
        st.header("System")
        if pipeline is None:
            st.error(error)
        else:
            healthy, message = pipeline.health()
            (st.success if healthy else st.error)(message)
            st.metric("Indexed chunks", len(pipeline.retriever))

        st.markdown("**Configuration**")
        st.code(
            f"retriever   {settings.retriever}\n"
            f"embeddings  {settings.embed_model.split('/')[-1]}\n"
            f"llm         {settings.ollama_model}\n"
            f"top_k       {settings.top_k}\n"
            f"min_score   {settings.min_score}",
            language="text",
        )
        st.caption(
            "Answers are generated only from retrieved passages. "
            "When the papers do not cover a question, the system says so "
            "instead of guessing."
        )

    if pipeline is None:
        st.info("Build the index first:\n\n```bash\npython scripts/ingest.py\n```")
        st.stop()

    healthy, message = pipeline.health()
    if not healthy:
        st.error(message)
        st.stop()

    st.markdown("**Try a question**")
    cols = st.columns(3)
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        label = sample.split("(")[0].strip()
        if cols[i % 3].button(label[:58] + ("…" if len(label) > 58 else ""),
                              key=f"s{i}", use_container_width=True):
            st.session_state.question = sample.split("  (not in corpus")[0]

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question", ""),
        height=90,
        placeholder="e.g. How is scaled dot-product attention computed?",
    )

    if st.button("Ask", type="primary") and question.strip():
        try:
            with st.spinner("Retrieving evidence and generating an answer…"):
                answer = pipeline.answer(question)
        except ValueError as exc:
            st.error(str(exc))
            return
        except RuntimeError as exc:
            st.error(f"{exc}\n\nIs Ollama running? Try: `ollama serve`")
            return

        st.divider()
        render_answer(answer)

        st.divider()
        cols = st.columns(4)
        cols[0].metric("Retrieval", f"{answer.retrieval_s * 1000:.0f} ms")
        cols[1].metric("Generation", f"{answer.generation_s:.1f} s")
        cols[2].metric("Answer length", f"{answer.completion_tokens} tok")
        coverage = (
            f"{answer.citations.citation_coverage:.0%}" if answer.citations else "—"
        )
        cols[3].metric("Citation coverage", coverage)


if __name__ == "__main__":
    main()
