import json
from pathlib import Path

import streamlit as st

from frontend.api_client import APIError, health

RAGAS_PATH = Path(__file__).parent.parent / "eval" / "ragas_scores.json"

_METRIC_META = {
    "faithfulness":      ("Faithfulness",       "Answer grounded in retrieved context"),
    "answer_relevancy":  ("Answer Relevancy",   "Answer addresses the question asked"),
    "context_precision": ("Context Precision",  "Retrieved chunks were relevant"),
    "context_recall":    ("Context Recall",     "Context covered the ground truth"),
}


def _color_for(val: float) -> str:
    if val >= 0.75:
        return "normal"
    if val >= 0.50:
        return "off"
    return "inverse"


def render() -> None:
    st.header("System Metrics")

    # ── Live API health ───────────────────────────────────────────────────────
    st.subheader("Live API Status")
    try:
        h = health()
        status = h.get("status", "unknown")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API Status",      status.upper())
        c2.metric("FAISS Chunks",    h.get("chunks_count", 0))
        c3.metric("BM25 Corpus",     h.get("corpus_chunks", 0))
        c4.metric("Retrieval Mode",  h.get("retrieval_mode", "—").replace("_", " "))

        c5, c6 = st.columns(2)
        c5.metric("Documents Loaded", h.get("documents_loaded", 0))
        c6.metric("LLM Available",    "Yes" if h.get("api_key_valid") else "No")

        if status != "healthy":
            st.warning(
                "API is in **degraded** mode — check GROQ_API_KEY or upload a document.",
                icon="⚠️",
            )

    except APIError as e:
        st.error(f"Cannot reach API: {e.detail}", icon="🔴")
        st.code("uv run uvicorn main:app --port 10000", language="bash")

    st.divider()

    # ── RAGAS scores ──────────────────────────────────────────────────────────
    st.subheader("RAGAS Evaluation Scores")

    if RAGAS_PATH.exists():
        with open(RAGAS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        ts = data.get("timestamp", "")[:19]

        st.caption(
            f"**{data.get('num_questions', '?')} questions**  ·  "
            f"Retrieval: `{data.get('retrieval_mode', '?')}`  ·  "
            f"Evaluated: {ts}"
        )

        # Score cards row
        cols = st.columns(4)
        for col, (key, (label, desc)) in zip(cols, _METRIC_META.items()):
            val = scores.get(key)
            if val is not None:
                delta_color = _color_for(float(val))
                col.metric(label, f"{float(val):.3f}", delta_color=delta_color)
                col.caption(desc)
            else:
                col.metric(label, "N/A")
                col.caption(desc)

        st.divider()

        # Bar chart
        import pandas as pd

        chart_data = pd.DataFrame(
            {
                "Score": [
                    scores[k] for k in _METRIC_META if k in scores
                ]
            },
            index=[v[0] for k, v in _METRIC_META.items() if k in scores],
        )
        st.bar_chart(chart_data, height=260, use_container_width=True)

    else:
        st.info(
            "No RAGAS scores found yet. Run the evaluation script to populate this panel:",
            icon="📊",
        )
        st.code("uv run python eval/run_eval.py", language="bash")
        st.caption(
            "Evaluation takes ~3–5 min for 10 questions × 4 metrics. "
            "Make sure the FAISS index exists first (start the server once)."
        )

    st.divider()

    # ── Retrieval pipeline reference ──────────────────────────────────────────
    st.subheader("Retrieval Pipeline")
    st.markdown(
        """
| Stage | Method | What it solves |
|-------|--------|---------------|
| Query Rewriting | LLM (llama-3.1-8b) | Vocabulary mismatch — user phrasing ≠ document language |
| Dense Retrieval | FAISS MMR | Semantic similarity, diverse chunk selection |
| Sparse Retrieval | BM25Okapi | Exact keyword / acronym / formula matches |
| Fusion | Reciprocal Rank Fusion (k=60) | Combines incompatible score scales using rank positions |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | High-precision relevance on the fused candidate set |
"""
    )

    st.divider()

    # ── RAGAS metric glossary ─────────────────────────────────────────────────
    st.subheader("What each RAGAS metric measures")
    for key, (label, _) in _METRIC_META.items():
        with st.expander(label):
            explanations = {
                "faithfulness": (
                    "The LLM breaks the answer into individual claims and checks each "
                    "against the retrieved context. Score = supported claims / total claims. "
                    "A low score means the LLM hallucinated facts not present in the documents."
                ),
                "answer_relevancy": (
                    "Generates N synthetic questions that the answer could be answering, "
                    "embeds them, and measures cosine similarity to the original question. "
                    "A low score means the answer is off-topic or too vague."
                ),
                "context_precision": (
                    "The LLM judges which retrieved chunks were actually useful for "
                    "generating the answer. Score = useful chunks / total retrieved. "
                    "A low score means the retriever is pulling in irrelevant chunks."
                ),
                "context_recall": (
                    "Each sentence of the ground-truth answer is checked against the "
                    "retrieved context. Score = attributable sentences / total sentences. "
                    "A low score means the retriever missed important passages."
                ),
            }
            st.write(explanations[key])
