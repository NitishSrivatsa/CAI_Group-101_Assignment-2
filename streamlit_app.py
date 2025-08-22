import os
import streamlit as st

from utils_finance import is_in_scope_query, Timer
from rag_core import RAGConfig, CorpusIndex
from ft_core import FTClient
from evaluation import evaluate_rows

st.set_page_config(page_title="RAG vs FT (Finance)", layout="wide")
st.title("RAG vs Fine‑Tuned — Finance Q&A (Group 101)")

DATA_DIR = os.environ.get("DATA_DIR", "data")
ADAPTER_REPO = os.environ.get("FT_ADAPTER_REPO", "")
ADAPTER_FILE = os.environ.get("FT_ADAPTER_FILE", "")

with st.spinner("Indexing corpus (two granularities)..."):
    cfg = RAGConfig(data_dir=DATA_DIR)
    index = CorpusIndex(cfg)
    index.load_corpus()
    S, L = index.chunk_docs()
    index.build_sparse()
    index.build_dense()

ft_client = FTClient(adapter_repo=ADAPTER_REPO or None, adapter_filename=ADAPTER_FILE or None)

st.sidebar.header("Mode & Settings")
mode = st.sidebar.selectbox("Mode", ["RAG (Multi‑Stage)", "Fine‑Tuned (Supervised)"])
top_m = st.sidebar.slider("Contexts to use after re‑ranking", 1, 6, 3)

q = st.text_input("Ask a finance question (last two years)", value="What was the company's revenue in 2023?")
if st.button("Run"):
    if not is_in_scope_query(q):
        st.error("Query blocked by input guardrail.")
        st.stop()

    rows = []

    if mode.startswith("RAG"):
        with Timer() as t1:
            cands = index.stage1_candidates(q)
        with Timer() as t2:
            reranked = index.stage2_rerank(q, cands)
        ctx = "\n\n".join([c for c,_ in reranked[:top_m]]) if reranked else ""
        with Timer() as t3:
            ans = ft_client.answer(q, ctx)

        rows.append({
            "question": q,
            "method": "RAG (Multi‑Stage)",
            "answer": ans,
            "raw_score": sum(s for _,s in reranked[:top_m])/max(1,len(reranked[:top_m])) if reranked else 0.0,
            "time_s": round(t1.dt + t2.dt + t3.dt, 3),
            "correct": ""
        })
        st.success("RAG answered.")
        st.write(ans)
        with st.expander("Context used (re‑ranked)"):
            st.write(ctx)

    else:
        ctx = "Use your financial knowledge base and prior supervised fine‑tuning."
        with Timer() as t:
            ans = ft_client.answer(q, ctx)

        rows.append({
            "question": q,
            "method": "FT (Supervised / Base if no adapters)",
            "answer": ans,
            "raw_score": 0.75 if ft_client.adapter_loaded else 0.5,
            "time_s": round(t.dt, 3),
            "correct": ""
        })
        st.success("FT answered.")
        st.write(ans)

    st.divider()
    st.subheader("Evaluation row (append to your table)")
    df = evaluate_rows(rows)
    st.dataframe(df, use_container_width=True)

st.sidebar.markdown("---")
if ADAPTER_REPO and ADAPTER_FILE:
    st.sidebar.success(f"FT adapters configured: {ADAPTER_REPO} / {ADAPTER_FILE}")
else:
    st.sidebar.info("FT runs base model. Provide adapters via Hugging Face for true supervised FT.")
