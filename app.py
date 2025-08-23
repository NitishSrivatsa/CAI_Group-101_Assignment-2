import os, time, json, pickle, zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st

# ===== Constants for auditability =====
GROUP_ID = 101
GROUP_TECH = "Multi-Stage Retrieval"
REQUIRE_BM25 = True   # enforce Dense+Sparse at Stage-1

# ===== Repo paths & setup =====
APP_DIR = Path(__file__).parent
ART_DIR = APP_DIR / "artifacts"
ZIP_DIR = ART_DIR / "zips"
RAG_DST = ART_DIR / "rag_index"
FT_DST  = ART_DIR / "ft_model"
for p in [ZIP_DIR, RAG_DST, FT_DST]: p.mkdir(parents=True, exist_ok=True)

RAG_ZIP_URL = (st.secrets.get("RAG_ZIP_URL", "") or os.getenv("RAG_ZIP_URL", "")).strip()
FT_ZIP_URL  = (st.secrets.get("FT_ZIP_URL",  "") or os.getenv("FT_ZIP_URL",  "")).strip()

def _download(url: str, dst: Path):
    import requests
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)

def _ensure_zip(name: str, url: str) -> Path:
    zp = ZIP_DIR / name
    if zp.exists(): return zp
    if url:
        _download(url, zp)
        return zp
    # else expect already committed
    assert zp.exists(), f"Missing {name}. Commit under artifacts/zips or set {name.replace('.zip','').upper()}_URL"
    return zp

def _extract(zip_path: Path, dst_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

def prepare_artifacts():
    if not any(RAG_DST.iterdir()):
        _extract(_ensure_zip("rag_index.zip", RAG_ZIP_URL), RAG_DST)
    if not any(FT_DST.iterdir()):
        _extract(_ensure_zip("ft_model.zip",  FT_ZIP_URL),  FT_DST)

# ===== Loaders =====
@st.cache_resource(show_spinner=False)
def load_ft_pipeline():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
    cfg = AutoConfig.from_pretrained(FT_DST)
    tok = AutoTokenizer.from_pretrained(FT_DST)
    if getattr(cfg, "is_encoder_decoder", False):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(FT_DST)
        pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")
        kind = "seq2seq"
    else:
        mdl = AutoModelForCausalLM.from_pretrained(FT_DST)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
        kind = "causal"
    return pipe, kind

@st.cache_resource(show_spinner=False)
def load_rag_components():
    """
    Expect in artifacts/rag_index:
      - index.faiss, bm25.pkl, texts.pkl
      - emb_model.txt, rerank_model.txt
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    faiss_path = RAG_DST / "index.faiss"
    bm25_path  = RAG_DST / "bm25.pkl"
    texts_path = RAG_DST / "texts.pkl"
    emb_name   = (RAG_DST / "emb_model.txt").read_text().strip() if (RAG_DST / "emb_model.txt").exists() else "sentence-transformers/all-MiniLM-L6-v2"
    rr_name    = (RAG_DST / "rerank_model.txt").read_text().strip() if (RAG_DST / "rerank_model.txt").exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"

    assert faiss_path.exists(), f"Missing {faiss_path}"
    assert texts_path.exists(), f"Missing {texts_path}"

    index = faiss.read_index(str(faiss_path))
    with open(texts_path, "rb") as f:
        passages_raw = pickle.load(f)

    # normalize to list[dict]
    passages: List[Dict[str, Any]] = []
    for i, t in enumerate(passages_raw):
        if isinstance(t, dict) and "text" in t:
            passages.append({"id": t.get("id", i), "text": t["text"], "meta": t.get("meta", {})})
        else:
            passages.append({"id": i, "text": str(t), "meta": {}})

    st_model = SentenceTransformer(emb_name)

    bm25 = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    if REQUIRE_BM25 and bm25 is None:
        raise AssertionError("Group 101 requires BM25. Provide bm25.pkl in artifacts/rag_index/")

    return index, passages, st_model, bm25, rr_name

@st.cache_resource(show_spinner=False)
def load_reranker(model_name: str):
    # Robust CrossEncoder loader: ensures pad token and aligned vocab/embeddings
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, max_length=512)  # downloads once, caches on disk
    tok = ce.tokenizer

    # Guarantee a pad token (common cause of IndexError)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    try:
        # If tokenizer length != embedding rows, align them
        vocab_n = len(tok)
        emb_n = ce.model.get_input_embeddings().weight.shape[0]
        if vocab_n != emb_n:
            ce.model.resize_token_embeddings(vocab_n)
        ce.model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    return ce


# ===== Utils =====
FIN_KWS = {
    "revenue","sales","income","profit","earnings","ebit","ebitda","operating","net",
    "expense","cost","cogs","cash","flow","assets","liabilities","equity","depreciation",
    "amortization","interest","tax","capex","opex","guidance","margin","balance","sheet",
    "income statement","p&l","statement","dividend","share","buyback","quarter","year"
}

def preprocess(q: str) -> str:
    return " ".join(q.strip().split())

def is_fin(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in FIN_KWS)

def fuse_scores(dense: List[Tuple[int,float]], sparse: List[Tuple[int,float]], w_dense=0.6, w_sparse=0.4, top_k=120):
    from collections import defaultdict
    scores = defaultdict(float)
    if dense:
        md = max((s for _, s in dense), default=1.0) or 1.0
        for i, s in dense: scores[i] += w_dense * (s / md)
    if sparse:
        ms = max((s for _, s in sparse), default=1.0) or 1.0
        for i, s in sparse: scores[i] += w_sparse * (s / ms)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ===== Retrieval & Generation =====
def stage1_retrieve(query: str, k_dense: int, k_sparse: int):
    import numpy as np, faiss
    index, passages, st_model, bm25, _ = load_rag_components()

    # Dense top-k
    qv = st_model.encode([query]).astype("float32")
    D, I = index.search(qv, k_dense)
    dense = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if 0 <= i < len(passages)]

    # Sparse top-k (required)
    if bm25 is None:
        raise AssertionError("BM25 missing; Group 101 requires Dense+Sparse at Stage-1.")
    scores = bm25.get_scores(query.split())
    top_idx = np.argsort(scores)[::-1][:k_sparse]
    sparse = [(int(i), float(scores[i])) for i in top_idx]

    fused = fuse_scores(dense, sparse, w_dense=0.6, w_sparse=0.4, top_k=max(k_dense, k_sparse))
    return fused, passages

def stage2_rerank(query: str, cand_ids: List[int], passages: List[Dict[str,Any]], keep_top=5):
    # Try CrossEncoder; if it fails, fall back to Stage-1 order (still Multi-Stage, with warning)
    try:
        _, _, _, _, rr_name = load_rag_components()
        reranker = load_reranker(rr_name)
        pairs = [(query, passages[i]["text"]) for i in cand_ids]
        # limit batch size to avoid OOM
        scores = reranker.predict(pairs, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
        order = list(sorted(range(len(scores)), key=lambda j: float(scores[j]), reverse=True))[:keep_top]
        return [(cand_ids[j], float(scores[j])) for j in order]
    except Exception as e:
        st.warning(f"Cross-encoder rerank failed ({e}); using Stage-1 fused order as fallback.")
        return [(i, 0.0) for i in cand_ids[:keep_top]]


def build_prompt(query: str, ctx_texts: List[str], max_ctx_chars=6000):
    buf = ""
    for t in ctx_texts:
        if len(buf) + len(t) + 2 > max_ctx_chars: break
        buf += t + "\n\n"
    return f"Context:\n{buf}\nQuestion: {query}\nAnswer:"

def generate_with_ft(prompt: str, max_new_tokens=160) -> str:
    pipe, kind = load_ft_pipeline()
    return pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]

def rag_pipeline(query: str, k_dense: int, k_sparse: int, keep_ctx: int):
    t0 = time.time()
    fused, passages = stage1_retrieve(query, k_dense, k_sparse)
    if not fused:
        return {"answer":"No relevant context retrieved.", "confidence":0.2, "method":"RAG (Multi-Stage)", "time_s":round(time.time()-t0,3), "contexts":[]}
    cand_ids = [i for i,_ in fused[:max(k_dense,k_sparse)]]
    reranked = stage2_rerank(query, cand_ids, passages, keep_top=keep_ctx)
    ctx = [passages[i]["text"] for i,_ in reranked]
    s = [sc for _, sc in reranked]
    import math
    avg3 = sum(s[:3])/max(1,min(3,len(s)))
    conf = 1/(1+math.exp(-avg3))
    ans = generate_with_ft(build_prompt(query, ctx))
    return {
        "answer": ans,
        "confidence": round(float(conf),3),
        "method": "RAG (Multi-Stage: Stage-1 Dense+Sparse → Stage-2 Cross-Encoder)",
        "time_s": round(time.time()-t0,3),
        "contexts": ctx
    }

def ft_pipeline(query: str):
    t0 = time.time()
    ans = generate_with_ft(query)
    return {"answer": ans, "confidence": None, "method": "FT (Supervised Instruction FT)", "time_s": round(time.time()-t0,3), "contexts": []}

# ===== Guardrails =====
def input_guard(q: str) -> Tuple[bool,str]:
    if not q.strip(): return False, "Empty query."
    if len(q) > 2000: return False, "Query too long."
    if not is_fin(q): return False, "Data not in scope. Ask a question grounded in financial statements."
    return True, ""

def output_guard(res: Dict[str,Any]) -> Dict[str,Any]:
    if res["method"].startswith("RAG") and (res.get("confidence",0)<0.45):
        res["answer"] = "[Low grounding confidence] " + res["answer"]
    return res

# ===== UI =====
st.set_page_config(page_title="Finance Q&A — RAG vs FT", layout="centered")
st.title("Finance Q&A — RAG vs FT")

with st.sidebar:
    st.subheader("Artifacts")
    if st.button("Initialize / Verify Artifacts"):
        prepare_artifacts(); st.success("Artifacts ready.")
    st.caption(f"Group {GROUP_ID}: {GROUP_TECH} — Stage-1 (Dense+Sparse) → Stage-2 (Cross-Encoder)")

with st.sidebar.expander("Retrieval settings"):
    k_dense  = st.slider("Stage-1 Dense top-k", 20, 200, 60, 10)
    k_sparse = st.slider("Stage-1 Sparse top-k", 20, 200, 60, 10)
    keep_ctx = st.slider("Stage-2 keep contexts", 3, 10, 5, 1)

engine = st.radio("Select method", ["RAG (Multi-Stage)", "FT (Fine-Tuned)"], horizontal=True)
q = st.text_input("Enter your question", placeholder="e.g., What was revenue in 2024?")
run = st.button("Run")

if run:
    prepare_artifacts()
    ok, msg = input_guard(preprocess(q))
    if not ok:
        st.warning(msg)
    else:
        if engine.startswith("RAG"):
            res = rag_pipeline(q, k_dense, k_sparse, keep_ctx)
        else:
            res = ft_pipeline(q)
        res = output_guard(res)
        st.subheader("Answer"); st.write(res["answer"])
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Method", res["method"])
        with c2: st.metric("Confidence", f"{res['confidence']:.2f}" if res["confidence"] is not None else "—")
        with c3: st.metric("Latency (s)", f"{res['time_s']:.3f}")
        if res.get("contexts"):
            with st.expander("Retrieved context"):
                for i, t in enumerate(res["contexts"], 1):
                    st.markdown(f"**{i}.** {t[:1500]}{'...' if len(t)>1500 else ''}")
