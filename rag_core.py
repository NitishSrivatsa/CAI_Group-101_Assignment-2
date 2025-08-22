from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils_finance import clean_text, tokenize

class RAGConfig:
    def __init__(self,
                 data_dir: str = "data",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_candidates: int = 12,
                 top_k_dense: int = 8,
                 top_k_sparse: int = 8,
                 fuse_alpha: float = 0.5):
        self.DATA_DIR = Path(data_dir)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.max_candidates = max_candidates
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.fuse_alpha = fuse_alpha

class CorpusIndex:
    # Multi-Stage Retrieval: Stage 1 broad recall; Stage 2 cross-encoder re-ranking
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.docs: List[Dict[str, Any]] = []
        self.bm25_S = None
        self.bm25_L = None
        self.emb_model = None
        self.dense_matrix_S = None
        self.dense_matrix_L = None
        self.nn_S = None
        self.nn_L = None
        self.cross_encoder = None
        self.S = None
        self.L = None
        self.tokens_S = None
        self.tokens_L = None

    def load_corpus(self) -> None:
        corpus_dir = self.cfg.DATA_DIR / "corpus"
        corpus_dir.mkdir(exist_ok=True)
        files = list(corpus_dir.glob("*.txt"))
        if not files:
            sample = (corpus_dir/"sample.txt")
            sample.write_text("Revenue 2023 was $4.13 billion. Revenue 2024 was $4.02 billion.\nOperating income improved year over year.")
            files = [sample]
        self.docs = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            self.docs.append({
                "id": f.stem,
                "text": text,
                "clean": clean_text(text)
            })

    def chunk_docs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def chunk_words(words, size):
            return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
        rows_s, rows_l = [], []
        for d in self.docs:
            words = d["clean"].split()
            for i, c in enumerate(chunk_words(words, 100)):
                rows_s.append({"doc_id": d["id"], "chunk_id": f"{d['id']}_s_{i}", "text": c})
            for i, c in enumerate(chunk_words(words, 400)):
                rows_l.append({"doc_id": d["id"], "chunk_id": f"{d['id']}_l_{i}", "text": c})
        self.S = pd.DataFrame(rows_s) if rows_s else pd.DataFrame(columns=["doc_id","chunk_id","text"])
        self.L = pd.DataFrame(rows_l) if rows_l else pd.DataFrame(columns=["doc_id","chunk_id","text"])
        return self.S, self.L

    def build_sparse(self):
        if self.S is None or self.L is None:
            raise RuntimeError("Call chunk_docs() first.")
        self.tokens_S = [tokenize(t) for t in self.S.text.tolist()]
        self.tokens_L = [tokenize(t) for t in self.L.text.tolist()]
        self.bm25_S = BM25Okapi(self.tokens_S) if len(self.tokens_S) else None
        self.bm25_L = BM25Okapi(self.tokens_L) if len(self.tokens_L) else None

    def build_dense(self):
        if self.S is None or self.L is None:
            raise RuntimeError("Call chunk_docs() first.")
        self.emb_model = SentenceTransformer(self.cfg.embed_model)
        if len(self.S):
            embs_S = self.emb_model.encode(self.S.text.tolist(), show_progress_bar=False, normalize_embeddings=True)
            self.dense_matrix_S = np.array(embs_S, dtype=np.float32)
            self.nn_S = NearestNeighbors(n_neighbors=min(self.cfg.top_k_dense, len(self.S)), metric="cosine")
            self.nn_S.fit(self.dense_matrix_S)
        if len(self.L):
            embs_L = self.emb_model.encode(self.L.text.tolist(), show_progress_bar=False, normalize_embeddings=True)
            self.dense_matrix_L = np.array(embs_L, dtype=np.float32)
            self.nn_L = NearestNeighbors(n_neighbors=min(self.cfg.top_k_dense, len(self.L)), metric="cosine")
            self.nn_L.fit(self.dense_matrix_L)

    def ensure_reranker(self):
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(self.cfg.rerank_model)

    def _bm25_top(self, query: str, which: str):
        if which == "S" and self.bm25_S is not None:
            q_tokens = tokenize(query)
            scores = self.bm25_S.get_scores(q_tokens)
            idx = np.argsort(scores)[::-1][: self.cfg.top_k_sparse]
            return [(int(i), float(scores[i])) for i in idx]
        if which == "L" and self.bm25_L is not None:
            q_tokens = tokenize(query)
            scores = self.bm25_L.get_scores(q_tokens)
            idx = np.argsort(scores)[::-1][: self.cfg.top_k_sparse]
            return [(int(i), float(scores[i])) for i in idx]
        return []

    def _dense_top(self, query: str, which: str):
        q = self.emb_model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        if which == "S" and self.nn_S is not None:
            dists, idxs = self.nn_S.kneighbors(q, n_neighbors=min(self.cfg.top_k_dense, len(self.S)))
            sims = 1 - dists[0]
            return [(int(i), float(s)) for i, s in zip(idxs[0], sims)]
        if which == "L" and self.nn_L is not None:
            dists, idxs = self.nn_L.kneighbors(q, n_neighbors=min(self.cfg.top_k_dense, len(self.L)))
            sims = 1 - dists[0]
            return [(int(i), float(s)) for i, s in zip(idxs[0], sims)]
        return []

    def stage1_candidates(self, query: str):
        S_bm25 = self._bm25_top(query, "S")
        L_bm25 = self._bm25_top(query, "L")
        S_dense = self._dense_top(query, "S")
        L_dense = self._dense_top(query, "L")

        def fuse(mod_bm25, mod_dense):
            rank = {}
            for r,(i,_) in enumerate(mod_bm25): rank[i] = rank.get(i,0)+ 1.0/(1+r)
            for r,(i,_) in enumerate(mod_dense): rank[i] = rank.get(i,0)+ 1.0/(1+r)
            return sorted(rank.items(), key=lambda x:x[1], reverse=True)

        fused_S = fuse(S_bm25, S_dense)[: self.cfg.max_candidates//2] if len(self.S) else []
        fused_L = fuse(L_bm25, L_dense)[: self.cfg.max_candidates//2] if len(self.L) else []

        cand = []
        for i,_ in fused_S:
            cand.append(("S", int(i), self.S.text.iloc[i]))
        for i,_ in fused_L:
            cand.append(("L", int(i), self.L.text.iloc[i]))
        return cand

    def stage2_rerank(self, query: str, candidates):
        self.ensure_reranker()
        pairs = [[query, c[2]] for c in candidates]
        if not pairs:
            return []
        scores = self.cross_encoder.predict(pairs)
        order = np.argsort(scores)[::-1][:8]
        return [(candidates[i][2], float(scores[i])) for i in order]
