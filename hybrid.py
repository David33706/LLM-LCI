#!/usr/bin/env python3
"""
hybrid.py - Hybrid (dense + BM25) retrieval with rank fusion (weighted or RRF).

WHAT THIS IS
------------
A capability module (like rag.py), NOT an eval script. It wraps:
  - the existing DENSE retriever (Chroma + embedder, via build_default_pipeline)
  - a SPARSE BM25 index (built by ingest/build_bm25_index.py)
and fuses their two ranked lists, by weighted 1/rank (the validated config,
alpha=0.4) or Reciprocal Rank Fusion (RRF).

WHY
---
Dense retrieval (any embedder we tried) misses on distinctive exact terms -
material names, "SimaPro v9.6", "HSC Chemistry" - because semantically-near
papers blur together. BM25 matches those exact terms. Fusing the two covers
each method's blind spot. RRF fuses on RANK (not score), sidestepping the
score-incompatibility between cosine similarity and BM25 scores.

DESIGN
------
HybridRetriever exposes a .query(question, top_k, generate_answer) method that
returns the SAME dict shape as RAGPipelineCoordinator.query:
    {"question", "answer", "retrieved_chunks": [...], "num_retrieved"}
so eval/evaluate.py can drive it with zero changes to its scoring loop.
Each retrieved chunk has the same shape the dense side returns:
    {"id", "text", "metadata": {"doi", "section_title"}, "score"}

RRF
---
For a chunk appearing at rank r_dense in the dense list and r_bm25 in the BM25
list, its fused score is:
    sum over each list it appears in of  1 / (rrf_k + rank)
with rrf_k = 60 (standard default). Chunks ranked high by EITHER method rise.
A chunk missing from one list simply contributes nothing from that list.

WEIGHTED (the validated config)
--------
fusion="weighted" scores alpha * 1/rank_dense + (1 - alpha) * 1/rank_bm25.
build_hybrid_retriever defaults to it (alpha=0.4, candidate_k=100); the
HybridRetriever class itself still defaults to RRF / candidate_k=50.
Endpoints are pure for top_k <= len(active list): alpha=1 returns exactly the
dense ranking and alpha=0 exactly the BM25 ranking (the other list scores 0).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class HybridRetriever:
    def __init__(
        self,
        dense_pipeline,                 # a RAGPipelineCoordinator (from build_default_pipeline)
        bm25_dir: str,                  # dir produced by build_bm25_index.py
        rrf_k: int = 60,
        candidate_k: int = 50,          # how many to pull from EACH retriever before fusing
        fusion: str = "rrf",            # "rrf" or "weighted"
        alpha: float = 0.4,             # only used when fusion="weighted" (dense weight)
    ):
        self.dense = dense_pipeline
        _bm = Path(bm25_dir)
        if not _bm.is_absolute():
            _bm = Path(__file__).resolve().parent / _bm
        self.bm25_dir = _bm
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.fusion = fusion
        self.alpha = alpha

        self._bm25 = None
        self._tokenize = None
        self._ids: List[str] = []
        self._dois: List[Optional[str]] = []
        self._sections: List[Optional[str]] = []
        self._texts: List[str] = []

    @property
    def vector_store(self):
        # expose the dense side's store so eval can count chunks and warn if empty
        return self.dense.vector_store

    # -- lazy load the BM25 index + sidecars once --
    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("bm25s is required for hybrid retrieval. Run: pip install bm25s") from exc

        index_path = self.bm25_dir / "bm25s"
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. Build it first with: "
                f"python ingest/build_bm25_index.py (from the repo root)"
            )
        self._bm25 = bm25s.BM25.load(str(index_path), load_corpus=False)
        self._tokenize = bm25s.tokenize

        with open(self.bm25_dir / "chunks.json", encoding="utf-8") as f:
            side = json.load(f)
        self._ids = side["ids"]
        self._dois = side["dois"]
        self._sections = side["sections"]
        with open(self.bm25_dir / "texts.json", encoding="utf-8") as f:
            self._texts = json.load(f)

    # -- dense side: reuse the existing pipeline's retriever, return ranked chunk dicts --
    def _dense_ranked(self, question: str) -> List[Dict[str, Any]]:
        out = self.dense.query(question, top_k=self.candidate_k, generate_answer=False)
        return out.get("retrieved_chunks", [])

    # -- sparse side: BM25 over the same corpus, return ranked chunk dicts in the same shape --
    def _bm25_ranked(self, question: str) -> List[Dict[str, Any]]:
        self._ensure_bm25()
        q_tokens = self._tokenize(question, show_progress=False)
        # bm25s.retrieve returns (results, scores); results are corpus indices
        results, scores = self._bm25.retrieve(q_tokens, k=self.candidate_k, show_progress=False)
        ranked = []
        row = results[0]            # single query
        srow = scores[0]
        for j in range(len(row)):
            idx = int(row[j])
            ranked.append({
                "id": self._ids[idx] if idx < len(self._ids) else None,
                "text": self._texts[idx] if idx < len(self._texts) else "",
                "metadata": {
                    "doi": self._dois[idx] if idx < len(self._dois) else None,
                    "section_title": self._sections[idx] if idx < len(self._sections) else None,
                },
                "score": float(srow[j]),
            })
        return ranked

    # -- a stable key to tell whether a dense hit and a bm25 hit are the SAME chunk --
    @staticmethod
    def _key(chunk: Dict[str, Any]) -> str:
        # Prefer the chunk id; fall back to (doi, section, text-prefix) if id missing.
        cid = chunk.get("id")
        if cid:
            return f"id::{cid}"
        meta = chunk.get("metadata", {}) or {}
        return f"x::{meta.get('doi')}::{meta.get('section_title')}::{(chunk.get('text') or '')[:80]}"

    # -- fuse the two ranked lists --
    def _fuse(self, dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunk_by_key: Dict[str, Dict[str, Any]] = {}
        score_by_key: Dict[str, float] = {}

        def add(ranked_list, is_dense):
            for rank, ch in enumerate(ranked_list, start=1):
                k = self._key(ch)
                if k not in chunk_by_key:
                    chunk_by_key[k] = ch
                if self.fusion == "weighted":
                    # convex combination of normalized contributions;
                    # use 1/rank as a rank-based proxy so the two are comparable
                    w = self.alpha if is_dense else (1.0 - self.alpha)
                    score_by_key[k] = score_by_key.get(k, 0.0) + w * (1.0 / rank)
                else:  # rrf
                    score_by_key[k] = score_by_key.get(k, 0.0) + 1.0 / (self.rrf_k + rank)

        add(dense, True)
        add(sparse, False)

        fused = []
        for k, ch in chunk_by_key.items():
            ch = dict(ch)
            ch["score"] = score_by_key[k]   # overwrite with fused score
            fused.append(ch)
        fused.sort(key=lambda c: c["score"], reverse=True)
        return fused

    # -- the public method: SAME shape as RAGPipelineCoordinator.query --
    def query(
        self,
        question: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,   # accepted for signature parity; unused here
        generate_answer: bool = False,
    ) -> Dict[str, Any]:
        dense = self._dense_ranked(question)
        sparse = self._bm25_ranked(question)
        fused = self._fuse(dense, sparse)[:top_k]

        answer = None
        if generate_answer and getattr(self.dense, "llm", None) is not None:
            answer = self.dense.llm.generate(question, fused)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": fused,
            "num_retrieved": len(fused),
        }


def build_hybrid_retriever(
    chroma_dir: str = "chroma_db",
    chroma_collection: str = "papers",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    bm25_dir: Optional[str] = None,
    llm_model: Optional[str] = None,
    rrf_k: int = 60,
    candidate_k: int = 100,
    fusion: str = "weighted",
    alpha: float = 0.4,
) -> HybridRetriever:
    """Construct a HybridRetriever from the same inputs build_default_pipeline uses,
    plus the BM25 index dir. Defaults assume the MiniLM 'papers' collection and are
    the validated config (weighted fusion, alpha=0.4, candidate_k=100)."""
    from rag import build_default_pipeline
    dense = build_default_pipeline(
        chroma_dir=chroma_dir,
        chroma_collection=chroma_collection,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    if bm25_dir is None:
        bm25_dir = str(Path("bm25_index") / chroma_collection)
    return HybridRetriever(
        dense_pipeline=dense,
        bm25_dir=bm25_dir,
        rrf_k=rrf_k,
        candidate_k=candidate_k,
        fusion=fusion,
        alpha=alpha,
    )