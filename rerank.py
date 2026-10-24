#!/usr/bin/env python3
"""
rerank.py - Cross-encoder reranking on top of hybrid (or dense) retrieval.

THE PROBLEM THIS SOLVES
-----------------------
Diagnosis from the eval: the 6 stuck questions are NOT missing from retrieval -
their correct paper sits in the wide candidate pool (ranks 6-78 of 200). The
system retrieves the right papers but RANKS them too low to appear in the top-5.
That is an ordering problem, and a reranker is the standard fix.

HOW IT WORKS
------------
A bi-encoder (the embedder) scores query and chunk SEPARATELY, then compares
vectors - fast but approximate. A cross-encoder reads (query, chunk) TOGETHER
through every transformer layer, directly judging "does this chunk answer this
question". Much more accurate at ordering, too slow to run over the 290,247
indexed records - so we only run it over the candidate pool the retriever
already narrowed to.

PIPELINE
--------
  1. Base retriever (hybrid or dense) returns a WIDE pool (candidate_k, e.g. 100).
  2. Cross-encoder scores each (question, chunk_text) pair.
  3. Re-sort by that score; return the top_k.

DESIGN
------
RerankRetriever wraps any base retriever that exposes
.query(question, top_k, generate_answer) -> {"retrieved_chunks": [...], ...}
and itself exposes the SAME interface, so eval/evaluate.py drives it unchanged.
It asks the base retriever for `candidate_k` chunks, reranks, returns `top_k`.

MODEL
-----
Default: BAAI/bge-reranker-v2-m3 - the 2026 open-weight default (Apache-2.0,
strong quality). Loaded via sentence-transformers' CrossEncoder (a clean,
one-line load - no adapter library needed). Swappable via --reranker-model.
"""

from typing import Any, Dict, List, Optional
import torch

class RerankRetriever:
    def __init__(
        self,
        base_retriever,                 # anything with .query(question, top_k, generate_answer)
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        candidate_k: int = 100,         # how many to pull from the base retriever before reranking
        device: Optional[str] = None,   # None -> CUDA if available, else CPU; MPS deliberately skipped (CrossEncoder deadlocked)
        max_length: int = 512,  # cross-encoder input cap (query+chunk truncated to this)
        batch_size: int = 32,
        within_paper: bool = False,  # freeze paper order, rerank chunks only within each paper
    ):
        self.base = base_retriever
        self.reranker_model = reranker_model
        self.candidate_k = candidate_k
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.within_paper = within_paper
        self._ce = None

        # expose the base's vector_store if it has one, so evaluate.py's count line works
        self.vector_store = getattr(base_retriever, "vector_store", None)
        self.llm = getattr(base_retriever, "llm", None)

    def _ensure_model(self):
        if self._ce is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "pip install sentence-transformers"
            ) from exc
        kwargs = {"max_length": self.max_length}

        # an explicit device wins; auto skips MPS: deadlocks on this model on Apple Silicon
        _dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ce = CrossEncoder(self.reranker_model, device=_dev, **kwargs)

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,   # signature parity only; IGNORED (not passed to base)
        generate_answer: bool = False,
    ) -> Dict[str, Any]:
        # 1) wide candidate pool from the base retriever (no generation yet)
        base_out = self.base.query(
            question,
            top_k=self.candidate_k,
            generate_answer=False,
        )
        candidates = base_out.get("retrieved_chunks", [])

        if not candidates:
            return {"question": question, "answer": None,
                    "retrieved_chunks": [], "num_retrieved": 0}

        # 2) cross-encoder scores each (question, chunk_text) pair
        self._ensure_model()
        pairs = [[question, (c.get("text") or "")] for c in candidates]
        scores = self._ce.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 3) attach scores, re-sort, take top_k
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
            c["score"] = float(s)   # overwrite primary score so downstream sees rerank order
        if self.within_paper:
            # Freeze paper order from the BASE ranking; rerank chunks only WITHIN each paper.
            # (Protects Hit@k/MRR exactly; only which chunk of each paper surfaces can change.)
            def _doi_of(c):
                return (c.get("id") or "").split("::", 1)[0]

            paper_order = []
            by_paper = {}
            for c in candidates:  # candidates are already in base-retriever order
                d = _doi_of(c)
                if d not in by_paper:
                    by_paper[d] = []
                    paper_order.append(d)
                by_paper[d].append(c)
            # Round-robin: one best chunk per paper (base order) before any 2nd chunk.
            # Makes top_k chunks == top_k papers, so Hit@k mirrors base paper ranking exactly.
            ranked_groups = [
                sorted(by_paper[d], key=lambda c: c["rerank_score"], reverse=True)
                for d in paper_order
            ]
            reranked = []
            depth = 0
            while len(reranked) < top_k:
                added_any = False
                for group in ranked_groups:
                    if depth < len(group):
                        reranked.append(group[depth])
                        added_any = True
                        if len(reranked) >= top_k:
                            break
                if not added_any:
                    break
                depth += 1
        else:
            reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

        # 4) optional generation over the reranked top_k
        answer = None
        if generate_answer and self.llm is not None:
            answer = self.llm.generate(question, reranked)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": reranked,
            "num_retrieved": len(reranked),
        }


def build_rerank_retriever(
    # base-retriever knobs (mirror build_hybrid_retriever / build_default_pipeline)
    chroma_dir: str = "chroma_db",
    chroma_collection: str = "papers",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    base: str = "hybrid",               # "hybrid" or "dense"
    bm25_dir: Optional[str] = None,
    fusion: str = "weighted",
    hybrid_candidate_k: int = 50,       # base hybrid's own internal pool (per-retriever)
    rrf_k: int = 60,
    alpha: float = 0.4,
    llm_model: Optional[str] = None,
    # reranker knobs
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    candidate_k: int = 100,
    device: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 32,
    within_paper: bool = False,
) -> RerankRetriever:
    """Build a reranking retriever over a hybrid (default) or dense base."""
    if base == "hybrid":
        from hybrid import build_hybrid_retriever
        base_ret = build_hybrid_retriever(
            chroma_dir=chroma_dir,
            chroma_collection=chroma_collection,
            embedding_model=embedding_model,
            bm25_dir=bm25_dir,
            llm_model=llm_model,
            rrf_k=rrf_k,
            candidate_k=max(hybrid_candidate_k, candidate_k),  # hybrid must return >= candidate_k
            fusion=fusion,
            alpha=alpha,
        )
    else:
        from rag import build_default_pipeline
        base_ret = build_default_pipeline(
            chroma_dir=chroma_dir,
            chroma_collection=chroma_collection,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )

    return RerankRetriever(
        base_retriever=base_ret,
        reranker_model=reranker_model,
        candidate_k=candidate_k,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        within_paper=within_paper,
    )