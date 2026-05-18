import argparse
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from utils.elsevier import (
	extract_structured_sections,
	fetch_elsevier_xml,
)


@dataclass
class TextChunk:
	chunk_id: str
	text: str
	metadata: Dict[str, Any]


class TextChunker:
	"""
	Section-aware chunker with overlap to preserve context continuity.
	"""

	def __init__(self, chunk_size: int = 1200, overlap: int = 200):
		if overlap >= chunk_size:
			raise ValueError("overlap must be smaller than chunk_size")
		self.chunk_size = chunk_size
		self.overlap = overlap

	@staticmethod
	def _split_sentences(text: str) -> List[str]:
		parts = re.split(r"(?<=[.!?])\s+", text.strip())
		return [p.strip() for p in parts if p.strip()]

	def _split_into_units(self, text: str) -> List[str]:
		paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
		if len(paragraphs) > 1:
			return paragraphs
		return self._split_sentences(text)

	def _chunk_text(self, text: str) -> List[str]:
		if not text:
			return []

		units = self._split_into_units(text)
		if not units:
			return []

		chunks: List[str] = []
		current = ""

		for unit in units:
			candidate = f"{current}\n\n{unit}".strip() if current else unit
			if len(candidate) <= self.chunk_size:
				current = candidate
				continue

			if current:
				chunks.append(current)

			if len(unit) <= self.chunk_size:
				current = unit
				continue

			# Hard fallback for very long units.
			start = 0
			while start < len(unit):
				end = min(start + self.chunk_size, len(unit))
				subchunk = unit[start:end].strip()
				if subchunk:
					chunks.append(subchunk)
				if end >= len(unit):
					break
				start = max(0, end - self.overlap)
			current = ""

		if current:
			chunks.append(current)

		if self.overlap <= 0 or len(chunks) <= 1:
			return chunks

		overlapped = [chunks[0]]
		for idx in range(1, len(chunks)):
			prev_tail = chunks[idx - 1][-self.overlap :]
			merged = f"{prev_tail}\n\n{chunks[idx]}".strip()
			overlapped.append(merged)

		return overlapped

	def chunk_sections(self, doi: str, sections: List[Dict[str, str]]) -> List[TextChunk]:
		chunks: List[TextChunk] = []
		global_idx = 0

		for section_idx, section in enumerate(sections):
			section_title = section.get("section_title", "Section")
			section_type = section.get("section_type", "unknown")
			section_text = section.get("text", "").strip()
			if not section_text:
				continue

			section_chunks = self._chunk_text(section_text)
			for local_idx, chunk_text in enumerate(section_chunks):
				chunk_id = f"{doi}::{section_idx}::{local_idx}::{uuid.uuid4().hex[:8]}"
				chunks.append(
					TextChunk(
						chunk_id=chunk_id,
						text=chunk_text,
						metadata={
							"doi": doi,
							"section_title": section_title,
							"section_type": section_type,
							"chunk_index": global_idx,
							"section_chunk_index": local_idx,
						},
					)
				)
				global_idx += 1

		return chunks


class EmbeddingInterface:
	"""
	Abstraction over sentence-transformers embeddings.
	"""

	def __init__(
		self,
		model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
		device: Optional[str] = None,
	):
		self.model_name = model_name
		self.device = device
		self._model = None

	def _ensure_model(self):
		if self._model is not None:
			return
		try:
			from sentence_transformers import SentenceTransformer
		except ImportError as exc:
			raise ImportError(
				"sentence-transformers is required for embeddings. "
				"Install with: pip install sentence-transformers"
			) from exc

		kwargs = {}
		if self.device:
			kwargs["device"] = self.device
		self._model = SentenceTransformer(self.model_name, **kwargs)

	def embed_texts(self, texts: List[str]) -> np.ndarray:
		if not texts:
			return np.array([])
		self._ensure_model()
		embeddings = self._model.encode(
			texts,
			normalize_embeddings=True,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		return embeddings

	def embed_query(self, query: str) -> np.ndarray:
		embeddings = self.embed_texts([query])
		if embeddings.size == 0:
			return np.array([])
		return embeddings[0]


class ChromaVectorStore:
	"""
	Persistent vector store based on Chroma DB.
	"""

	def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "papers"):
		self.persist_directory = persist_directory
		self.collection_name = collection_name
		self._collection = None

	def _ensure_collection(self):
		if self._collection is not None:
			return
		try:
			import chromadb
		except ImportError as exc:
			raise ImportError(
				"chromadb is required for vector storage. "
				"Install with: pip install chromadb"
			) from exc

		os.makedirs(self.persist_directory, exist_ok=True)
		client = chromadb.PersistentClient(path=self.persist_directory)
		self._collection = client.get_or_create_collection(
			name=self.collection_name,
			metadata={"hnsw:space": "cosine"},
		)

	def add(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
		if not chunks:
			return
		if embeddings is None or len(embeddings) != len(chunks):
			raise ValueError("embeddings length must match chunks length")

		self._ensure_collection()
		ids = [chunk.chunk_id for chunk in chunks]
		documents = [chunk.text for chunk in chunks]
		metadatas = [chunk.metadata for chunk in chunks]

		self._collection.add(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings.tolist(),
		)

	def similarity_search(
		self,
		query_embedding: np.ndarray,
		top_k: int = 5,
		where: Optional[Dict[str, Any]] = None,
	) -> List[Dict[str, Any]]:
		if query_embedding is None or query_embedding.size == 0:
			return []

		self._ensure_collection()
		results = self._collection.query(
			query_embeddings=[query_embedding.tolist()],
			n_results=top_k,
			where=where,
			include=["documents", "metadatas", "distances"],
		)

		docs = results.get("documents", [[]])[0]
		metas = results.get("metadatas", [[]])[0]
		distances = results.get("distances", [[]])[0]
		ids = results.get("ids", [[]])[0]

		output = []
		for idx, doc in enumerate(docs):
			distance = float(distances[idx]) if idx < len(distances) else None
			score = None if distance is None else 1.0 - distance
			output.append(
				{
					"id": ids[idx] if idx < len(ids) else None,
					"text": doc,
					"metadata": metas[idx] if idx < len(metas) else {},
					"distance": distance,
					"score": score,
				}
			)
		return output

	def count(self) -> int:
		self._ensure_collection()
		return self._collection.count()


class RetrievalComponent:
	def __init__(self, embedder: EmbeddingInterface, vector_store: ChromaVectorStore):
		self.embedder = embedder
		self.vector_store = vector_store

	def retrieve(
		self,
		query: str,
		top_k: int = 5,
		where: Optional[Dict[str, Any]] = None,
	) -> List[Dict[str, Any]]:
		query_embedding = self.embedder.embed_query(query)
		return self.vector_store.similarity_search(query_embedding, top_k=top_k, where=where)


class LLMWrapper:
	"""
	Lightweight wrapper for Hugging Face causal language models.
	"""

	def __init__(
		self,
		model_name: str = "Qwen/Qwen2.5-7B-Instruct",
		max_new_tokens: int = 512,
		temperature: float = 0.1,
		top_p: float = 0.9,
		device_map: str = "auto",
	):
		self.model_name = model_name
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.device_map = device_map
		self._tokenizer = None
		self._model = None

	def _ensure_model(self):
		if self._tokenizer is not None and self._model is not None:
			return
		try:
			import torch
			from transformers import AutoModelForCausalLM, AutoTokenizer
		except ImportError as exc:
			raise ImportError(
				"transformers and torch are required for generation. "
				"Install with: pip install transformers torch"
			) from exc

		self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self._model = AutoModelForCausalLM.from_pretrained(
			self.model_name,
			device_map=self.device_map,
			torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
		)

	def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
		context_lines = []
		for i, chunk in enumerate(retrieved_chunks, start=1):
			meta = chunk.get("metadata", {})
			doi = meta.get("doi", "unknown")
			section = meta.get("section_title", "unknown")
			text = chunk.get("text", "")
			context_lines.append(
				f"[{i}] DOI: {doi} | Section: {section}\n{text}"
			)

		context_text = "\n\n".join(context_lines) if context_lines else "No context retrieved."
		return (
			"You are a scientific assistant. Answer the question using only the provided context. "
			"If the answer is not contained in the context, say you do not have enough evidence. "
			"Cite sources using [n] markers from the context list.\n\n"
			f"Context:\n{context_text}\n\n"
			f"Question: {question}\n\n"
			"Answer:"
		)

	def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
		self._ensure_model()
		prompt = self.build_prompt(question, retrieved_chunks)

		if hasattr(self._tokenizer, "apply_chat_template"):
			messages = [
				{"role": "system", "content": "Answer with grounded citations from the context."},
				{"role": "user", "content": prompt},
			]
			input_ids = self._tokenizer.apply_chat_template(
				messages,
				add_generation_prompt=True,
				return_tensors="pt",
			).to(self._model.device)
		else:
			inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
			input_ids = inputs["input_ids"]

		output_ids = self._model.generate(
			input_ids,
			max_new_tokens=self.max_new_tokens,
			do_sample=self.temperature > 0,
			temperature=self.temperature,
			top_p=self.top_p,
			eos_token_id=self._tokenizer.eos_token_id,
			pad_token_id=self._tokenizer.eos_token_id,
		)

		generated_ids = output_ids[:, input_ids.shape[-1] :]
		return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


class RAGPipelineCoordinator:
	"""
	Coordinates ingestion and query pipelines end-to-end.
	"""

	def __init__(
		self,
		chunker: TextChunker,
		embedder: EmbeddingInterface,
		vector_store: ChromaVectorStore,
		retriever: RetrievalComponent,
		llm: Optional[LLMWrapper] = None,
	):
		self.chunker = chunker
		self.embedder = embedder
		self.vector_store = vector_store
		self.retriever = retriever
		self.llm = llm

	def ingest_xml(self, doi: str, xml_input: Any) -> Dict[str, Any]:
		sections = extract_structured_sections(xml_input)
		if not sections:
			return {
				"doi": doi,
				"status": "failed",
				"reason": "no_text_extracted",
				"chunks_added": 0,
			}

		chunks = self.chunker.chunk_sections(doi, sections)
		if not chunks:
			return {
				"doi": doi,
				"status": "failed",
				"reason": "no_chunks_created",
				"chunks_added": 0,
			}

		embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
		self.vector_store.add(chunks, embeddings)
		return {
			"doi": doi,
			"status": "success",
			"sections_extracted": len(sections),
			"chunks_added": len(chunks),
		}

	def ingest_doi(
		self,
		doi: str,
		api_key: Optional[str] = None,
		timeout: int = 30,
	) -> Dict[str, Any]:
		xml_soup = fetch_elsevier_xml(doi, api_key=api_key, timeout=timeout)
		if xml_soup is None:
			return {
				"doi": doi,
				"status": "failed",
				"reason": "fetch_failed",
				"chunks_added": 0,
			}
		return self.ingest_xml(doi=doi, xml_input=xml_soup)

	def ingest_dois(
		self,
		dois: List[str],
		api_key: Optional[str] = None,
		timeout: int = 30,
	) -> List[Dict[str, Any]]:
		results = []
		for doi in dois:
			try:
				result = self.ingest_doi(doi=doi, api_key=api_key, timeout=timeout)
			except Exception as exc:
				result = {
					"doi": doi,
					"status": "failed",
					"reason": str(exc),
					"chunks_added": 0,
				}
			results.append(result)
		return results

	def query(
		self,
		question: str,
		top_k: int = 5,
		where: Optional[Dict[str, Any]] = None,
		generate_answer: bool = True,
	) -> Dict[str, Any]:
		retrieved = self.retriever.retrieve(question, top_k=top_k, where=where)
		answer = None
		if generate_answer and self.llm is not None:
			answer = self.llm.generate(question, retrieved)

		return {
			"question": question,
			"answer": answer,
			"retrieved_chunks": retrieved,
			"num_retrieved": len(retrieved),
		}


def build_default_pipeline(
	chroma_dir: str = "chroma_db",
	chroma_collection: str = "papers",
	embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
	llm_model: Optional[str] = None,
) -> RAGPipelineCoordinator:
	chunker = TextChunker(chunk_size=1200, overlap=200)
	embedder = EmbeddingInterface(model_name=embedding_model)
	vector_store = ChromaVectorStore(
		persist_directory=chroma_dir,
		collection_name=chroma_collection,
	)
	retriever = RetrievalComponent(embedder=embedder, vector_store=vector_store)

	llm = None
	if llm_model:
		llm = LLMWrapper(model_name=llm_model)

	return RAGPipelineCoordinator(
		chunker=chunker,
		embedder=embedder,
		vector_store=vector_store,
		retriever=retriever,
		llm=llm,
	)


def _parse_where_filters(filter_args: List[str]) -> Dict[str, Any]:
	where: Dict[str, Any] = {}
	for raw in filter_args:
		if "=" not in raw:
			continue
		key, value = raw.split("=", 1)
		key = key.strip()
		value = value.strip()
		if key:
			where[key] = value
	return where


def main():
	parser = argparse.ArgumentParser(description="XML Paper RAG Pipeline")
	subparsers = parser.add_subparsers(dest="command", required=True)

	ingest_parser = subparsers.add_parser("ingest", help="Ingest one or many DOIs")
	ingest_parser.add_argument("--doi", action="append", required=True, help="DOI to ingest. Use multiple times for multiple DOIs.")
	ingest_parser.add_argument("--api-key", default=None, help="Elsevier API key")
	ingest_parser.add_argument("--chroma-dir", default="chroma_db", help="Chroma persistence directory")
	ingest_parser.add_argument("--collection", default="papers", help="Chroma collection name")
	ingest_parser.add_argument(
		"--embedding-model",
		default="sentence-transformers/all-MiniLM-L6-v2",
		help="Sentence-transformers model",
	)

	query_parser = subparsers.add_parser("query", help="Query indexed papers")
	query_parser.add_argument("--question", required=True, help="Question for retrieval/generation")
	query_parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks to retrieve")
	query_parser.add_argument("--filter", action="append", default=[], help="Metadata filters in key=value format")
	query_parser.add_argument("--chroma-dir", default="chroma_db", help="Chroma persistence directory")
	query_parser.add_argument("--collection", default="papers", help="Chroma collection name")
	query_parser.add_argument(
		"--embedding-model",
		default="sentence-transformers/all-MiniLM-L6-v2",
		help="Sentence-transformers model",
	)
	query_parser.add_argument("--llm-model", default=None, help="Optional HF model for answer generation")

	args = parser.parse_args()

	pipeline = build_default_pipeline(
		chroma_dir=args.chroma_dir,
		chroma_collection=args.collection,
		embedding_model=args.embedding_model,
		llm_model=getattr(args, "llm_model", None),
	)

	if args.command == "ingest":
		results = pipeline.ingest_dois(dois=args.doi, api_key=args.api_key)
		for row in results:
			print(row)
		print({"total_vectors": pipeline.vector_store.count()})
		return

	if args.command == "query":
		where = _parse_where_filters(args.filter)
		output = pipeline.query(
			question=args.question,
			top_k=args.top_k,
			where=where if where else None,
			generate_answer=bool(getattr(args, "llm_model", None)),
		)
		print({"answer": output["answer"], "num_retrieved": output["num_retrieved"]})
		for idx, chunk in enumerate(output["retrieved_chunks"], start=1):
			meta = chunk.get("metadata", {})
			print(
				{
					"rank": idx,
					"score": chunk.get("score"),
					"doi": meta.get("doi"),
					"section_title": meta.get("section_title"),
					"text": chunk.get("text"),
				}
			)


if __name__ == "__main__":
	main()