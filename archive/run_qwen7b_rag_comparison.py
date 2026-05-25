import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import transformers
from dotenv import load_dotenv
from tqdm import tqdm

from rag import build_default_pipeline
from utils.elsevier import extract_structured_sections, fetch_elsevier_xml


DEFAULT_DATA_PATH = "data/uselca1.csv"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "papers"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEST_MAX_DOIS = 50
PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
]


def _run_system_command(command: str) -> None:
    """
    Execute shell-level helper commands such as proxy_on/proxy_off.
    Uses `bash -lc` so user shell commands/functions are available.
    """
    print(f"[proxy] running: {command}")
    proc = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"[proxy] warning: command failed ({command}) rc={proc.returncode} err={stderr}")
    elif proc.stdout:
        print(proc.stdout.strip())


def _apply_proxy_command(command: str) -> None:
    """
    Run proxy command in a shell and import that shell's environment into
    this Python process. This makes proxy_on/proxy_off effective for requests
    and huggingface downloads performed after the call.
    """
    print(f"[proxy] apply env via: {command}")
    proc = subprocess.run(
        ["bash", "-lc", f"{command} >/dev/null 2>&1; env -0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"[proxy] warning: failed to apply env from {command}; rc={proc.returncode} err={stderr}")
        return

    output = proc.stdout or b""
    shell_env: Dict[str, str] = {}
    for entry in output.split(b"\x00"):
        if not entry or b"=" not in entry:
            continue
        key_b, val_b = entry.split(b"=", 1)
        key = key_b.decode("utf-8", errors="ignore")
        if not key:
            continue
        val = val_b.decode("utf-8", errors="replace")
        shell_env[key] = val

    # Keep behavior broad for non-proxy vars, but ensure proxy vars are strictly synced.
    updated = 0
    for key, val in shell_env.items():
        os.environ[key] = val
        updated += 1

    removed = 0
    for key in PROXY_ENV_KEYS:
        if key in shell_env:
            os.environ[key] = shell_env[key]
        elif key in os.environ:
            del os.environ[key]
            removed += 1

    print(f"[proxy] synced env: updated={updated}, removed_proxy_keys={removed}")


def _extract_unique_dois(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    candidates = ["DOI", "doi", "Doi", "DOI Link", "doi_link"]

    doi_col = None
    for col in candidates:
        if col in df.columns:
            doi_col = col
            break

    if doi_col is None:
        raise ValueError(
            f"Could not find DOI column. Looked for: {candidates}. Available columns: {list(df.columns)}"
        )

    dois = []
    for raw in df[doi_col].dropna().astype(str).tolist():
        value = raw.strip()
        if not value:
            continue
        if value.startswith("http://dx.doi.org/"):
            value = value.replace("http://dx.doi.org/", "")
        if value.startswith("https://dx.doi.org/"):
            value = value.replace("https://dx.doi.org/", "")
        if value.startswith("https://doi.org/"):
            value = value.replace("https://doi.org/", "")
        dois.append(value)

    # Preserve order while removing duplicates.
    unique = list(dict.fromkeys(dois))
    return unique


class QwenGenerator:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 384,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        load_kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16

        _run_system_command("proxy_on")
        _apply_proxy_command("proxy_on")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.device = self.model.get_input_embeddings().weight.device
        except Exception:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[:, input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def run_smoke_test(pipeline, doi: str, api_key: str, timeout: int) -> Dict[str, Any]:
    print("\n===== RAG INGESTION SMOKE TEST =====")
    print(f"Smoke DOI: {doi}")
    _run_system_command("proxy_off")
    _apply_proxy_command("proxy_off")
    xml_soup = fetch_elsevier_xml(doi, api_key=api_key, timeout=timeout)
    if xml_soup is None:
        raise RuntimeError(f"Smoke test failed: could not fetch XML for DOI {doi}")

    sections = extract_structured_sections(xml_soup)
    if not sections:
        raise RuntimeError(f"Smoke test failed: no sections extracted for DOI {doi}")

    chunks = pipeline.chunker.chunk_sections(doi, sections)
    if not chunks:
        raise RuntimeError(f"Smoke test failed: no chunks created for DOI {doi}")

    first_chunk = chunks[0]
    chunk_preview = first_chunk.text[:500].replace("\n", " ")
    print(f"First chunk metadata: {first_chunk.metadata}")
    print(f"First chunk preview (500 chars): {chunk_preview}")

    print(f"Embedding model: {pipeline.embedder.model_name}")
    print("Loading embedding model...")
    _run_system_command("proxy_on")
    _apply_proxy_command("proxy_on")
    # Explicitly force model load so failures are surfaced early.
    pipeline.embedder._ensure_model()
    model_obj = pipeline.embedder._model
    model_type = type(model_obj).__name__ if model_obj is not None else "unknown"
    print(f"Embedding model loaded: {model_type}")

    first_embedding = pipeline.embedder.embed_texts([first_chunk.text])
    if first_embedding.size == 0:
        raise RuntimeError(f"Smoke test failed: empty embedding for DOI {doi}")

    vector = first_embedding[0]
    print(f"First embedding vector shape: {vector.shape}")
    print(f"First embedding first 12 dims: {vector[:12].tolist()}")
    print("===== SMOKE TEST PASSED =====\n")

    return {
        "doi": doi,
        "sections": len(sections),
        "chunks": len(chunks),
        "embedding_dim": int(vector.shape[0]),
        "chunk_preview": chunk_preview,
        "first_vector_preview": vector[:12].tolist(),
    }


def _build_rag_user_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        doi = meta.get("doi", "unknown")
        section = meta.get("section_title", "unknown")
        text = chunk.get("text", "")
        context_lines.append(f"[{i}] DOI: {doi} | Section: {section}\n{text}")

    context_text = "\n\n".join(context_lines) if context_lines else "No context retrieved."
    return (
        "Use only the provided context to answer the question. "
        "If the answer is not in the context, say you do not have enough evidence. "
        "Use citation markers [n] to cite the context entries.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def run_ingestion(
    pipeline,
    dois: List[str],
    api_key: str,
    timeout: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    results = []
    success = 0
    failed = 0

    for doi in tqdm(dois, desc="Ingesting DOIs"):
        try:
            _run_system_command("proxy_off")
            _apply_proxy_command("proxy_off")
            xml_soup = fetch_elsevier_xml(doi=doi, api_key=api_key, timeout=timeout)
            if xml_soup is None:
                result = {
                    "doi": doi,
                    "status": "failed",
                    "reason": "fetch_failed",
                    "chunks_added": 0,
                }
            else:
                _run_system_command("proxy_on")
                _apply_proxy_command("proxy_on")
                result = pipeline.ingest_xml(doi=doi, xml_input=xml_soup)
        except Exception as exc:
            result = {
                "doi": doi,
                "status": "failed",
                "reason": str(exc),
                "chunks_added": 0,
            }

        results.append(result)
        if result.get("status") == "success":
            success += 1
        else:
            failed += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {
        "success": success,
        "failed": failed,
        "total": len(dois),
        "results": results,
        "total_vectors": pipeline.vector_store.count(),
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingest DOI papers and compare Qwen-7B with/without RAG on the same question"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--question", required=True)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--model-cache-dir", default=os.getenv("MODEL_CACHE_DIR", None))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-dois", type=int, default=DEFAULT_TEST_MAX_DOIS)
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--output", default="data/qwen7b_rag_comparison.json")

    args = parser.parse_args()

    api_key = os.getenv("ELSEVIER_API_KEY", "") or os.getenv("ELSEVIER_API", "")
    if not args.skip_ingestion and not api_key:
        raise RuntimeError(
            "ELSEVIER_API_KEY is not set. Please add it to .env or export it before running ingestion."
        )

    dois = _extract_unique_dois(args.data_path)
    if args.max_dois is not None:
        dois = dois[: args.max_dois]

    pipeline = build_default_pipeline(
        chroma_dir=args.chroma_dir,
        chroma_collection=args.collection,
        embedding_model=args.embedding_model,
        llm_model=None,
    )

    ingestion_summary = None
    smoke_test_summary = None
    if not args.skip_ingestion:
        print(f"Found {len(dois)} unique DOIs in {args.data_path}")
        if not args.skip_smoke_test:
            smoke_test_summary = run_smoke_test(
                pipeline=pipeline,
                doi=dois[0],
                api_key=api_key,
                timeout=args.timeout,
            )
        ingestion_summary = run_ingestion(
            pipeline=pipeline,
            dois=dois,
            api_key=api_key,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )
        print(
            f"Ingestion done: success={ingestion_summary['success']} "
            f"failed={ingestion_summary['failed']} total_vectors={ingestion_summary['total_vectors']}"
        )

    no_rag_answer = None
    rag_answer = None
    retrieved: List[Dict[str, Any]] = []

    if not args.skip_generation:
        _run_system_command("proxy_on")
        _apply_proxy_command("proxy_on")
        generator = QwenGenerator(
            model_name=args.llm_model,
            cache_dir=args.model_cache_dir,
            max_new_tokens=384,
            temperature=0.1,
            top_p=0.9,
        )

        no_rag_system = "You are a helpful scientific assistant."
        no_rag_user = args.question
        no_rag_answer = generator.generate(no_rag_system, no_rag_user)

        _run_system_command("proxy_on")
        _apply_proxy_command("proxy_on")
        retrieved = pipeline.retriever.retrieve(args.question, top_k=args.top_k)
        rag_system = "You are a scientific assistant that answers only from provided evidence."
        rag_user = _build_rag_user_prompt(args.question, retrieved)
        rag_answer = generator.generate(rag_system, rag_user)

    output_payload = {
        "timestamp": datetime.now().isoformat(),
        "question": args.question,
        "chroma_dir": args.chroma_dir,
        "collection": args.collection,
        "llm_model": args.llm_model,
        "top_k": args.top_k,
        "smoke_test": smoke_test_summary,
        "ingestion": ingestion_summary,
        "no_rag": {
            "answer": no_rag_answer,
        },
        "with_rag": {
            "answer": rag_answer,
            "retrieved_chunks": retrieved,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    if not args.skip_generation:
        print("\n===== NO RAG ANSWER =====")
        print(no_rag_answer)
        print("\n===== WITH RAG ANSWER =====")
        print(rag_answer)
    else:
        print("\nSkipped answer generation (--skip-generation).")
    print(f"\nSaved comparison output to {args.output}")


if __name__ == "__main__":
    main()
