import os
import gc
import random
import re
import statistics
import subprocess
import time
from datetime import datetime
from typing import Dict

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from utils.elsevier import extract_methodology_text, fetch_elsevier_xml


DATA_PATH = "data/uselca1.csv"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
OUTPUT_DATA_PATH = "data/uselca_with_methods_predictions.csv"
REPORT_PATH = "data/zero_shot_classification_report_with_methods_no_limit.txt"
SMOKE_METHODS_PATH = "data/methodology_smoke_test.txt"
PROMPT_SMOKE_PATH = "prompt_smoke_test.txt"
BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_SEED = 42
MAX_NEW_TOKENS = 12
MAX_PROMPT_TOKENS = 1024
USE_GENERATION_CACHE = True
EMPTY_CACHE_EVERY_N_STEPS = 20
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MODEL_NAMES = [
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-it",
]

SYSTEM_PROMPT = """
You are a helpful research assistant with 20 years of experience in environmental science.
Classify whether an article contains methods or concepts of Life Cycle Assessment (LCA), using only the provided paper text.

Definitions:
1. Life Cycle Assessment (LCA):
- ISO: Compilation and evaluation of inputs, outputs and potential environmental impacts of a product system throughout its life cycle.
- SETAC: Evaluation of environmental burdens across the life cycle, from raw material extraction to disposal.

Important notes:
- The article must include LCA methods or concepts consistent with these definitions.
- Life Cycle Costing (LCC) and Social Life Cycle Assessment (S-LCA) are not counted as LCA for this task.

Output exactly one label:
- 1 = applies LCA
- 0 = does not apply LCA
Do not output anything else.
""".strip()


def _run_system_command(command: str) -> None:
    """
    Execute shell-level helper commands such as proxy_on/proxy_off.
    Uses `bash -lc` so user shell commands/functions are available.
    """
    # print(f"[proxy] running: {command}")
    proc = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # print(f"[proxy] warning: command failed ({command}) rc={proc.returncode} err={stderr}")
    elif proc.stdout:
        print(proc.stdout.strip())


def _apply_proxy_command(command: str) -> None:
    """
    Run proxy command in a shell and import that shell's environment into
    this Python process.
    """
    # print(f"[proxy] apply env via: {command}")
    proc = subprocess.run(
        ["bash", "-lc", f"{command} >/dev/null 2>&1; env -0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        # print(f"[proxy] warning: failed to apply env from {command}; rc={proc.returncode} err={stderr}")
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

    # print(f"[proxy] synced env: updated={updated}, removed_proxy_keys={removed}")


def normalize_doi(doi, doi_link):
    candidate = doi if pd.notna(doi) and str(doi).strip() else doi_link
    if pd.isna(candidate) or candidate is None:
        return ""

    candidate = str(candidate).strip()
    if not candidate:
        return ""

    if candidate.lower().startswith("http"):
        match = re.search(r"10\.\d{4,9}/[^\s?#]+", candidate)
        if match:
            return match.group(0).rstrip("/")
        candidate = candidate.rsplit("/", 1)[-1]

    return candidate.rstrip("/")


def parse_binary_label(text):
    if text is None:
        return None
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    strict = re.search(r"^\s*([01])\b", str(text))
    if strict:
        return int(strict.group(1))
    fallback = re.search(r"\b([01])\b", str(text))
    return int(fallback.group(1)) if fallback else None


def compute_binary_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


def compute_bootstrap_std(y_true, y_pred, n_bootstrap=300, seed=42):
    n = len(y_true)
    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0

    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    rng = random.Random(seed)

    acc_vals = []
    prec_vals = []
    rec_vals = []
    f1_vals = []

    for _ in range(n_bootstrap):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        s_true = y_true.iloc[sample_idx]
        s_pred = y_pred.iloc[sample_idx]
        acc, prec, rec, f1 = compute_binary_metrics(s_true, s_pred)
        acc_vals.append(acc)
        prec_vals.append(prec)
        rec_vals.append(rec)
        f1_vals.append(f1)

    return (
        statistics.pstdev(acc_vals),
        statistics.pstdev(prec_vals),
        statistics.pstdev(rec_vals),
        statistics.pstdev(f1_vals),
    )


def infer_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


def infer_context_limit(model, tokenizer):
    candidates = []

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        candidates.append(tokenizer_limit)

    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(model_limit, int) and 0 < model_limit < 100000:
        candidates.append(model_limit)

    return min(candidates) if candidates else None


def _get_chat_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def _render_prompt(chat_tokenizer, model_name, title, abstract, methods_text=None):
    user_parts = [f"Title: {title}", f"Abstract: {abstract}"]
    if methods_text:
        user_parts.append(f"Methodology: {methods_text}")
    user_parts.append("Classification (output only 1 or 0):")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    if hasattr(chat_tokenizer, "apply_chat_template"):
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "gemma-4" in model_name or "Qwen3.5" in model_name:
                kwargs["enable_thinking"] = False
            return chat_tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass

    base_text = [SYSTEM_PROMPT, ""]
    base_text.extend(user_parts)
    return "\n".join(base_text)


def _count_tokens(chat_tokenizer, text):
    tokenized = chat_tokenizer(text, add_special_tokens=False)
    return len(tokenized["input_ids"])


def _fit_methods_text_to_budget(chat_tokenizer, model_name, title, abstract, methods_text, max_prompt_tokens):
    if not methods_text:
        return ""

    # If the full prompt already fits, keep methodology unchanged.
    candidate = _render_prompt(chat_tokenizer, model_name, title, abstract, methods_text)
    if _count_tokens(chat_tokenizer, candidate) <= max_prompt_tokens:
        return methods_text

    method_ids = chat_tokenizer(methods_text, add_special_tokens=False)["input_ids"]
    if not method_ids:
        return ""

    # Binary search the largest prefix of methodology tokens that still fits.
    lo = 0
    hi = len(method_ids)
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        trimmed = chat_tokenizer.decode(method_ids[:mid], skip_special_tokens=True)
        prompt = _render_prompt(chat_tokenizer, model_name, title, abstract, trimmed)
        prompt_tokens = _count_tokens(chat_tokenizer, prompt)
        if prompt_tokens <= max_prompt_tokens:
            best = trimmed
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def build_input_text(tokenizer, model_name, title, abstract, methods_text=None, max_prompt_tokens=None):
    chat_tokenizer = _get_chat_tokenizer(tokenizer)

    if not methods_text or max_prompt_tokens is None:
        return _render_prompt(chat_tokenizer, model_name, title, abstract, methods_text)
    # if model_name.lower() == 'qwen/qwen2.5-7b-instruct':
    #     # smaller model so skip trimming the methods section to preserve as much context as possible.
    #     return _render_prompt(chat_tokenizer, model_name, title, abstract, methods_text)
    trimmed_methods = _fit_methods_text_to_budget(
        chat_tokenizer=chat_tokenizer,
        model_name=model_name,
        title=title,
        abstract=abstract,
        methods_text=methods_text,
        max_prompt_tokens=max_prompt_tokens,
    )
    return _render_prompt(chat_tokenizer, model_name, title, abstract, trimmed_methods)


def load_model_and_tokenizer(model_name):
    _run_system_command("proxy_on")
    _apply_proxy_command("proxy_on")

    load_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16

    if "Qwen3.5" in model_name:
        model = transformers.AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if "gemma-4" in model_name or "Qwen3.5" in model_name:
        tokenizer = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        pad_token_id = tokenizer.tokenizer.eos_token_id
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, pad_token_id


def tokenize_inputs(tokenizer, model_name, input_text, input_device, context_limit):
    encode_kwargs = {"return_tensors": "pt"}
    target_limit = context_limit
    if target_limit is None:
        target_limit = MAX_PROMPT_TOKENS
    else:
        target_limit = min(target_limit, MAX_PROMPT_TOKENS)

    if "gemma-4" in model_name or "Qwen3.5" in model_name:
        inputs = tokenizer(text=input_text, **encode_kwargs)
    else:
        inputs = tokenizer(input_text, **encode_kwargs)

    # Safety net: preserve the end of the prompt (assistant cue + classification request).
    if "input_ids" in inputs:
        seq_len = int(inputs["input_ids"].shape[1])
        if target_limit is not None and seq_len > target_limit:
            for key, value in list(inputs.items()):
                if hasattr(value, "shape") and len(value.shape) == 2 and int(value.shape[1]) == seq_len:
                    inputs[key] = value[:, -target_limit:]

    if hasattr(inputs, "to"):
        inputs = inputs.to(input_device)
    else:
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

    return inputs


def generate_prediction(model, tokenizer, model_name, input_text, input_device, pad_token_id, context_limit):
    decoder = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inputs = tokenize_inputs(tokenizer, model_name, input_text, input_device, context_limit)
    input_tokens = int(inputs["input_ids"].shape[1])

    if input_device.type == "cuda":
        torch.cuda.synchronize(input_device)
        torch.cuda.reset_peak_memory_stats(input_device)

    start_time = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=USE_GENERATION_CACHE,
                pad_token_id=pad_token_id,
            )
    except torch.OutOfMemoryError:
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

        retry_tokens = max(256, min(768, input_tokens // 2))
        for key, value in list(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) == 2 and int(value.shape[1]) >= retry_tokens:
                inputs[key] = value[:, -retry_tokens:]
        input_tokens = int(inputs["input_ids"].shape[1])
        print(
            f"OOM during generation; retrying with truncated prompt ({input_tokens} tokens)."
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=False,
                pad_token_id=pad_token_id,
            )
    if input_device.type == "cuda":
        torch.cuda.synchronize(input_device)
    elapsed_seconds = time.perf_counter() - start_time

    generated_tokens = outputs[0][input_tokens:]
    generated_tokens_count = int(generated_tokens.shape[0])
    response = decoder.decode(generated_tokens, skip_special_tokens=True).strip()
    predicted_label = parse_binary_label(response)

    peak_allocated_mb = None
    peak_reserved_mb = None
    if input_device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(input_device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(input_device) / (1024**2)

    # Release per-sample tensors as early as possible to avoid allocator growth.
    del outputs
    del inputs
    del generated_tokens

    return {
        "response": response,
        "predicted_label": predicted_label,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
    }


def fetch_methodology_dataset(data):
    fetched_records = []
    smoke_written = False

    _run_system_command("proxy_off")
    _apply_proxy_command("proxy_off")

    for row in tqdm(data.to_dict(orient="records"), total=len(data), desc="Fetching Elsevier XML"):
        doi = normalize_doi(row.get("DOI"), row.get("DOI Link"))
        if not doi:
            continue

        _run_system_command("proxy_off")
        _apply_proxy_command("proxy_off")
        xml_doc = fetch_elsevier_xml(doi)
        if xml_doc is None:
            continue

        methods_text = extract_methodology_text(xml_doc)
        if not methods_text.strip():
            continue

        if not smoke_written:
            with open(SMOKE_METHODS_PATH, "w", encoding="utf-8") as handle:
                handle.write(f"DOI: {doi}\n")
                handle.write(f"Title: {row.get('Article Title', '')}\n\n")
                handle.write(methods_text.strip())
                handle.write("\n")
            print(f"Saved methodology smoke test to: {SMOKE_METHODS_PATH}")
            smoke_written = True

        fetched_records.append(
            {
                "DOI": doi,
                "DOI Link": row.get("DOI Link"),
                "Article Title": row.get("Article Title"),
                "Abstract": row.get("Abstract"),
                "Methodology": methods_text,
                "Label": row.get("Label"),
            }
        )

    return pd.DataFrame(fetched_records)


def fmt_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"


def main():
    data = pd.read_csv(DATA_PATH)
    required_cols = ["Article Title", "Abstract", "Label", "DOI", "DOI Link"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if os.path.exists(OUTPUT_DATA_PATH):
        fetched_data = pd.read_csv(OUTPUT_DATA_PATH)
        required_cached_cols = ["Article Title", "Abstract", "Label", "DOI", "Methodology"]
        missing_cached_cols = [col for col in required_cached_cols if col not in fetched_data.columns]
        if missing_cached_cols:
            print(
                f"Cached dataset found at {OUTPUT_DATA_PATH} but missing columns {missing_cached_cols}; refetching."
            )
            fetched_data = fetch_methodology_dataset(data)
            if fetched_data.empty:
                raise ValueError("No papers were fetched successfully with a non-empty methodology section.")
            fetched_data.to_csv(OUTPUT_DATA_PATH, index=False)
            print(f"Saved fetched dataset to: {OUTPUT_DATA_PATH}")
        else:
            print(f"Using cached fetched dataset from: {OUTPUT_DATA_PATH}")
    else:
        fetched_data = fetch_methodology_dataset(data)
        if fetched_data.empty:
            raise ValueError("No papers were fetched successfully with a non-empty methodology section.")
        fetched_data.to_csv(OUTPUT_DATA_PATH, index=False)
        print(f"Saved fetched dataset to: {OUTPUT_DATA_PATH}")

    print(f"Rows with methodology section: {len(fetched_data)}")

    run_summary = []
    comparison_rows = []
    prompt_smoke_written = False

    for model_name in MODEL_NAMES:
        print(f"Running model: {model_name}")
        try:
            model, tokenizer, pad_token_id = load_model_and_tokenizer(model_name)
        except ImportError as err:
            print(f"Skipping {model_name} due to ImportError: {err}")
            run_summary.append(
                {
                    "model": model_name,
                    "status": "skipped-import-error",
                    "coverage_baseline": None,
                    "coverage_methods": None,
                    "coverage_paired": None,
                    "baseline_accuracy": None,
                    "baseline_precision": None,
                    "baseline_recall": None,
                    "baseline_f1": None,
                    "methods_accuracy": None,
                    "methods_precision": None,
                    "methods_recall": None,
                    "methods_f1": None,
                    "baseline_accuracy_std": None,
                    "baseline_precision_std": None,
                    "baseline_recall_std": None,
                    "baseline_f1_std": None,
                    "methods_accuracy_std": None,
                    "methods_precision_std": None,
                    "methods_recall_std": None,
                    "methods_f1_std": None,
                    "mean_token_delta": None,
                    "mean_time_delta_seconds": None,
                    "mean_peak_allocated_delta_mb": None,
                    "mean_peak_reserved_delta_mb": None,
                    "valid_rows": 0,
                    "total_rows": len(fetched_data),
                }
            )
            continue
        except OSError as err:
            print(f"Skipping {model_name}: {err}")
            run_summary.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "coverage_baseline": None,
                    "coverage_methods": None,
                    "coverage_paired": None,
                    "baseline_accuracy": None,
                    "baseline_precision": None,
                    "baseline_recall": None,
                    "baseline_f1": None,
                    "methods_accuracy": None,
                    "methods_precision": None,
                    "methods_recall": None,
                    "methods_f1": None,
                    "baseline_accuracy_std": None,
                    "baseline_precision_std": None,
                    "baseline_recall_std": None,
                    "baseline_f1_std": None,
                    "methods_accuracy_std": None,
                    "methods_precision_std": None,
                    "methods_recall_std": None,
                    "methods_f1_std": None,
                    "mean_token_delta": None,
                    "mean_time_delta_seconds": None,
                    "mean_peak_allocated_delta_mb": None,
                    "mean_peak_reserved_delta_mb": None,
                    "valid_rows": 0,
                    "total_rows": len(fetched_data),
                }
            )
            continue

        input_device = infer_input_device(model)
        context_limit = infer_context_limit(model, tokenizer)
        prompt_token_limit = MAX_PROMPT_TOKENS if context_limit is None else min(context_limit, MAX_PROMPT_TOKENS)
        if hasattr(model, "hf_device_map"):
            devices = sorted({str(value) for value in model.hf_device_map.values()})
            print(f"Device map: {devices}")
        else:
            print(f"Model device: {model.device}")
        if context_limit is not None:
            print(f"Context limit: {context_limit}")

        baseline_predictions = []
        methods_predictions = []
        baseline_input_tokens = []
        methods_input_tokens = []
        baseline_generated_tokens = []
        methods_generated_tokens = []
        baseline_elapsed_seconds = []
        methods_elapsed_seconds = []
        baseline_peak_allocated_mb = []
        methods_peak_allocated_mb = []
        baseline_peak_reserved_mb = []
        methods_peak_reserved_mb = []

        for step_idx, row in enumerate(
            tqdm(fetched_data.to_dict(orient="records"), total=len(fetched_data), desc=f"Inference: {model_name}"),
            start=1,
        ):
            title = "" if pd.isna(row.get("Article Title")) else str(row.get("Article Title"))
            abstract = "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract"))
            methods_text = "" if pd.isna(row.get("Methodology")) else str(row.get("Methodology"))

            baseline_input_text = build_input_text(
                tokenizer=tokenizer,
                model_name=model_name,
                title=title,
                abstract=abstract,
                methods_text=None,
                max_prompt_tokens=prompt_token_limit,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            baseline_result = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                input_text=baseline_input_text,
                input_device=input_device,
                pad_token_id=pad_token_id,
                context_limit=context_limit,
            )

            methods_input_text = build_input_text(
                tokenizer=tokenizer,
                model_name=model_name,
                title=title,
                abstract=abstract,
                methods_text=methods_text,
                max_prompt_tokens=prompt_token_limit,
            )

            if not prompt_smoke_written:
                with open(PROMPT_SMOKE_PATH, "w", encoding="utf-8") as handle:
                    handle.write(f"Model: {model_name}\n")
                    handle.write(f"DOI: {row.get('DOI', '')}\n")
                    handle.write("Variant: methods\n")
                    handle.write("=" * 80 + "\n")
                    handle.write(methods_input_text)
                    handle.write("\n")

            methods_result = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                input_text=methods_input_text,
                input_device=input_device,
                pad_token_id=pad_token_id,
                context_limit=context_limit,
            )

            if not prompt_smoke_written:
                with open(PROMPT_SMOKE_PATH, "a", encoding="utf-8") as handle:
                    handle.write("\n")
                    handle.write("Model output (methods variant):\n")
                    handle.write(methods_result["response"])
                    handle.write("\n")
                    handle.write(f"Parsed label: {methods_result['predicted_label']}\n")
                print(f"Saved prompt smoke test to: {PROMPT_SMOKE_PATH}")
                prompt_smoke_written = True

            baseline_predictions.append(baseline_result["predicted_label"])
            methods_predictions.append(methods_result["predicted_label"])
            baseline_input_tokens.append(baseline_result["input_tokens"])
            methods_input_tokens.append(methods_result["input_tokens"])
            baseline_generated_tokens.append(baseline_result["generated_tokens"])
            methods_generated_tokens.append(methods_result["generated_tokens"])
            baseline_elapsed_seconds.append(baseline_result["elapsed_seconds"])
            methods_elapsed_seconds.append(methods_result["elapsed_seconds"])
            baseline_peak_allocated_mb.append(baseline_result["peak_allocated_mb"])
            methods_peak_allocated_mb.append(methods_result["peak_allocated_mb"])
            baseline_peak_reserved_mb.append(baseline_result["peak_reserved_mb"])
            methods_peak_reserved_mb.append(methods_result["peak_reserved_mb"])

            comparison_rows.append(
                {
                    "model": model_name,
                    "DOI": row.get("DOI"),
                    "Article Title": title,
                    "Label": row.get("Label"),
                    "baseline_prediction": baseline_result["predicted_label"],
                    "methods_prediction": methods_result["predicted_label"],
                    "baseline_input_tokens": baseline_result["input_tokens"],
                    "methods_input_tokens": methods_result["input_tokens"],
                    "token_delta": methods_result["input_tokens"] - baseline_result["input_tokens"],
                    "baseline_generated_tokens": baseline_result["generated_tokens"],
                    "methods_generated_tokens": methods_result["generated_tokens"],
                    "baseline_elapsed_seconds": baseline_result["elapsed_seconds"],
                    "methods_elapsed_seconds": methods_result["elapsed_seconds"],
                    "elapsed_delta_seconds": methods_result["elapsed_seconds"] - baseline_result["elapsed_seconds"],
                    "baseline_peak_allocated_mb": baseline_result["peak_allocated_mb"],
                    "methods_peak_allocated_mb": methods_result["peak_allocated_mb"],
                    "peak_allocated_delta_mb": (
                        None
                        if baseline_result["peak_allocated_mb"] is None or methods_result["peak_allocated_mb"] is None
                        else methods_result["peak_allocated_mb"] - baseline_result["peak_allocated_mb"]
                    ),
                    "baseline_peak_reserved_mb": baseline_result["peak_reserved_mb"],
                    "methods_peak_reserved_mb": methods_result["peak_reserved_mb"],
                    "peak_reserved_delta_mb": (
                        None
                        if baseline_result["peak_reserved_mb"] is None or methods_result["peak_reserved_mb"] is None
                        else methods_result["peak_reserved_mb"] - baseline_result["peak_reserved_mb"]
                    ),
                }
            )

            if torch.cuda.is_available() and step_idx % EMPTY_CACHE_EVERY_N_STEPS == 0:
                gc.collect()
                torch.cuda.empty_cache()

        fetched_data[f"{model_name}_baseline_prediction"] = baseline_predictions
        fetched_data[f"{model_name}_methods_prediction"] = methods_predictions
        fetched_data[f"{model_name}_baseline_input_tokens"] = baseline_input_tokens
        fetched_data[f"{model_name}_methods_input_tokens"] = methods_input_tokens
        fetched_data[f"{model_name}_baseline_generated_tokens"] = baseline_generated_tokens
        fetched_data[f"{model_name}_methods_generated_tokens"] = methods_generated_tokens
        fetched_data[f"{model_name}_baseline_elapsed_seconds"] = baseline_elapsed_seconds
        fetched_data[f"{model_name}_methods_elapsed_seconds"] = methods_elapsed_seconds
        fetched_data[f"{model_name}_baseline_peak_allocated_mb"] = baseline_peak_allocated_mb
        fetched_data[f"{model_name}_methods_peak_allocated_mb"] = methods_peak_allocated_mb
        fetched_data[f"{model_name}_baseline_peak_reserved_mb"] = baseline_peak_reserved_mb
        fetched_data[f"{model_name}_methods_peak_reserved_mb"] = methods_peak_reserved_mb

        baseline_valid_mask = fetched_data["Label"].notna() & fetched_data[f"{model_name}_baseline_prediction"].notna()
        methods_valid_mask = fetched_data["Label"].notna() & fetched_data[f"{model_name}_methods_prediction"].notna()
        paired_valid_mask = baseline_valid_mask & methods_valid_mask

        if paired_valid_mask.sum() == 0:
            print(f"No paired valid predictions for {model_name}; skipping metrics.")
            run_summary.append(
                {
                    "model": model_name,
                    "status": "no-valid-predictions",
                    "coverage_baseline": float(baseline_valid_mask.mean()),
                    "coverage_methods": float(methods_valid_mask.mean()),
                    "coverage_paired": float(paired_valid_mask.mean()),
                    "baseline_accuracy": None,
                    "baseline_precision": None,
                    "baseline_recall": None,
                    "baseline_f1": None,
                    "methods_accuracy": None,
                    "methods_precision": None,
                    "methods_recall": None,
                    "methods_f1": None,
                    "baseline_accuracy_std": None,
                    "baseline_precision_std": None,
                    "baseline_recall_std": None,
                    "baseline_f1_std": None,
                    "methods_accuracy_std": None,
                    "methods_precision_std": None,
                    "methods_recall_std": None,
                    "methods_f1_std": None,
                    "mean_token_delta": None,
                    "mean_time_delta_seconds": None,
                    "mean_peak_allocated_delta_mb": None,
                    "mean_peak_reserved_delta_mb": None,
                    "valid_rows": 0,
                    "total_rows": len(fetched_data),
                }
            )
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        y_true = fetched_data.loc[paired_valid_mask, "Label"].astype(int)
        baseline_y_pred = fetched_data.loc[paired_valid_mask, f"{model_name}_baseline_prediction"].astype(int)
        methods_y_pred = fetched_data.loc[paired_valid_mask, f"{model_name}_methods_prediction"].astype(int)

        baseline_accuracy, baseline_precision, baseline_recall, baseline_f1 = compute_binary_metrics(
            y_true, baseline_y_pred
        )
        methods_accuracy, methods_precision, methods_recall, methods_f1 = compute_binary_metrics(
            y_true, methods_y_pred
        )
        baseline_accuracy_std, baseline_precision_std, baseline_recall_std, baseline_f1_std = compute_bootstrap_std(
            y_true, baseline_y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )
        methods_accuracy_std, methods_precision_std, methods_recall_std, methods_f1_std = compute_bootstrap_std(
            y_true, methods_y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )

        mean_token_delta = statistics.mean(
            methods_input_tokens[idx] - baseline_input_tokens[idx] for idx in range(len(baseline_input_tokens))
        )
        mean_time_delta_seconds = statistics.mean(
            methods_elapsed_seconds[idx] - baseline_elapsed_seconds[idx] for idx in range(len(baseline_elapsed_seconds))
        )
        if all(value is not None for value in baseline_peak_allocated_mb) and all(
            value is not None for value in methods_peak_allocated_mb
        ):
            mean_peak_allocated_delta_mb = statistics.mean(
                methods_peak_allocated_mb[idx] - baseline_peak_allocated_mb[idx]
                for idx in range(len(baseline_peak_allocated_mb))
            )
        else:
            mean_peak_allocated_delta_mb = None
        if all(value is not None for value in baseline_peak_reserved_mb) and all(
            value is not None for value in methods_peak_reserved_mb
        ):
            mean_peak_reserved_delta_mb = statistics.mean(
                methods_peak_reserved_mb[idx] - baseline_peak_reserved_mb[idx]
                for idx in range(len(baseline_peak_reserved_mb))
            )
        else:
            mean_peak_reserved_delta_mb = None

        print(f"****** Model: {model_name} Results ******")
        print(f"Baseline coverage: {baseline_valid_mask.mean():.4f}")
        print(f"Methods coverage: {methods_valid_mask.mean():.4f}")
        print(f"Paired coverage: {paired_valid_mask.mean():.4f}")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"Baseline precision: {baseline_precision:.4f}")
        print(f"Baseline recall: {baseline_recall:.4f}")
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"Methods accuracy: {methods_accuracy:.4f}")
        print(f"Methods precision: {methods_precision:.4f}")
        print(f"Methods recall: {methods_recall:.4f}")
        print(f"Methods F1: {methods_f1:.4f}")
        print(
            f"Baseline std -> Acc: {baseline_accuracy_std:.4f}, Prec: {baseline_precision_std:.4f}, "
            f"Rec: {baseline_recall_std:.4f}, F1: {baseline_f1_std:.4f}"
        )
        print(
            f"Methods std -> Acc: {methods_accuracy_std:.4f}, Prec: {methods_precision_std:.4f}, "
            f"Rec: {methods_recall_std:.4f}, F1: {methods_f1_std:.4f}"
        )
        print(f"Mean token delta: {mean_token_delta:.2f}")
        print(f"Mean time delta (s): {mean_time_delta_seconds:.4f}")
        if mean_peak_allocated_delta_mb is not None:
            print(f"Mean peak allocated delta (MB): {mean_peak_allocated_delta_mb:.4f}")
        if mean_peak_reserved_delta_mb is not None:
            print(f"Mean peak reserved delta (MB): {mean_peak_reserved_delta_mb:.4f}")
        print("***************************************")

        run_summary.append(
            {
                "model": model_name,
                "status": "ok",
                "coverage_baseline": float(baseline_valid_mask.mean()),
                "coverage_methods": float(methods_valid_mask.mean()),
                "coverage_paired": float(paired_valid_mask.mean()),
                "baseline_accuracy": baseline_accuracy,
                "baseline_precision": baseline_precision,
                "baseline_recall": baseline_recall,
                "baseline_f1": baseline_f1,
                "methods_accuracy": methods_accuracy,
                "methods_precision": methods_precision,
                "methods_recall": methods_recall,
                "methods_f1": methods_f1,
                "baseline_accuracy_std": baseline_accuracy_std,
                "baseline_precision_std": baseline_precision_std,
                "baseline_recall_std": baseline_recall_std,
                "baseline_f1_std": baseline_f1_std,
                "methods_accuracy_std": methods_accuracy_std,
                "methods_precision_std": methods_precision_std,
                "methods_recall_std": methods_recall_std,
                "methods_f1_std": methods_f1_std,
                "mean_token_delta": mean_token_delta,
                "mean_time_delta_seconds": mean_time_delta_seconds,
                "mean_peak_allocated_delta_mb": mean_peak_allocated_delta_mb,
                "mean_peak_reserved_delta_mb": mean_peak_reserved_delta_mb,
                "valid_rows": int(paired_valid_mask.sum()),
                "total_rows": len(fetched_data),
            }
        )

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fetched_data.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_DATA_PATH}")

    comparison_path = "data/zero_shot_classification_with_methods_rows.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"Saved row-level comparison to: {comparison_path}")

    report_lines = [
        "Zero-Shot LCA Classification Benchmark With Methodology Section",
        "=" * 68,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {DATA_PATH}",
        f"Rows fetched with methodology section: {len(fetched_data)}",
        f"Models attempted: {len(MODEL_NAMES)}",
        f"Bootstrap sensitivity: {BOOTSTRAP_SAMPLES} resamples (seed={BOOTSTRAP_SEED})",
        f"Inference variants: baseline vs baseline+methodology",
        "",
        "Per-model summary",
        "-" * 68,
    ]

    header = (
        f"{'Model':<36} {'Status':<20} {'BaseCov':>8} {'MethCov':>8} {'PairCov':>8} "
        f"{'BaseF1':>8} {'MethF1':>8} {'TokDelta':>10} {'TimeDelta':>10} {'PairRows':>10}"
    )
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for item in run_summary:
        pair_rows = f"{item['valid_rows']}/{item['total_rows']}"
        report_lines.append(
            f"{item['model'][:36]:<36} {item['status'][:20]:<20} {fmt_metric(item['coverage_baseline']):>8} "
            f"{fmt_metric(item['coverage_methods']):>8} {fmt_metric(item['coverage_paired']):>8} "
            f"{fmt_metric(item['baseline_f1']):>8} {fmt_metric(item['methods_f1']):>8} "
            f"{fmt_metric(item['mean_token_delta']):>10} {fmt_metric(item['mean_time_delta_seconds']):>10} "
            f"{pair_rows:>10}"
        )

    report_lines.extend(["", "Metric Std (Sensitivity)", "-" * 68])
    std_header = (
        f"{'Model':<36} {'BaseF1Std':>10} {'MethF1Std':>10} {'BaseAccStd':>10} {'MethAccStd':>10}"
    )
    report_lines.append(std_header)
    report_lines.append("-" * len(std_header))

    for item in run_summary:
        report_lines.append(
            f"{item['model'][:36]:<36} {fmt_metric(item['baseline_f1_std']):>10} {fmt_metric(item['methods_f1_std']):>10} "
            f"{fmt_metric(item['baseline_accuracy_std']):>10} {fmt_metric(item['methods_accuracy_std']):>10}"
        )

    report_lines.extend(
        [
            "",
            "Notes",
            "- The fetched dataset only includes papers where Elsevier XML retrieval succeeded and a non-empty methodology section was found.",
            "- Baseline uses title + abstract only.",
            "- Methods uses title + abstract + extracted methodology section only.",
            "- Token delta is the mean increase in input tokens after adding methodology.",
            "- Time delta is the mean per-row generation time increase after adding methodology.",
            "- GPU memory deltas are tracked per-row when CUDA is available.",
            f"- Row-level results are saved in {comparison_path}.",
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    print(f"Saved run report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
