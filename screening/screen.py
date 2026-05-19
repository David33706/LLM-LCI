"""
screening/screen.py
-------------------
Main CLI entry point for LCA paper screening.
Auto-detects backend: CUDA GPU → HuggingFace, otherwise → Ollama.

Usage:
    python -m screening.screen --input List_1.xls --output screening/results/List_1_screened.csv
    python -m screening.screen --input List_2.xls --output screening/results/List_2_screened.csv
    python -m screening.screen --input List_1.xls --output screening/results/List_1_screened.csv --backend ollama
    python -m screening.screen --input data.csv --output screening/results/data_screened.csv --backend hf --hf-model Qwen/Qwen2.5-7B-Instruct

Options:
    --backend       Force backend: "ollama" or "hf" (default: auto-detect)
    --model         Ollama model tag (default: qwen3:8b-q8_0)
    --hf-model      HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)
    --hf-quant      HuggingFace quantization: none, 8bit, 16bit (default: none)
    --resume        Resume from where a previous run left off
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_input(path: str, title_col: str, abstract_col: str) -> pd.DataFrame:
    """Load input file (.xls, .xlsx, or .csv)."""
    p = Path(path)
    if p.suffix == ".xls":
        df = pd.read_excel(path, engine="xlrd")
    elif p.suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif p.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    if title_col not in df.columns:
        raise ValueError(f"Column '{title_col}' not found. Available: {list(df.columns)}")
    if abstract_col not in df.columns:
        raise ValueError(f"Column '{abstract_col}' not found. Available: {list(df.columns)}")

    return df


def select_backend(args):
    """Auto-detect or manually select backend."""
    if args.backend == "hf":
        from screening.hf_backend import HuggingFaceBackend
        return HuggingFaceBackend(model_name=args.hf_model, quantization=args.hf_quant)

    if args.backend == "ollama":
        from screening.ollama_backend import OllamaBackend
        backend = OllamaBackend(model=args.model)
        if not backend.is_available():
            print(f"ERROR: Ollama not running or model '{args.model}' not found.")
            print("  Start Ollama:  ollama serve")
            print(f"  Pull model:    ollama pull {args.model}")
            sys.exit(1)
        return backend

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA detected → using HuggingFace backend")
            from screening.hf_backend import HuggingFaceBackend
            return HuggingFaceBackend(model_name=args.hf_model, quantization=args.hf_quant)
    except ImportError:
        pass

    print("No CUDA → using Ollama backend")
    from screening.ollama_backend import OllamaBackend
    backend = OllamaBackend(model=args.model)
    if not backend.is_available():
        print(f"ERROR: Ollama not running or model '{args.model}' not found.")
        print("  Start Ollama:  ollama serve")
        print(f"  Pull model:    ollama pull {args.model}")
        sys.exit(1)
    return backend


def save_progress(df: pd.DataFrame, results: list, output_path: str):
    """Merge results into the original dataframe and save."""
    out = df.copy()
    out["LCA_prediction"] = None
    out["LCA_raw_response"] = None
    out["LCA_elapsed_s"] = None
    out["abstract_missing"] = None

    for r in results:
        out.at[r["idx"], "LCA_prediction"] = r["prediction"]
        out.at[r["idx"], "LCA_raw_response"] = r["raw_response"]
        out.at[r["idx"], "LCA_elapsed_s"] = r["elapsed_s"]
        out.at[r["idx"], "abstract_missing"] = r["abstract_missing"]

    out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Screen papers for LCA relevance")
    parser.add_argument("--input", required=True, help="Input file (.xls, .xlsx, .csv)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--backend", choices=["ollama", "hf", "auto"], default="auto",
                        help="Inference backend (default: auto-detect)")
    parser.add_argument("--model", default="qwen3:8b-q8_0", help="Ollama model tag")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model")
    parser.add_argument("--hf-quant", choices=["none", "8bit", "16bit"], default="none",
                        help="HuggingFace quantization level")
    parser.add_argument("--title-col", default="Article Title", help="Title column name")
    parser.add_argument("--abstract-col", default="Abstract", help="Abstract column name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_input(args.input, args.title_col, args.abstract_col)
    total = len(df)
    print(f"Loaded {total} papers from {args.input}")

    # Handle resume
    start_idx = 0
    if args.resume and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        start_idx = len(existing)
        print(f"Resuming from row {start_idx} ({start_idx}/{total} already done)")

    # Select backend
    backend = select_backend(args)
    print(f"Backend: {backend.__class__.__name__}")
    print(f"{'='*60}")

    # Run classification
    results = []
    failed = 0

    for idx in range(start_idx, total):
        row = df.iloc[idx]
        title = str(row[args.title_col]) if pd.notna(row[args.title_col]) else ""
        abstract = str(row[args.abstract_col]) if pd.notna(row[args.abstract_col]) else ""
        abstract_missing = abstract.strip() == ""

        result = backend.classify(title, abstract)

        if result["prediction"] is None:
            failed += 1

        results.append({
            "idx": idx,
            "prediction": result["prediction"],
            "raw_response": result["raw_response"],
            "elapsed_s": result["elapsed_s"],
            "abstract_missing": abstract_missing,
        })

        pred_display = result["prediction"] if result["prediction"] is not None else "FAIL"
        flag = " [NO ABSTRACT]" if abstract_missing else ""
        print(f"  [{idx+1}/{total}] pred={pred_display}  time={result['elapsed_s']}s{flag}  {title[:60]}...")

        # Incremental save every 50 rows
        if (idx + 1) % 50 == 0 or idx == total - 1:
            save_progress(df, results, args.output)

    # Final save
    save_progress(df, results, args.output)

    # Summary
    preds = [r["prediction"] for r in results if r["prediction"] is not None]
    positive = sum(1 for p in preds if p == 1)
    negative = sum(1 for p in preds if p == 0)

    print(f"\n{'='*60}")
    print(f"Done. {total} papers processed.")
    print(f"  LCA relevant (1): {positive}")
    print(f"  Not relevant (0): {negative}")
    print(f"  Parse failures:   {failed}")
    print(f"  Missing abstracts: {sum(1 for r in results if r['abstract_missing'])}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()