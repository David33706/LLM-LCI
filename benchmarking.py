import transformers
import torch
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime
import random
import statistics


DATA_PATH = "data/uselca1.csv"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_SEED = 42

SYSTEM_PROMPT = """
You are a helpful research assistant with 20 years of experience in environmental science.
Classify whether an article contains methods or concepts of Life Cycle Assessment (LCA), using only title and abstract.

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


def build_input_text(tokenizer, title, abstract):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
                "Classification (output only 1 or 0):"
            ),
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # return tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "gemma-4" in model_name:
                kwargs["enable_thinking"] = False  # Required for Gemma 4 as shown in https://huggingface.co/google/gemma-4-26B-A4B-it
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        "Classification (output only 1 or 0):"
    )


def parse_binary_label(text):
    match = re.search(r"\b([01])\b", text)
    return int(match.group(1)) if match else None


def infer_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


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


data = pd.read_csv(DATA_PATH)
required_cols = ["Article Title", "Abstract", "Label"]
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

base_data = data.copy()

model_names = [
    # "google/gemma-3-1b-it",
    # "google/gemma-3-4b-it",
    # "google/gemma-3-12b-it",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct"
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-it"
]

quantization_modes = {
    # "8bit": {"load_in_8bit": True},
    # "16bit": {"torch_dtype": torch.float16},
    # no quantization full model precision BF16
    "full_precision": {"torch_dtype": torch.bfloat16}
}

for quant_label, quant_kwargs in quantization_modes.items():
    print(f"\n===== Running quantization mode: {quant_label} =====")
    data = base_data.copy()
    run_summary = []

    for model_name in model_names:
        print(f"Running model: {model_name} [{quant_label}]")
        load_kwargs = {
            "cache_dir": MODEL_CACHE_DIR,
            "low_cpu_mem_usage": True,
            # **quant_kwargs,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except ImportError as err:
            if "accelerate" in str(err).lower() and "device_map" in load_kwargs:
                print("Accelerate is missing; retrying model load on default device.")
                load_kwargs.pop("device_map", None)
                load_kwargs.pop("low_cpu_mem_usage", None)
                model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            else:
                print(f"Skipping {model_name} [{quant_label}] due to ImportError: {err}")
                run_summary.append(
                    {
                        "model": model_name,
                        "status": "skipped-import-error",
                        "coverage": None,
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "accuracy_std": None,
                        "precision_std": None,
                        "recall_std": None,
                        "f1_std": None,
                        "valid_rows": 0,
                        "total_rows": len(data),
                    }
                )
                continue
        except OSError as err:
            print(f"Skipping {model_name} [{quant_label}]: {err}")
            run_summary.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "coverage": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy_std": None,
                    "precision_std": None,
                    "recall_std": None,
                    "f1_std": None,
                    "valid_rows": 0,
                    "total_rows": len(data),
                }
            )
            continue
        # if "gemma-4" in model_name:
        #     tokenizer = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        # else:
        #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        #     if tokenizer.pad_token_id is None:
        #         tokenizer.pad_token = tokenizer.eos_token
        if "gemma-4" in model_name:
            tokenizer = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            pad_token_id = tokenizer.tokenizer.eos_token_id
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.pad_token_id

        input_device = infer_input_device(model)
        if hasattr(model, "hf_device_map"):
            devices = sorted({str(v) for v in model.hf_device_map.values()})
            print(f"Device map: {devices}")
        else:
            print(f"Model device: {model.device}")

        preds = []
        for _, title, abstract in tqdm(
            zip(data.index, data["Article Title"], data["Abstract"]), total=len(data)
        ):
            title = "" if pd.isna(title) else str(title)
            abstract = "" if pd.isna(abstract) else str(abstract)
            input_text = build_input_text(tokenizer, title=title, abstract=abstract)
            if "gemma-4" in model_name:
                inputs = tokenizer(text=input_text, return_tensors="pt").to(input_device)
            else:
                inputs = tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(input_device) for k, v in inputs.items()}
            # if model_name == "google/gemma-4-26B-A4B-it":
            #     # using recommendations from model card
            #     temperature = 1.0
            #     top_p = 0.95
            #     top_k = 64

            with torch.no_grad():
                # if model_name == "google/gemma-4-26B-A4B-it":
                #     outputs = model.generate(
                #         **inputs,
                #         max_new_tokens=4,
                #         do_sample=True,
                #         temperature=1.0,
                #         top_p=0.95,
                #         top_k=64,
                #         pad_token_id=pad_token_id,
                #     )
                # else:
                #     outputs = model.generate(
                #         **inputs,
                #         max_new_tokens=4,
                #         do_sample=False,
                #         pad_token_id=pad_token_id,
                #     )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                ) 
                # actually using same generation params for a much fairer comparison + deterministic is probably better for classification task
                # in other words im just hoping for the best xd
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            preds.append(parse_binary_label(response))

        pred_col = f"{model_name}_{quant_label}_classification"
        data[pred_col] = preds

        valid_mask = data["Label"].notna() & data[pred_col].notna()
        if valid_mask.sum() == 0:
            print(f"No valid predictions for {model_name} [{quant_label}]; skipping metrics.")
            run_summary.append(
                {
                    "model": model_name,
                    "status": "no-valid-predictions",
                    "coverage": 0.0,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy_std": None,
                    "precision_std": None,
                    "recall_std": None,
                    "f1_std": None,
                    "valid_rows": 0,
                    "total_rows": len(data),
                }
            )
            del model
            del tokenizer
            torch.cuda.empty_cache()
            print("***************************************")
            continue

        y_true = data.loc[valid_mask, "Label"].astype(int)
        y_pred = data.loc[valid_mask, pred_col].astype(int)
        accuracy, precision, recall, f1 = compute_binary_metrics(y_true, y_pred)
        accuracy_std, precision_std, recall_std, f1_std = compute_bootstrap_std(
            y_true, y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )
        coverage = valid_mask.mean()

        print(f"****** Model: {model_name} [{quant_label}] Results ******")
        print(f"Coverage: {coverage:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(
            f"Std (bootstrap n={BOOTSTRAP_SAMPLES}) -> Acc: {accuracy_std:.4f}, "
            f"Prec: {precision_std:.4f}, Rec: {recall_std:.4f}, F1: {f1_std:.4f}"
        )
        print("***************************************")

        run_summary.append(
            {
                "model": model_name,
                "status": "ok",
                "coverage": coverage,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy_std": accuracy_std,
                "precision_std": precision_std,
                "recall_std": recall_std,
                "f1_std": f1_std,
                "valid_rows": int(valid_mask.sum()),
                "total_rows": len(data),
            }
        )

        del model
        del tokenizer
        torch.cuda.empty_cache()

    output_path = f"data/uselca_with_predictions1_{quant_label}.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

    report_path = f"data/zero_shot_classification_report1_{quant_label}.txt"
    report_lines = [
        "Zero-Shot LCA Classification Benchmark Report",
        "=" * 56,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {DATA_PATH}",
        f"Quantization mode: {quant_label}",
        f"Rows: {len(data)}",
        f"Models attempted: {len(model_names)}",
        f"Sensitivity: bootstrap std over {BOOTSTRAP_SAMPLES} resamples (seed={BOOTSTRAP_SEED})",
        "",
        "Per-model summary",
        "-" * 56,
    ]

    header = (
        f"{'Model':<36} {'Status':<22} {'Coverage':>8} "
        f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Valid/Total':>14}"
    )
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for item in run_summary:
        def fmt_metric(value):
            return f"{value:.4f}" if value is not None else "   N/A  "

        coverage_str = fmt_metric(item["coverage"])
        accuracy_str = fmt_metric(item["accuracy"])
        precision_str = fmt_metric(item["precision"])
        recall_str = fmt_metric(item["recall"])
        f1_str = fmt_metric(item["f1"])
        valid_ratio = f"{item['valid_rows']}/{item['total_rows']}"

        report_lines.append(
            f"{item['model'][:36]:<36} {item['status'][:22]:<22} {coverage_str:>8} "
            f"{accuracy_str:>8} {precision_str:>8} {recall_str:>8} {f1_str:>8} {valid_ratio:>14}"
        )

    report_lines.extend(["", "Metric Std (Sensitivity)", "-" * 56])
    std_header = (
        f"{'Model':<36} {'Acc_std':>10} {'Prec_std':>10} {'Rec_std':>10} {'F1_std':>10}"
    )
    report_lines.append(std_header)
    report_lines.append("-" * len(std_header))

    for item in run_summary:
        def fmt_std(value):
            return f"{value:.4f}" if value is not None else "N/A"

        report_lines.append(
            f"{item['model'][:36]:<36} {fmt_std(item['accuracy_std']):>10} "
            f"{fmt_std(item['precision_std']):>10} {fmt_std(item['recall_std']):>10} {fmt_std(item['f1_std']):>10}"
        )

    report_lines.extend(
        [
            "",
            "Notes",
            "- Coverage is the fraction of rows with both ground-truth label and valid model prediction.",
            "- Std metrics come from bootstrap resampling and quantify sensitivity to dataset perturbation.",
            f"- Predictions are saved in {output_path}.",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved run report to: {report_path}")



