import re
import random
import statistics
from datetime import datetime

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DATA_PATH = "data/uselca1.csv"
FT_MODEL_PATH = "/work/lamlab/nt140/qwen_lca_full/final_model"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
VAL_SIZE = 0.1
RANDOM_SEED = 42
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
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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


def get_heldout_split() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required_cols = ["Article Title", "Abstract", "Label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Match finetune_q7b.py preprocessing and split parameters exactly.
    df = df.dropna(subset=["Label"]).copy()
    df["Article Title"] = df["Article Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Label"] = df["Label"].astype(int)

    _, heldout_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["Label"],
    )
    return heldout_df.reset_index(drop=True)


def evaluate_config(base_data, config):
    config_name = config["name"]
    model_ref = config["model_ref"]
    quant_kwargs = config["quant_kwargs"]

    print(f"\n===== Running config: {config_name} =====")
    data = base_data.copy()

    load_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "low_cpu_mem_usage": True,
        **quant_kwargs,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    except ImportError as err:
        if "accelerate" in str(err).lower() and "device_map" in load_kwargs:
            print("Accelerate is missing; retrying model load on default device.")
            load_kwargs.pop("device_map", None)
            load_kwargs.pop("low_cpu_mem_usage", None)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
        else:
            raise RuntimeError(f"Could not load model in {config_name}: {err}") from err
    except OSError as err:
        raise RuntimeError(f"Could not load model in {config_name}: {err}") from err

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        preds.append(parse_binary_label(response))

    pred_col = f"{config_name}_classification"
    data[pred_col] = preds

    valid_mask = data["Label"].notna() & data[pred_col].notna()
    if valid_mask.sum() == 0:
        raise RuntimeError(f"No valid predictions in config {config_name}.")

    y_true = data.loc[valid_mask, "Label"].astype(int)
    y_pred = data.loc[valid_mask, pred_col].astype(int)

    accuracy, precision, recall, f1 = compute_binary_metrics(y_true, y_pred)
    accuracy_std, precision_std, recall_std, f1_std = compute_bootstrap_std(
        y_true, y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    coverage = valid_mask.mean()

    print(f"****** Results [{config_name}] ******")
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

    del model
    del tokenizer
    torch.cuda.empty_cache()

    summary = {
        "name": config_name,
        "model_ref": model_ref,
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

    return data[[pred_col]], summary


def write_report(report_path, run_summaries, heldout_rows):
    report_lines = [
        "Qwen2.5-7B Held-Out Classification Report",
        "=" * 56,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {DATA_PATH}",
        f"Held-out split: val ({int(VAL_SIZE * 100)}%) recreated from finetune settings",
        f"Split seed: {RANDOM_SEED}",
        f"Rows (held-out): {heldout_rows}",
        f"Sensitivity: bootstrap std over {BOOTSTRAP_SAMPLES} resamples (seed={BOOTSTRAP_SEED})",
        "",
        "Per-config Summary",
        "-" * 56,
        "",
    ]

    header = (
        f"{'Config':<20} {'Status':<10} {'Coverage':>8} {'Acc':>8} "
        f"{'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc_std':>8} {'F1_std':>8}"
    )
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for item in run_summaries:
        if item["status"] != "ok":
            report_lines.append(
                f"{item['name'][:20]:<20} {item['status'][:10]:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
            )
            continue

        report_lines.append(
            f"{item['name'][:20]:<20} {item['status'][:10]:<10} {item['coverage']:>8.4f} "
            f"{item['accuracy']:>8.4f} {item['precision']:>8.4f} {item['recall']:>8.4f} "
            f"{item['f1']:>8.4f} {item['accuracy_std']:>8.4f} {item['f1_std']:>8.4f}"
        )

    report_lines.extend(
        [
            "",
            "Notes",
            "- Coverage is the fraction of rows with both ground-truth label and valid model prediction.",
            "- Std metrics come from bootstrap resampling and quantify sensitivity to dataset perturbation.",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    heldout_data = get_heldout_split()

    model_configs = [
        {
            "name": "ft_no_quant",
            "model_ref": FT_MODEL_PATH,
            "quant_kwargs": {"torch_dtype": torch.bfloat16},
        },
        {
            "name": "ft_8bit",
            "model_ref": FT_MODEL_PATH,
            "quant_kwargs": {"load_in_8bit": True},
        },
        {
            "name": "ft_16bit",
            "model_ref": FT_MODEL_PATH,
            "quant_kwargs": {"torch_dtype": torch.float16},
        },
        {
            "name": "base_no_quant",
            "model_ref": BASE_MODEL_NAME,
            "quant_kwargs": {"torch_dtype": torch.bfloat16},
        },
    ]

    prediction_cols = []
    run_summaries = []

    for config in model_configs:
        try:
            preds_df, summary = evaluate_config(heldout_data, config)
            heldout_data = pd.concat([heldout_data, preds_df], axis=1)
            prediction_cols.extend(preds_df.columns.tolist())
            run_summaries.append(summary)
        except Exception as err:
            print(f"Config failed: {config['name']} -> {err}")
            run_summaries.append(
                {
                    "name": config["name"],
                    "model_ref": config["model_ref"],
                    "status": "failed",
                }
            )

    output_path = "data/q7b_ft_heldout_predictions.csv"
    base_cols = ["Article Title", "Abstract", "Label"]
    kept_cols = [c for c in base_cols + prediction_cols if c in heldout_data.columns]
    heldout_data[kept_cols].to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

    report_path = "data/q7b_ft_heldout_report.txt"
    write_report(report_path, run_summaries, heldout_rows=len(heldout_data))
    print(f"Saved run report to: {report_path}")
