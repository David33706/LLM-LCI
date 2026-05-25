import re
import random
import statistics
import os
from datetime import datetime

import pandas as pd
import torch
import transformers
from tqdm import tqdm

print("Starting Qwen 7B Few-Shot Classification Script")

DATA_PATH = "data/uselca1.csv"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
RANDOM_SEED = 42
FEWSHOT_PER_CLASS = 4
FEWSHOT_SEEDS = [42, 52, 62]
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


def load_data():
    df = pd.read_csv(DATA_PATH)
    required_cols = ["Article Title", "Abstract", "Label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=["Label"]).copy()
    df["Article Title"] = df["Article Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Label"] = df["Label"].astype(int)
    return df.reset_index(drop=True)


def choose_fewshot_examples(val_df, per_class=4, seed=42):
    chosen_parts = []
    for label in [0, 1]:
        label_df = val_df[val_df["Label"] == label]
        if len(label_df) < per_class:
            raise ValueError(
                f"Not enough samples for label={label}. Needed {per_class}, found {len(label_df)}"
            )
        chosen_parts.append(label_df.sample(n=per_class, random_state=seed))

    fewshot_df = pd.concat(chosen_parts).sample(frac=1.0, random_state=seed).copy()
    eval_df = val_df.drop(index=fewshot_df.index).reset_index(drop=True)
    fewshot_df = fewshot_df.reset_index(drop=True)
    return fewshot_df, eval_df


def build_input_text(tokenizer, title, abstract, fewshot_df):
    fewshot_lines = []
    for idx, row in fewshot_df.iterrows():
        fewshot_lines.append(
            f"Example {idx + 1}\n"
            f"Title: {row['Article Title']}\n"
            f"Abstract: {row['Abstract']}\n"
            f"Classification: {int(row['Label'])}"
        )

    fewshot_block = "\n\n".join(fewshot_lines)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here are labeled examples:\n\n"
                f"{fewshot_block}\n\n"
                "Now classify the following article.\n"
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
        "Here are labeled examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following article.\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        "Classification (output only 1 or 0):"
    )


def evaluate_mode(mode_name, quant_kwargs, eval_df, fewshot_df, seed):
    print(f"\n===== Running mode: {mode_name} | few-shot seed: {seed} =====")

    load_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "low_cpu_mem_usage": True,
        **quant_kwargs,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_device = infer_input_device(model)
    if hasattr(model, "hf_device_map"):
        devices = sorted({str(v) for v in model.hf_device_map.values()})
        print(f"Device map: {devices}")
    else:
        print(f"Model device: {model.device}")

    preds = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        input_text = build_input_text(
            tokenizer,
            title=row["Article Title"],
            abstract=row["Abstract"],
            fewshot_df=fewshot_df,
        )
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

    pred_col = f"seed{seed}_{mode_name}_classification"
    result_df = eval_df.copy()
    result_df[pred_col] = preds

    valid_mask = result_df["Label"].notna() & result_df[pred_col].notna()
    if valid_mask.sum() == 0:
        raise RuntimeError(f"No valid predictions for mode {mode_name}.")

    y_true = result_df.loc[valid_mask, "Label"].astype(int)
    y_pred = result_df.loc[valid_mask, pred_col].astype(int)

    accuracy, precision, recall, f1 = compute_binary_metrics(y_true, y_pred)
    accuracy_std, precision_std, recall_std, f1_std = compute_bootstrap_std(
        y_true, y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )

    summary = {
        "seed": seed,
        "mode": mode_name,
        "status": "ok",
        "coverage": valid_mask.mean(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_std": accuracy_std,
        "precision_std": precision_std,
        "recall_std": recall_std,
        "f1_std": f1_std,
        "valid_rows": int(valid_mask.sum()),
        "total_rows": len(result_df),
    }

    print(f"****** Qwen 7B Few-shot Results [{mode_name} | seed {seed}] ******")
    print(f"Coverage: {summary['coverage']:.4f}")
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

    return result_df[[pred_col]], summary


def aggregate_by_mode(run_summaries):
    grouped = {}
    for item in run_summaries:
        if item.get("status") != "ok":
            continue
        grouped.setdefault(item["mode"], []).append(item)

    aggregate_rows = []
    for mode, rows in grouped.items():
        def calc(metric):
            vals = [r[metric] for r in rows]
            return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

        acc_mean, acc_sd = calc("accuracy")
        f1_mean, f1_sd = calc("f1")
        aggregate_rows.append(
            {
                "mode": mode,
                "runs": len(rows),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_sd,
                "f1_mean": f1_mean,
                "f1_std": f1_sd,
            }
        )
    return aggregate_rows


def write_report(report_path, run_summaries, aggregate_rows, eval_rows_by_seed):
    report_lines = [
        "Qwen2.5-7B Few-Shot Classification Report",
        "=" * 56,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {DATA_PATH}",
        f"Model: {MODEL_NAME}",
        "Evaluation source: full dataset",
        f"Split seed: {RANDOM_SEED}",
        f"Few-shot seeds: {FEWSHOT_SEEDS}",
        f"Few-shot examples per class: {FEWSHOT_PER_CLASS}",
        f"Few-shot total examples per run: {2 * FEWSHOT_PER_CLASS}",
        f"Evaluation rows by seed after removing few-shot examples: {eval_rows_by_seed}",
        f"Sensitivity: bootstrap std over {BOOTSTRAP_SAMPLES} resamples (seed={BOOTSTRAP_SEED})",
        "",
        "Per-run Summary",
        "-" * 56,
    ]

    header = (
        f"{'Seed':<6} {'Mode':<12} {'Status':<10} {'Coverage':>8} {'Acc':>8} {'Prec':>8} "
        f"{'Rec':>8} {'F1':>8} {'Acc_std':>8} {'F1_std':>8}"
    )
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for item in run_summaries:
        if item["status"] != "ok":
            report_lines.append(
                f"{str(item.get('seed', 'NA')):<6} {item['mode'][:12]:<12} {item['status'][:10]:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
            )
            continue

        report_lines.append(
            f"{str(item['seed']):<6} {item['mode'][:12]:<12} {item['status'][:10]:<10} {item['coverage']:>8.4f} "
            f"{item['accuracy']:>8.4f} {item['precision']:>8.4f} {item['recall']:>8.4f} "
            f"{item['f1']:>8.4f} {item['accuracy_std']:>8.4f} {item['f1_std']:>8.4f}"
        )

    report_lines.extend(["", "Aggregate Summary By Mode", "-" * 56])
    agg_header = f"{'Mode':<12} {'Runs':>5} {'Acc_mean':>10} {'Acc_std':>10} {'F1_mean':>10} {'F1_std':>10}"
    report_lines.append(agg_header)
    report_lines.append("-" * len(agg_header))

    for row in aggregate_rows:
        report_lines.append(
            f"{row['mode'][:12]:<12} {row['runs']:>5} {row['accuracy_mean']:>10.4f} "
            f"{row['accuracy_std']:>10.4f} {row['f1_mean']:>10.4f} {row['f1_std']:>10.4f}"
        )

    report_lines.extend(
        [
            "",
            "Notes",
            "- Few-shot examples were removed from evaluation rows for each seed run.",
            "- Per-seed few-shot examples are saved to separate CSV files in the data folder.",
            "- Coverage is the fraction of rows with both ground-truth label and valid prediction.",
            "- Std metrics come from bootstrap resampling and quantify sensitivity.",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    full_df = load_data()

    mode_configs = [
        ("no_quant", {"torch_dtype": torch.bfloat16}),
    ]

    os.makedirs("data", exist_ok=True)

    all_run_summaries = []
    eval_rows_by_seed = {}

    for seed in FEWSHOT_SEEDS:
        print(f"\n########## Few-shot seed: {seed} ##########")
        fewshot_df, eval_df = choose_fewshot_examples(
            full_df, per_class=FEWSHOT_PER_CLASS, seed=seed
        )
        eval_rows_by_seed[seed] = len(eval_df)

        fewshot_examples_path = f"data/qwen7b_fewshot_examples_seed{seed}.csv"
        fewshot_df.to_csv(fewshot_examples_path, index=False)
        print(f"Saved few-shot examples to: {fewshot_examples_path}")

        seed_preds_df = eval_df.copy()
        for mode_name, quant_kwargs in mode_configs:
            mode_summary = {"seed": seed, "mode": mode_name, "status": "failed"}
            try:
                mode_pred_df, mode_summary = evaluate_mode(
                    mode_name, quant_kwargs, eval_df, fewshot_df, seed
                )
            except Exception as err:
                print(f"Mode failed: {mode_name} (seed={seed}) -> {err}")
                all_run_summaries.append(mode_summary)
                continue

            seed_preds_df = pd.concat([seed_preds_df, mode_pred_df], axis=1)
            all_run_summaries.append(mode_summary)

        seed_predictions_path = f"data/qwen7b_fewshot_predictions_seed{seed}.csv"
        seed_preds_df.to_csv(seed_predictions_path, index=False)
        print(f"Saved predictions to: {seed_predictions_path}")

    runs_table_path = "data/qwen7b_fewshot_runs.csv"
    pd.DataFrame(all_run_summaries).to_csv(runs_table_path, index=False)
    print(f"Saved per-run summary table to: {runs_table_path}")

    aggregate_rows = aggregate_by_mode(all_run_summaries)
    report_path = "data/qwen7b_fewshot_report.txt"
    write_report(report_path, all_run_summaries, aggregate_rows, eval_rows_by_seed)
    print(f"Saved run report to: {report_path}")
