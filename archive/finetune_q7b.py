"""
finetune_qwen_lca_full.py
--------------------------
Full fine-tuning of Qwen2.5-7B-Instruct for binary LCA classification.
Designed for a single H20 GPU (143 GB VRAM).

After training, the saved model can be post-training quantized — see
the quantize_model() function at the bottom of this file.

Requirements:
    pip install transformers trl accelerate datasets scikit-learn
    pip install optimum auto-gptq  # only needed for GPTQ quantization
"""

import os
import re
import random
import statistics
import torch
import pandas as pd
from datetime import datetime
from typing import Optional
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
# from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH       = "data/uselca1.csv"
MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
OUTPUT_DIR      = "/work/lamlab/nt140/qwen_lca_full"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

VAL_SIZE        = 0.2
RANDOM_SEED     = 42
BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_SEED = 42

# Training — larger batches are now viable with 143 GB
MAX_SEQ_LENGTH      = 1024
NUM_EPOCHS          = 5
BATCH_SIZE          = 8   # increase until VRAM is ~85% utilized
GRAD_ACCUM_STEPS    = 1    # effective batch = 16; no need to accumulate
LEARNING_RATE       = 1e-5  # lower than QLoRA — full weights are more sensitive
WARMUP_RATIO        = 0.1
WEIGHT_DECAY        = 0.01
EARLY_STOP_PATIENCE = 3
MAX_GRAD_NORM       = 1.0

# Reuse the exact same system prompt as the benchmark for consistency
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


# ---------------------------------------------------------------------------
# Prompt builder — identical to benchmark and QLoRA scripts
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, title: str, abstract: str, label: Optional[int] = None) -> str:
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
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if label is not None:
        prompt += f"{label}"
    return prompt


def build_input_text(tokenizer, title: str, abstract: str) -> str:
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


def parse_binary_label(text: str):
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


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_datasets(tokenizer) -> tuple[Dataset, Dataset, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)

    required = ["Article Title", "Abstract", "Label"]
    missing =[c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["Label"])
    df["Article Title"] = df["Article Title"].fillna("").astype(str)
    df["Abstract"]      = df["Abstract"].fillna("").astype(str)
    df["Label"]         = df["Label"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["Label"],
    )

    def to_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        records = []
        for _, row in split_df.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Title: {row['Article Title']}\n"
                        f"Abstract: {row['Abstract']}\n"
                        "Classification (output only 1 or 0):"
                    ),
                },
            ]
            records.append({
                "prompt": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                "completion": str(int(row["Label"])),
            })
        return Dataset.from_list(records)

    return to_hf_dataset(train_df), to_hf_dataset(val_df), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model + tokenizer loading — no quantization, full bf16 weights
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Gradient checkpointing trades compute for memory:
    # recomputes activations on the backward pass instead of storing them.
    # Still worth enabling — activations from large batches add up fast.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    return model, tokenizer


def supports_assistant_mask(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "{% generation %}" in template


def evaluate_model_config(base_data: pd.DataFrame, config: dict):
    config_name = config["name"]
    model_ref = config["model_ref"]
    quant_kwargs = config["quant_kwargs"]

    print(f"\n===== Running held-out evaluation: {config_name} =====")
    data = base_data.copy()

    load_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "low_cpu_mem_usage": True,
        **quant_kwargs,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_device = infer_input_device(model)
    if hasattr(model, "hf_device_map"):
        devices = sorted({str(v) for v in model.hf_device_map.values()})
        print(f"Device map: {devices}")
    else:
        print(f"Model device: {model.device}")

    preds = []
    for _, row in val_data_iter(data):
        title = row["Article Title"]
        abstract = row["Abstract"]
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
    y_true = data.loc[valid_mask, "Label"].astype(int)
    y_pred = data.loc[valid_mask, pred_col].astype(int)

    accuracy, precision, recall, f1 = compute_binary_metrics(y_true, y_pred)
    accuracy_std, precision_std, recall_std, f1_std = compute_bootstrap_std(
        y_true, y_pred, n_bootstrap=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )

    summary = {
        "name": config_name,
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
    }

    print(f"****** Held-out Results [{config_name}] ******")
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

    return data[[pred_col]], summary


def val_data_iter(df: pd.DataFrame):
    for idx, row in df.iterrows():
        yield idx, row


def run_post_training_evaluation(final_model_path: str, val_df: pd.DataFrame):
    model_configs = [
        {
            "name": "ft_no_quant",
            "model_ref": final_model_path,
            "quant_kwargs": {"torch_dtype": torch.bfloat16},
        },
        {
            "name": "ft_8bit",
            "model_ref": final_model_path,
            "quant_kwargs": {"load_in_8bit": True},
        },
        {
            "name": "ft_16bit",
            "model_ref": final_model_path,
            "quant_kwargs": {"torch_dtype": torch.float16},
        },
        {
            "name": "base_no_quant",
            "model_ref": BASE_MODEL_NAME,
            "quant_kwargs": {"torch_dtype": torch.bfloat16},
        },
    ]

    eval_df = val_df.copy()
    run_summaries = []
    pred_cols = []

    for config in model_configs:
        try:
            pred_df, summary = evaluate_model_config(eval_df, config)
            eval_df = pd.concat([eval_df, pred_df], axis=1)
            pred_cols.extend(pred_df.columns.tolist())
            run_summaries.append(summary)
        except Exception as err:
            print(f"Config failed: {config['name']} -> {err}")
            run_summaries.append({"name": config["name"], "status": "failed"})

    pred_path = os.path.join(OUTPUT_DIR, "heldout_predictions.csv")
    base_cols = ["Article Title", "Abstract", "Label"]
    keep_cols = [c for c in base_cols + pred_cols if c in eval_df.columns]
    eval_df[keep_cols].to_csv(pred_path, index=False)
    print(f"Held-out predictions saved to: {pred_path}")

    report_lines = [
        "Qwen2.5-7B Post-Training Held-Out Report",
        "=" * 56,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Dataset: {DATA_PATH}",
        f"Held-out source: validation split from current run (VAL_SIZE={VAL_SIZE}, seed={RANDOM_SEED})",
        f"Rows (held-out): {len(eval_df)}",
        f"Sensitivity: bootstrap std over {BOOTSTRAP_SAMPLES} resamples (seed={BOOTSTRAP_SEED})",
        "",
        "Per-config Summary",
        "-" * 56,
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

    report_path = os.path.join(OUTPUT_DIR, "heldout_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Held-out report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Preparing datasets...")
    train_dataset, val_dataset, val_df = prepare_datasets(tokenizer)

    # assistant_only_loss_enabled = supports_assistant_mask(tokenizer)
    # if not assistant_only_loss_enabled:
    #     print(
    #         "[Warning] Tokenizer chat_template does not support assistant masks "
    #         "(`{% generation %}` not found). Falling back to assistant_only_loss=False."
    #     )

    # Mask prompt tokens — compute loss only on the "0" or "1" answer
    # response_template = tokenizer.encode(
    #     "<|im_start|>assistant\n", add_special_tokens=False
    # )
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    # )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        # dataset_text_field="text",
        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Save full model, not just adapter
        save_total_limit=2,         # keep only the 2 best checkpoints
        # Logging
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        report_to="none",
        # Reproducibility
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        completion_only_loss=True,
        max_grad_norm=MAX_GRAD_NORM,
        save_only_model=True,  # save only the model weights, not the optimizer or scheduler states just because of space constraints
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting full fine-tuning...")
    trainer.train()

    # Save the full fine-tuned model in bf16
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Full model saved to: {final_model_path}")

    # Training summary
    log_history = trainer.state.log_history
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model:         {MODEL_NAME}\n")
        f.write(f"Fine-tune:     full (no LoRA)\n")
        f.write(f"Saved to:      {final_model_path}\n")
        f.write(f"Train rows:    {len(train_dataset)}\n")
        f.write(f"Val rows:      {len(val_dataset)}\n")
        f.write(f"Batch size:    {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Epochs run:    {int(trainer.state.epoch)}\n")
        f.write("\nLog history (last 20 entries):\n")
        for entry in log_history[-20:]:
            f.write(f"  {entry}\n")
    print(f"Training summary saved to: {summary_path}")

    run_post_training_evaluation(final_model_path, val_df)
    return final_model_path


# ---------------------------------------------------------------------------
# Post-training quantization
# ---------------------------------------------------------------------------

def quantize_model(model_path: str):
    """
    Quantize the full fine-tuned model to 4-bit NF4 (bitsandbytes) for
    memory-efficient inference. Run this after train() completes.

    The quantized model is inference-only — do not use it for further training.
    """
    from transformers import BitsAndBytesConfig

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Quantizing model at: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    quant_path = model_path + "_4bit"
    # bitsandbytes quantized models cannot be saved with save_pretrained in the
    # traditional sense — serialize the config and tokenizer, and document that
    # the model must be re-loaded with the same BitsAndBytesConfig at inference.
    tokenizer.save_pretrained(quant_path)
    model.config.save_pretrained(quant_path)

    # Save the bnb config separately so inference code can reconstruct it
    import json
    bnb_params = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    with open(os.path.join(quant_path, "bnb_config.json"), "w") as f:
        json.dump(bnb_params, f, indent=2)

    print(f"Quantization config saved to: {quant_path}")
    print("At inference, reload with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{model_path}', quantization_config=bnb_config)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    final_model_path = train()

    # Optional: quantize immediately after training
    # quantize_model(final_model_path)