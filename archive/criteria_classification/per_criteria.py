"""
code to produce per criteria classification results and saves per class produced labels
"""

import json
import torch
import re
import pandas as pd
import transformers
from tqdm import tqdm
from truncate import clean_and_truncate_pdf_text
from system_prompts import SYSTEM_PROMPTS

CORPUS_PATH = "data/corpus.json"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
MAX_INPUT_TOKENS = 30000
OUTPUT_CSV_PATH = "data/per_criteria_labels.csv"
THINKING = False # whether to activate thinking on both qwen3.5 and gemma4

def parse_yn_label(text):
    # Strip the thinking block completely
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    match = re.search(r"\b([YNyn])\b", text)
    if not match:
        return None
    return 1 if match.group(1).upper() == "Y" else 0


def label_int_to_str(label):
    if label == 1:
        return "Y"
    if label == 0:
        return "N"
    return ""


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)

def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    records = [r for r in corpus.get("records", []) if r.get("match_status") == "MATCHED"]
    if not records:
        print("No MATCHED records found in corpus.json")
        return

    model_names = [
        "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen3.5-9B", 
        # "google/gemma-4-26B-A4B-it"
    ]

    all_model_rows = []

    for model_name in model_names:
        print(f"\nLoading {model_name}...")
        load_kwargs = {"cache_dir": MODEL_CACHE_DIR, "torch_dtype": torch.bfloat16}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        if "gemma-4" in model_name:
            tokenizer = transformers.AutoProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            pad_token_id = tokenizer.tokenizer.eos_token_id
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.pad_token_id

        truncation_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        model_rows = []

        for record in tqdm(records, desc=f"Evaluating {model_name}"):
            raw_text = record.get("full_text")
            raw_text = clean_and_truncate_pdf_text(raw_text, truncation_tokenizer, MAX_INPUT_TOKENS)

            row = {
                "model": model_name,
                "title": record.get("gt_title", ""),
                "author": record.get("gt_author", ""),
                "year": record.get("gt_year", ""),
            }

            for idx, prompt in enumerate(SYSTEM_PROMPTS, start=1):
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Title: {record['gt_title']}\n\nFull Text Extract:\n{raw_text}"}
                ]

                try:
                    kwargs = {"tokenize": False, "add_generation_prompt": True}
                    if "gemma-4" in model_name or "Qwen3.5" in model_name:
                        kwargs["enable_thinking"] = False

                    input_text = tokenizer.apply_chat_template(messages, **kwargs)
                    if "gemma-4" in model_name or "Qwen3.5" in model_name:
                        inputs = tokenizer(text=input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
                    else:
                        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)

                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False, pad_token_id=pad_token_id)

                    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    extracted_label = parse_yn_label(response)
                    row[f"criteria_{idx}"] = label_int_to_str(extracted_label)

                except Exception as e:
                    print(
                        "Error processing "
                        f"{record.get('pdf_filename', '')} / criteria_{idx} with {model_name}: {e}"
                    )
                    row[f"criteria_{idx}"] = ""

            model_rows.append(row)
            all_model_rows.append(row)

        model_output_path = f"data/per_criteria_labels_{sanitize_model_name(model_name)}.csv"
        pd.DataFrame(model_rows).to_csv(model_output_path, index=False)
        print(f"Saved per-criteria labels to {model_output_path}")

        del model, tokenizer
        torch.cuda.empty_cache()

    pd.DataFrame(all_model_rows).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved combined per-criteria labels to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()