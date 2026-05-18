import json
import torch
import transformers
import pandas as pd
from tqdm import tqdm
import re
from truncate import clean_and_truncate_pdf_text

CORPUS_PATH = "data/corpus.json"
MODEL_CACHE_DIR = "/work/lamlab/huggingface/models"
MAX_INPUT_TOKENS = 30000
SMOKE_TEST_PATH = "data/full_text_smoke_test.txt"
DEBUG_OUTPUTS_CSV_PATH = "data/full_text_model_outputs_debug.csv"

SYSTEM_PROMPT = """
You are a senior environmental science researcher assessing literature for a meta-analysis.
Determine if the provided academic paper meets ANY of the following relevance criteria:
- Assessing the environmental impacts of a resource recovery process, or
- Comparing the environmental impacts of different resource recovery processes, or
- Comparing the environmental impacts before and after introducing resource recovery, or
- Assessing the environmental impacts of using recovered products, or
- Assessing the environmental impacts of resource recovery with wastewater source separation, or

Output exactly ONLY one letter, no reasoning only if specifically asked, keep reasoning short:
- Y = Yes, it meets at least one criterion
- N = No, it does not meet any criteria
""".strip()

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

def compute_metrics(y_true, y_pred):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        return 0, 0, 0, 0, 0
    
    y_t = torch.tensor([x[0] for x in valid])
    y_p = torch.tensor([x[1] for x in valid])
    
    tp = ((y_t == 1) & (y_p == 1)).sum().item()
    tn = ((y_t == 0) & (y_p == 0)).sum().item()
    fp = ((y_t == 0) & (y_p == 1)).sum().item()
    fn = ((y_t == 1) & (y_p == 0)).sum().item()
    
    acc = (tp + tn) / len(valid)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    
    return acc, prec, rec, f1, len(valid)

def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    records = [r for r in corpus.get("records", []) if r["match_status"] == "MATCHED"]
    
    model_names = [
        "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen3.5-9B", 
        # "google/gemma-4-26B-A4B-it"
    ]
    
    results = []
    debug_rows = []
    smoke_test_written = False

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

        y_true, y_pred = [], []
        truncation_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        
        for record in tqdm(records, desc=f"Evaluating {model_name}"):
            gt_label = str(record.get("gt_label", "")).strip().upper()
            gt_val = 1 if gt_label == "Y" else 0
            y_true.append(gt_val)
            
            # Truncate raw text to prevent OOM or severe attention degradation
            # Note: A proper token-based truncation via tokenizer is safer than string slicing
            raw_text = record.get("full_text")
            raw_text = clean_and_truncate_pdf_text(raw_text, truncation_tokenizer, MAX_INPUT_TOKENS)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {record['gt_title']}\n\nFull Text Extract:\n{raw_text}"}
            ]

            try:
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "gemma-4" in model_name or "Qwen3.5" in model_name:
                    kwargs["enable_thinking"] = True
                
                input_text = tokenizer.apply_chat_template(messages, **kwargs)
                if "gemma-4" in model_name or "Qwen3.5" in model_name:
                    inputs = tokenizer(text=input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
                else:
                    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                
                generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                extracted_label = parse_yn_label(response)
                y_pred.append(extracted_label)

                debug_rows.append(
                    {
                        "model": model_name,
                        "paper_title": record.get("gt_title", ""),
                        "ground_truth": gt_label,
                        "complete_model_output": response,
                        "extracted_label": label_int_to_str(extracted_label),
                    }
                )

                if not smoke_test_written:
                    with open(SMOKE_TEST_PATH, "w", encoding="utf-8") as smoke_file:
                        smoke_file.write(f"Model: {model_name}\n")
                        smoke_file.write(f"PDF: {record.get('pdf_filename', '')}\n")
                        smoke_file.write(f"Title: {record.get('gt_title', '')}\n\n")
                        smoke_file.write("Prompt:\n")
                        smoke_file.write(f"{input_text}\n\n")
                        smoke_file.write("Model output:\n")
                        smoke_file.write(f"{response}\n")
                    smoke_test_written = True
                
            except Exception as e:
                print(f"Error processing {record['pdf_filename']}: {e}")
                y_pred.append(None)
                debug_rows.append(
                    {
                        "model": model_name,
                        "paper_title": record.get("gt_title", ""),
                        "ground_truth": gt_label,
                        "complete_model_output": f"<ERROR> {e}",
                        "extracted_label": "",
                    }
                )

        acc, prec, rec, f1, valid_count = compute_metrics(y_true, y_pred)
        print(f"\n{model_name} Results:")
        print(f"Valid: {valid_count}/{len(records)} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")
        
        results.append({"model": model_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1})
        
        del model, tokenizer
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv("data/full_text_classification_summary.csv", index=False)
    pd.DataFrame(debug_rows).to_csv(DEBUG_OUTPUTS_CSV_PATH, index=False)
    print("\nBenchmark complete. Saved to data/full_text_classification_summary.csv")
    print(f"Debug outputs saved to {DEBUG_OUTPUTS_CSV_PATH}")
    if smoke_test_written:
        print(f"Smoke test saved to {SMOKE_TEST_PATH}")

if __name__ == "__main__":
    main()