from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pandas as pd
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
	sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from extract import ConverterExtractor


critique_prompt = PromptTemplate.from_template(
"""You are an expert scientific data editor. Your task is to correct formatting errors in a Markdown table extracted via a Docling model from an environmental science research paper. 

Docling often ruins mathematical notation, specifically negative signs, superscripts, subscripts, and scientific notation. 

You will be provided with:
1. RAW CONTEXT: The raw text and context extracted from the PDF near the table.
2. OCR TABLE: The flawed Markdown table.

RAW CONTEXT:
{raw_context}

DOCLING MARKDOWN TABLE:
{ocr_table}

CRITICAL CONSTRAINTS:
1. STRICT GRID INTEGRITY: You ABSOLUTELY MUST NOT add, remove, merge, or split any rows or columns. The output table must have the exact same dimensions and pipe (|) structure as the DOCLING MARKDOWN TABLE.
2. NO CONVERSATION: Output ONLY the corrected Markdown table enclosed in ```markdown ``` blocks. Do not include any greetings, explanations, or concluding remarks.

TRANSFORMATION RULES:
- ZERO EXPONENTS: DOCLING drops superscripts. If DOCLING shows "1.0 x 10 0" or "1.0 10 0", format it as "1.0 x 10^0". NEVER combine it into "1.0 x 100".
- NEGATIVE EXPONENTS: DOCLING drops minus signs and superscripts. If DOCLING shows "1.0 x 10 1" but the RAW CONTEXT shows "1.0 x 10-1" or "10^{{-1}}", you MUST correct it to "1.0 x 10^{{-1}}".
- NEGATIVE NUMBERS: Restore any missing minus signs to negative values based on the context.
- UNITS: Fix units and chemical formulas (e.g., "kg CO2 eq", "m^3", "PO4^3-").

CORRECTED TABLE:"""
)


def infer_input_device(model: torch.nn.Module) -> torch.device:
	try:
		return model.get_input_embeddings().weight.device
	except Exception:
		return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_pdf_text(pdf_path: Path) -> str:
	pages: list[str] = []
	with fitz.open(str(pdf_path)) as doc:
		for page in doc:
			text = page.get_text("text").strip()
			if text:
				pages.append(text)
	return "\n\n".join(pages)


def extract_table_context(pdf_path: Path, table) -> str:
	page_numbers: list[int] = []
	for provenance in getattr(table, "prov", []) or []:
		page_no = getattr(provenance, "page_no", None)
		if page_no is not None:
			page_numbers.append(int(page_no))

	if not page_numbers:
		return extract_pdf_text(pdf_path)

	page_numbers = sorted(set(page_numbers))
	start_page = max(1, page_numbers[0] - 1)
	end_page = page_numbers[-1] + 1

	snippets: list[str] = []
	with fitz.open(str(pdf_path)) as doc:
		for page_no in range(start_page, min(end_page, doc.page_count) + 1):
			page = doc[page_no - 1]
			text = page.get_text("text").strip()
			if text:
				snippets.append(f"[PAGE {page_no}]\n{text}")

	return "\n\n".join(snippets)


def build_table_markdown(table, doc) -> str:
	table_df: pd.DataFrame = table.export_to_dataframe(doc=doc)
	return table_df.to_markdown(index=False)


def load_critic(
	model_name: str,
	cache_dir: Optional[str] = None,
	device_map: str | dict | None = "auto",
):
	load_kwargs = {
		"cache_dir": cache_dir,
		"low_cpu_mem_usage": True,
	}
	if torch.cuda.is_available():
		load_kwargs["device_map"] = device_map
		load_kwargs["torch_dtype"] = torch.float16

	model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	text_generation = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		max_new_tokens=4096,
		do_sample=False,
		return_full_text=False,
		temperature=0.1,
	)
	return HuggingFacePipeline(pipeline=text_generation)


def resolve_device_map(gpu_index: Optional[int]):
	if gpu_index is None:
		return "auto"

	if not torch.cuda.is_available():
		raise ValueError("--gpu-index was provided but CUDA is not available on this machine.")

	if gpu_index not in (0, 1):
		raise ValueError("--gpu-index must be 0 or 1.")

	if gpu_index >= torch.cuda.device_count():
		raise ValueError(
			f"Requested GPU index {gpu_index}, but only {torch.cuda.device_count()} CUDA device(s) are available."
		)

	# Force every model module onto a single GPU when requested.
	return {"": f"cuda:{gpu_index}"}


def critique_table(
	pdf_path: Path,
	table_index: int = 0,
	model_name: str = "Qwen/Qwen2.5-7B-Instruct",
	cache_dir: Optional[str] = None,
	gpu_index: Optional[int] = None,
) -> str:
	converter = ConverterExtractor()
	doc = converter.convert(str(pdf_path))

	if not getattr(doc, "tables", None):
		raise ValueError(f"No tables found in PDF: {pdf_path}")
	if table_index < 0 or table_index >= len(doc.tables):
		raise IndexError(f"Table index {table_index} out of range for {len(doc.tables)} tables")

	table = doc.tables[table_index]
	raw_context = extract_table_context(pdf_path, table)
	ocr_table = build_table_markdown(table, doc)
	# save full prompt to txt file for debugging
	with open("debug_prompt.txt", "w", encoding="utf-8") as f:
		f.write(critique_prompt.format(raw_context=raw_context, ocr_table=ocr_table))

	critic = load_critic(
		model_name=model_name,
		cache_dir=cache_dir,
		device_map=resolve_device_map(gpu_index),
	)
	# save full prompt with context and extracted table for debugging
    
	chain = critique_prompt | critic | StrOutputParser()
	return chain.invoke({"raw_context": raw_context, "ocr_table": ocr_table})


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Critique a Docling-extracted markdown table using nearby PDF text."
	)
	parser.add_argument("--pdf-path", type=Path, default="/dkucc/home/nt140/LLM-LCI/criteria_classification/data/papers/Renoufetal2008.pdf", help="Path to the source PDF")
	parser.add_argument("--table-index", type=int, default=0, help="Zero-based table index to critique")
	parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model name")
	parser.add_argument("--cache-dir", default='/work/lamlab/huggingface/models', help="Optional Hugging Face cache directory")
	parser.add_argument(
		"--gpu-index",
		type=int,
		choices=[0, 1],
		default=None,
		help="Force model loading on a single GPU (0 or 1). If omitted, uses Transformers auto device_map.",
	)
	args = parser.parse_args()

	output = critique_table(
		pdf_path=args.pdf_path,
		table_index=args.table_index,
		model_name=args.model,
		cache_dir=args.cache_dir,
		gpu_index=args.gpu_index,
	)
	print(output)


if __name__ == "__main__":
	main()

