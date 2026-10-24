#!/usr/bin/env python3
"""
reader_probe.py — does the reader do its core job?

Validate ONE thing in isolation: given the CORRECT paper, can the 32B reader
locate the assumption a question asks about and quote it with its section?
No retrieval, no tables — just paper in, located+quoted answer out. This is the
smallest test that proves the full-paper-feed idea before wiring anything else.

Run it in venv-llmlci, from the repo root, on the same GPU node as the server
(so localhost:3000 is reachable).
"""

import argparse
import json
import time
from pathlib import Path

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import requests

# Reuse the pipeline's own extraction. It already excludes references —
# it only pulls title, abstract, sections, captions, never ce:bibliography.
from utils.elsevier import extract_plain_text


SYSTEM_PROMPT = """You are a precise research librarian. You are given the full text of ONE scientific life-cycle-assessment (LCA) paper and a question about an assumption or methodological choice made in that paper.

Your job is ONLY to locate and report — never to compute or infer:
- Find where in THIS paper the question is answered.
- Quote the relevant text verbatim: copy it exactly, including any numbers and units as written.
- Do NOT calculate, convert, average, or infer any value. Only report what the text states.
- If the paper does not contain the answer, set "relevant" to false.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "relevant": true,
  "location": "the section title or table where the answer appears, e.g. '2.2.1.2 System boundary' or 'Table 3'",
  "assumption": "a short statement of the assumption or choice, in your own words",
  "supporting_quote": "the exact sentence(s) from the paper that contain the answer, copied verbatim"
}
If "relevant" is false, use empty strings for the other three fields."""


def doi_to_filename(doi: str) -> str:
    # Same rule as utils.elsevier.save_xml_to_file: / \ : -> _ , dots kept.
    clean = doi.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{clean}.xml"


def load_question(eval_path, qid):
    from step3_orchestrate import load_questions   # same .csv/.xlsx loader as step3
    for q in load_questions(eval_path):
        if str(q.get("id")) == str(qid):
            return q
    raise SystemExit(f"id {qid} not found in {eval_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, default=8)
    ap.add_argument("--eval-set", default="eval/LCA_RAG_eval_set_v3_50.xlsx")
    ap.add_argument("--xml-dir", default="elsevier_xml")
    ap.add_argument("--url", default="http://localhost:3000/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.5-35b-a3b-fp8")   # adopted reader (as step3)
    args = ap.parse_args()

    q = load_question(args.eval_set, args.qid)
    doi = str(q["source_doi"]).strip()

    xml_path = Path(args.xml_dir) / doi_to_filename(doi)
    if not xml_path.exists():
        print(f"XML not found: {xml_path}")
        sample = [p.name for p in sorted(Path(args.xml_dir).glob("*.xml"))[:3]]
        print(f"  (first files in dir: {sample})")
        sys.exit(1)

    paper_text = extract_plain_text(xml_path.read_text(encoding="utf-8", errors="ignore"))
    if not paper_text.strip():
        sys.exit(f"extract_plain_text returned empty text for {xml_path}")
    if len(paper_text) > 110_000:
        print(f"  WARNING: paper is {len(paper_text):,} chars (~{len(paper_text)//4:,} tokens) — may exceed 32K")

    user_msg = (
        f"PAPER:\n<<<\n{paper_text}\n>>>\n\n"
        f"QUESTION: {q['question']}\n\n"
        f"Return only the JSON object."
    )

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},  # vLLM JSON mode -> valid JSON out
    }
    if args.model.lower().startswith("qwen3"):          # same rule as step3 resolve_args
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.time()
    resp = requests.post(args.url, json=payload, timeout=300)
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):                     # None: no answer text came back
        print("WARNING: reader returned no message content (None); treating it as empty output")
        content = ""
    usage = data.get("usage", {})

    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1:
            try:
                parsed = json.loads(content[s:e + 1])
            except Exception:
                pass

    print("=" * 78)
    print(f"Q{args.qid}  |  DOI {doi}  |  prompt_tokens={usage.get('prompt_tokens', '?')}  |  {dt:.1f}s")
    print("=" * 78)
    print("QUESTION     :", q["question"])
    print()
    print("GOLD answer  :", q.get("expected_answer") or "(not in this eval set)")
    print("GOLD location:", q.get("source_location") or "(not in this eval set)")
    print("-" * 78)
    if isinstance(parsed, dict) and parsed:
        print("READER relevant         :", parsed.get("relevant"))
        print("READER location         :", parsed.get("location"))
        print("READER assumption       :", parsed.get("assumption"))
        print("READER supporting_quote :", parsed.get("supporting_quote"))
    else:
        print("Could not parse JSON. Raw model output:")
        print(content)


if __name__ == "__main__":
    main()