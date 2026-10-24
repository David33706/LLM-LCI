#!/usr/bin/env python3
"""
regrade_judge.py - re-grade an existing judged eval run with a DIFFERENT judge
model, to test whether the qwen2.5:7b judge is too lenient.

It does NOT regenerate answers. It loads the answers already produced (from the
results_*.json of a --generate --judge run) and re-runs ONLY the judge step with
a second model, then diffs the two verdicts. This isolates the judge: same
questions, same answers, only the grader changes - so any difference is the
grader, not the generator.

Uses the SAME grading prompt as eval/evaluate.py.

It needs results with expected answers, i.e. a run on the full eval workbook. That
workbook and the June results files are kept in the author's private archive (not
published: they contain verbatim paper text or licensed data).

Run from the repo root:
  python eval/regrade_judge.py \
    --results eval_runs/results_<ts>_hybrid_judged.json \
    --judge-model qwen3:8b-q8_0
"""

import argparse
import glob
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def judge(llm, question, expected, actual):
    """Identical prompt to eval/evaluate.py's llm_judge, so only the MODEL differs."""
    prompt = (
        "You are grading a question-answering system. Decide whether the SYSTEM ANSWER "
        "is correct, using the REFERENCE ANSWER as ground truth. Minor wording or unit "
        "formatting differences are acceptable; the facts/values must match.\n\n"
        f"QUESTION: {question}\n"
        f"REFERENCE ANSWER: {expected}\n"
        f"SYSTEM ANSWER: {actual}\n\n"
        "Reply with exactly one word on the first line: PASS or FAIL. "
        "Optionally add a brief reason on the next line."
    )
    raw = llm.generate(prompt, [])
    verdict = raw.strip().splitlines()[0].upper() if raw.strip() else ""
    return verdict.startswith("PASS"), raw.strip()[:200]


def main():
    ap = argparse.ArgumentParser(description="Re-grade saved eval answers with a different judge model.")
    ap.add_argument("--results", default=None,
                    help="results_*.json from a --generate --judge run "
                         "(default: latest eval_runs/results_*hybrid_judged.json)")
    ap.add_argument("--judge-model", default="qwen3:8b-q8_0",
                    help="Ollama tag of the alternate judge model.")
    args = ap.parse_args()

    path = args.results
    if not path:
        cands = sorted(glob.glob("eval_runs/results_*hybrid_judged.json"))
        path = cands[-1] if cands else None
    if not path or not Path(path).exists():
        print("ERROR: no results JSON found. Pass --results path/to/results_*.json")
        sys.exit(1)

    data = json.load(open(path, encoding="utf-8"))
    rows = [r for r in data.get("results", []) if r.get("answer")]
    if not rows:
        print(f"ERROR: {path} has no generated answers (was it run with --generate?).")
        sys.exit(1)
    if not any(r.get("expected_answer") for r in rows):
        print("ERROR: this needs the full private eval workbook (expected_answer/source_location), "
              f"which is not published. {path} has no expected answers to grade against.")
        sys.exit(1)

    try:
        from rag import LLMWrapper
    except ImportError:
        print("ERROR: could not import LLMWrapper from rag.py.")
        print("Run from the repo root: python eval/regrade_judge.py ...")
        sys.exit(1)

    llm = LLMWrapper(model_name=args.judge_model)
    if not llm.is_available():
        print(f"ERROR: Ollama not reachable, or '{args.judge_model}' not pulled.")
        print(f"  ollama pull {args.judge_model}")
        sys.exit(1)

    print(f"file       : {path}")
    print(f"new judge  : {args.judge_model}")
    print(f"answers    : {len(rows)} (no regeneration)\n")

    regraded = []
    for r in rows:
        now, raw = judge(llm, r["question"], r["expected_answer"], r["answer"])
        regraded.append((r, now, raw))

    n = len(rows)
    old_pass = sum(1 for r in rows if r.get("judge_pass") is True)
    new_pass = sum(1 for (_, now, _) in regraded if now)
    # headline = retrieved AND judged-correct (excludes lucky-guess-on-miss)
    old_head = sum(1 for r in rows if r.get("judge_pass") is True and r.get("hit_at_k"))
    new_head = sum(1 for (r, now, _) in regraded if now and r.get("hit_at_k"))

    print(f"judge PASS (any)            old 7b: {old_pass}/{n} = {old_pass/n:.1%}"
          f"   |   {args.judge_model}: {new_pass}/{n} = {new_pass/n:.1%}")
    print(f"retrieved AND judged-correct  old 7b: {old_head}/{n} = {old_head/n:.1%}"
          f"   |   {args.judge_model}: {new_head}/{n} = {new_head/n:.1%}\n")

    flips = [(r["id"], bool(r.get("judge_pass")), now, bool(r.get("hit_at_k")), raw)
             for (r, now, raw) in regraded if bool(r.get("judge_pass")) != now]
    if flips:
        print(f"{len(flips)} verdict(s) changed:")
        for qid, was, now, hit, raw in flips:
            arrow = f"{'PASS' if was else 'FAIL'} -> {'PASS' if now else 'FAIL'}"
            print(f"  Q{qid} [{'hit' if hit else 'MISS'}]: {arrow}")
            print(f"     {raw}")
    else:
        print("No verdicts changed - the two judges agree on every answer.")

    print("\nReading it: a hit flipping PASS->FAIL lowers the real grounded-answer rate")
    print("(stricter grader); a MISS flipping PASS->FAIL just removes a false positive.")
    print("The gap between the two graders is your error bar on that number.")


if __name__ == "__main__":
    main()