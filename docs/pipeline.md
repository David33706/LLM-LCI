# Pipeline architecture

LLM-LCI is a **librarian** for life-cycle-assessment papers: given a question about an
inventory assumption, it returns the study, the section or table that states it, and the
verbatim quote. It locates and quotes — it never computes, converts, or compares numbers.

The headline configuration is `--trim --tables`: the reader receives each paper's
**methods, materials, results, and full table contents** — abstract, introduction,
discussion, and conclusions are removed in Python before the model sees a byte.

## Stages

**0 · Corpus.** Screened WoS list-2: 1,965 papers, 1,959 full-text Elsevier XMLs on disk
(6 unavailable). XML, not PDF — Elsevier XML carries labelled sections and real table
markup, which is what makes "the answer is in §2.1.3" possible at all.

**1 · Indexing (offline).** Prose is chunked (1,200 chars, 200 overlap) and embedded
(MiniLM) into ChromaDB; tables are additionally indexed as title-enriched records
(`ingest/index_tables.py`). A BM25 keyword index is built from the same chunks.
Rebuilds must **wipe first** — chunk IDs embed a UUID and never overwrite.

**2 · Retrieval (per question).** Dense and BM25 rankings are fused with weight
alpha = 0.4 (40% meaning, 60% exact keywords — the measured optimum on this corpus).
The top 40 chunks are deduplicated to 8 unique papers by DOI. DOIs come from chunk
metadata; the model is never asked to produce one.

**3 · Routing.** *"Which study contains X?"* (TASK1) is answered by the retrieval
ranking directly — top-3 titles + DOIs, **no LLM call**. *"What assumption was made?"*
(TASK2) proceeds to the reader.

**4 · Reader (TASK2), per paper — five steps, 8 papers in parallel:**

1. **Trim** (`build_trimmed_text`): drop Abstract / Introduction / Discussion /
   Conclusions / acknowledgements-class sections; keep Title, Methods, Materials,
   Results, LCI/goal-scope sections, and captions. KEEP overrides DROP, so
   *"Results and discussion"* survives. Under 300 surviving tokens → full-text
   fallback. Validated: `tools/measure_trim.py` confirms **no gold location lives
   in a dropped section**.
2. **Inject table cells** (`render_tables_text`): every table as a
   `## Table N: caption` block with headers and all rows — the LCI payload that
   prose captions alone never carry.
3. **Fit the context budget** (context − 1,500 answer − 500 scaffold; 129,072
   tokens with the adopted reader) — see *Update logs* below.
4. **Read**: one paper per call to vLLM (Qwen3.5-35B-A3B-FP8; previously
   Qwen2.5-32B-AWQ), librarian system prompt ("locate and quote, never compute"),
   temperature 0, guided JSON output.
5. **Emit JSON**: `{relevant, location, assumption, supporting_quote}`.

**5 · Resolve in Python (the trust boundary).** The model's *quote* is trusted; its
*location claim* is not. `derive_location()` finds the quote in the feed
(whitespace/case-normalized; `rfind` so the last occurrence lands in the deepest
subsection) and walks back to the nearest `## ` heading — overriding the model
(74/74 derived in the headline run). Table quotes anchor to their `## Table N` block.
Results land in `eval_runs/answers_*.xlsx` with a blank `valid?` column for expert
review.

**6 · Evaluation.** Four levels, each stricter; drops between levels localize the
failure: gold-fed → gold-in-answer → section-hit (`eval/audit_locations.py`) →
answer-match (`eval/judge_answers.py`, calibrated against Prof. Lam's labels).
Expert-marked alternates flow back via `eval/apply_valid_dois.py` (multi-gold).

## Update logs

### 2026-07-14 · Feed truncation policy (reserve-budget)

The feed (trimmed prose + tables) must fit the reader's 30,768-token budget; trimmed
feeds run ~9–11K tokens, so truncation is rare and flagged `[TRUNC]` in the run log.

**Before:** one cut kept the first 30,768 tokens. Tables sat at the end, so an
over-budget paper lost its **tables first** — where gold answers sometimes live.

**Now:** the tables' tokens are **reserved first** and appended intact; the **prose is
cut from its tail** (late Results) — Title and Methods stay safe. Exception: tables
alone nearing the whole budget (> budget − 4,000) are tail-cut, with ~4,000 tokens of
prose head guaranteed.

Earlier numbers are unaffected unless their run log shows `[TRUNC]`.

### 2026-07-15 · Reader upgraded: Qwen2.5-32B-AWQ → Qwen3.5-35B-A3B-FP8

Tested two newer-generation readers against the protected baseline (job 28900),
everything else frozen. Qwen3.5-9B lost: it declined four answerable questions
and was slower. **Qwen3.5-35B-A3B-FP8 adopted** — official FP8 checkpoint, MoE
(35B stored, ~3B active per token), served on an L20 with a 131,072-token context.

| | old (2.5-32B, A40) | new (3.5-35B-A3B, L20) |
|---|---|---|
| Composite gold-in-answer | 47/50 | 47/50 |
| Truncated feeds | 45 | **0** |
| Answer-match judge* | 29/34 | 33/35 |
| Section-hit (raw) | 22/28 | 24/29 |
| Wall per question | 139.2 s | **29.2 s** |

\* Same referee for every run — Qwen2.5-32B-AWQ; the challenger never grades itself.

Adoption rests on the two deterministic wins — no truncation, ~4.8× speed — with
accuracy equal or better on every layer. The judge and section gains are within
small-n noise: noted, not claimed. Speed partly reflects the A40→L20 card change,
and CUDA graphs are still off, so headroom remains. A BF16 reference run (H20) is
queued to price the FP8 quantization; expected cost ≈ 0.

### 2026-07-21 · Gold expanded with expert validation; two prompt experiments not adopted

Prof. Lam's validation round is now part of the scoring. Twelve questions
accept expert-confirmed alternate papers (21 in total) besides the original
gold paper. Verdicts are cached: a confirmed or rejected (question, paper)
pair is never re-reviewed, so each validation round is smaller than the last.
Composite unchanged — no current failure was affected.

Reader prompt v2 added strict scope and quote rules. The reader became too
cautious: it declined five answerable questions, including papers the expert
had already confirmed (composite 47→42). Rejected; the original prompt stays.

A stricter judge rule ("naming the same things is not the same as stating the
same fact") changed no verdicts. Three judge configurations across two model
families give identical verdicts: 29/34 = 85% agreement with the expert labels.
The five remaining differences trace to reference wording and question
phrasing, not to the judge — all five are in the current validation round.

### 2026-07-23 — defaults flipped to adopted reader. 
run_gpu.sbatch, serve_vllm.sh, step3_orchestrate.py defaults now = Qwen3.5-35B-A3B-FP8 / L20 / 131K / trim+tables on; 
thinking auto-disabled for qwen3* model names. 
Judge referee unchanged (judge_answers.py pinned to qwen2.5-32b-awq). 
Baseline 28900 preserved as documented override block in run_gpu.sbatch. 
Bare submission now reproduces the adopted 47/50 configuration.

tools/ask.py + ask.sh: one free-text question in, answer out (locate = retrieval
only, no GPU; assume = reader path; auto-routing; loud index/server preflights).
tools/demo.py retired — it demoed the superseded chunk-feed + Ollama path.
ingest/add_papers.py: incremental corpus expansion — DOI dedupe (wipe rule never
triggered), staged canonical builders, count assertions, --redo recovery.
Verified by no-op + delete/re-ingest round-trip landing exactly on 290,247.

### 2026-07-27 — documentation set finalized for handover.
RUNBOOK.md added (root): linear §0–§10 — entry-point branch table, prerequisites,
environment, data (copy / build / download / models), verify, use, expand, then
reproduce / change protocol / limitations for maintainers. Full-rebuild commands
moved out of README into RUNBOOK §3b; README keeps the explanation and the
sanity counts as status. pipeline.md moved to docs/ (README already referenced
that path). requirements-mac.lock, requirements-cluster.lock and
requirements-cluster-vllm.lock added = pip freeze of the three working
environments.

### 2026-07-28 — handover kit rehearsed from a fresh checkout.
Fresh Mac clone installed from requirements-mac.lock + rsynced indexes: locate
path correct. Fresh cluster checkout, bare-default submission (job 31898):
47/50, 0 truncated, 31.3 s/question, locations 80/80 — frozen baseline
reproduced. Three documentation/tooling deviations found and fixed (ask.py index
count print; run_gpu.sbatch REPO now = SLURM_SUBMIT_DIR + echoes repo path,
verified by job 31964; RUNBOOK venv/purge/user-path notes). Provenance artifacts
cited below copied into the repo — /work purges files older than 75 days. (Since
the public release they are kept in the author's private archive, not published:
they contain verbatim paper text or licensed data.)

### 2026-07-28 — kit rehearsed; artifacts archived.
Fresh Mac clone (env from lock) and fresh cluster checkout: bare-default run
reproduced 47/50, 0 truncated, 31.3 s/q, locations 80/80 (job 31898). Three
deviations fixed — ask.py prints the index count; run_gpu.sbatch REPO now
defaults to SLURM_SUBMIT_DIR and echoes it (it had hardcoded the canonical repo,
so jobs from other checkouts ran the wrong tree; verified by job 31964); RUNBOOK
§1/§7 note the /work 75-day purge, user-specific paths, absent cluster git
remote, and the venv the eval scripts need. `.env` stays gitignored.
Archive: cited artifacts, full logs+eval_runs (625 KB) → the author's private
archive (not published).

### 2026-10-24 — Mac restore, audit corrections, loud-failure guards (no metric change claimed)
Branch `oct-cleanup-work`, not pushed. Older entries above stay as written; this
entry corrects them. Sources: archived logs and eval_runs in the author's private
archive (outside the repo), re-checked today.

**Corrections to earlier entries and stage text.**
1. Correction to the header (line 7): since the 2026-07-23 default flip,
   `--trim --tables` are defaults. The adopted command is the bare one.
2. Correction to stage 0: "6 unavailable" = 6 non-`10.1016` DOIs (4× `10.3168/jds`,
   J. Dairy Sci.; 2× `10.1053/j.ajkd`). They were never attempted:
   `download_papers.py` keeps only the `10.1016` prefix.
3. Correction to stage 2: dedup gives *up to* 8 unique papers (7.7 on average),
   not always 8.
4. Correction to stage 4 (Trim) and the "Trim safety" provenance row:
   `tools/measure_trim.py` checked only Q1–30 (Q31–38 have no `source_location`).
   Its old matcher also counted unmatched gold locations (e.g. "Table 3") as safe.
   The fixed script reports them as "UNMATCHED (not verified)". No new trim-safety
   number is claimed here.
5. Correction to stage 5: "74/74 derived in the headline run" is job 28452 (8 Jul).
   Headline job 28900 was 73/73; 29103 was 78/78; 31898 was 80/80.
6. Correction to 2026-07-14: the 30,768-token budget (32K reader) did not make
   truncation rare. At 32K there were 45 truncated feeds per run. The adopted
   reader's budget is 129,072, with 0 truncated.
7. Correction to 2026-07-15: the BF16 H20 reference run was not "queued"; it failed.
   Job 29353 died at server start (`vLLM did not become ready in 10 min`: weights
   loaded in 279 s over NFS, plus 66 s torch.compile, then run_gpu.sbatch's 10-min
   readiness loop killed it). It also used non-matching server args (gpu_util 0.9,
   no fp8 KV cache, no enforce-eager). Jobs 29354/29355 left no logs. A retry needs
   `READY_TIMEOUT_MIN=30` and the FP8 baseline's EXTRA_ARGS.
8. Correction to 2026-07-15 / the "judge + sections" provenance row: judge 33/35 came
   from judge job 29269, run on 29103's dump (`answers_20260715_074948.xlsx`), not
   from job 29103 itself.
9. Correction to 2026-07-21: "Verdicts are cached; never re-reviewed" is not true of
   the code. No code stores rejected (question, paper) pairs.
10. Note (other docs said 0/8; this log never recorded it): judge question-mode
    jobs 28481–28484 all wrote `judged_question_20260708_153558.xlsx` (0/8, 1/8,
    5/8, 4/8), so the archived file shows 4/8. "0/8" is true of job 28481 only.
11. Correction to both 2026-07-28 entries (near-duplicates) and the 31898 provenance
    row: job 31898 ran in the canonical cluster repo, not a fresh checkout. Only job
    31964 ran from the fresh clone.
12. Correction to the 29103 and 31898 provenance rows: 29103 (and 28900) logged
    `multi-gold active on 2 question(s)`; 31898 logged `12`. The current workbook
    has valid_dois on 12 qids, so a bare re-run today scores like 31898. Composite
    is 47/50 under both. 29103 ran with explicit flags; only 31898 was bare.
13. Missing artifacts: `eval_runs/results_20260624_120336_v3.json` (50-Q retrieval
    row), `eval_runs/results_20260623_121336_a_04.json` (30-Q sweep row) and
    `logs/download_log_v2.csv` (corpus row) exist neither in the repo nor in the
    archive. The corpus copy has `elsevier_xml/download_log_v2.csv`; which
    run it records is undocumented.

**Restored on the Mac.**
1. The Mac checkout did not exist. Fresh `git clone` from GitHub.
2. Corpus restored from an existing copy (not re-downloaded): `elsevier_xml/` 1,959 `*.xml` (plus
   `download_log_v2.csv`, so count with `ls elsevier_xml/*.xml | wc -l`) and
   `tables_xml/` 1,824 `*.xlsx` (copied, not regenerated).
3. Conda env `llm-lci`: torch 2.12.0's `libtorch_cpu.dylib` had been deleted, so
   `import torch` failed. `ingest_dir.py` then logged every file as an error with a
   misleading "pip install sentence-transformers" message and still exited 0 with
   "chunks added: 0". Fixed with `pip install --force-reinstall --no-deps
   torch==2.12.0` (the requirements-mac.lock pin). No other package changed.
4. Indexes rebuilt per RUNBOOK §3b (no wipe needed: no `chroma_db/` existed):
   ingest_dir → index_tables (`--wos paper_list2_2025.csv`) → build_bm25_index.
   Counts: 1,959 files, 0 errors, 280,253 prose chunks; 1,965 titles and 9,994 table records from 1,824 papers (280,253 → 290,247); BM25 290,247 chunks, all exact. Wall time (MacBook Air M4 16 GB, lid open; on battery until about 21:30, then on power):
   about 38 min awake (prose ingest about 2,155 s, tables about 100 s, BM25 about 30 s; the raw 4,939 s includes a 2,652 s lid-closed sleep). `caffeinate -i` does not stop lid-close sleep.
5. Retrieval eval on the fresh index: Hit@10 46/50 = 92.0%, MRR 0.671, identical for the 28 Jul code with the documented flags, the 2026-10-24 code with bare defaults, and the 2026-10-24 code at candidate_k 100. Smoke test: `vector store: 290,247 chunks`, top 3 resconrec.2025.108220 / cesys.2025.100297 / cej.2024.158905, as expected.
6. Section-hit recomputed offline with `eval/audit_locations.py`: 29103 dump 24/29,
   31898 dump 24/29, 28900 dump 22/28 (NO_GOLD qids 33–38 in all).
7. Expert label files copied (not moved) into the author's private archive
   (outside the repo). The 34-label file recomputes judge–expert agreement 29/34
   with `eval/compare_judges.py`.

**Code changes (loud failures; details in the branch diff).**
1. Reader path: sequential and ThreadPool runs share one call path. A new SUMMARY
   line counts reader errors (http / parse / other) and skipped papers; all
   existing SUMMARY lines are byte-identical. New header lines `tokenizer:` and
   `titles:`, with a WARNING on fallback or 0 titles. An empty vector store exits 1.
   `--dump-answers` also writes `eval_runs/manifest_<ts>.json`.
2. **Could move a reader number:** `relevant` is now parsed strictly. The string
   "false" used to count as YES and now is NO; a non-boolean other than
   true/false/yes/no strings (e.g. `1`) used to count as YES if truthy and now is an
   error. No change is claimed until the cluster frozen-baseline re-run is done and
   compared on all four layers (RUNBOOK §8).
3. Eval scripts: judge_answers exits 1 on a missing/empty titles file (opt-out
   `--no-titles`), refuses to overwrite, and parses `match` strictly; prompt text is
   unchanged when titles exist. compare_judges no longer counts a missing judged row
   as agreement. evaluate.py defaults are now hybrid / weighted / α=0.4 / top-k 10
   (candidate_k stays 50, the measured config). New read-only
   `eval/validate_eval_set.py`; it rejects the stale June workbook copy.
4. Ingest and libraries: ingest_dir exits 1 if any file errors and refuses a
   non-empty collection without `--resume`. index_tables defaults to
   `--wos paper_list2_2025.csv` and exits 1 on 0 titles. `hybrid.build_hybrid_retriever`
   defaults to weighted / 0.4 / candidate_k 100. `utils/elsevier.py` redacts the key
   in printed URLs and exceptions.
5. Cluster scripts: run_gpu.sbatch and serve_vllm.sh refuse a busy port 3000 and
   wait for the served model id; `READY_TIMEOUT_MIN` (default 10). tools/ask.py
   checks the served model id and max_model_len.

**Not changed.** Eval workbooks and gold labels, judge prompts, reader prompts,
chunking and parser, chunk IDs and metadata, `audit_locations` matching, legacy
`screening/` and Docling `tables/` code. Frozen baseline values in README
were not edited; whether the new code reproduces them is unverified until the
cluster re-run.

### 2026-10-24 (later) — merged to `main` without a cluster re-run

The author chose not to rebuild the cluster just to re-run the frozen baseline: `/work` was
probably purged, and a rebuild costs two venvs, 57 GB of models and the MiniLM cache.
Evidence for merging: the only change that could move a reader number is strict
`relevant` parsing (item 2 above). The prompt makes a "false" answer carry empty
fields, and 0 of 499 YES rows across the 7 archived answer dumps (6–27 Jul) have an
empty quote or assumption, so the old `bool("false")` bug never turned a NO into a
YES. A missing key or other value never counted as YES under either version. No
metric change is claimed. The first bare `--all` run after the next cluster rebuild is
the confirmation (all four layers; expect `reader errors: 0`).

### 2026-10-25 — public release

Code published under the MIT license with the supervisor's approval, on condition
that it carries no copyrighted content and no API key. Removed (verbatim paper text
or licensed data): eval workbooks, expert labels, answer dumps, judged files, run
logs, WoS screening exports (abstracts). Full text, tables and indexes were never
tracked. The eval set is not published at all (questions, gold DOIs, expected
answers, section locations): the agreement with the supervisor was code only, with
the eval set kept private. The eval scripts default to the private workbook
`eval/LCA_RAG_eval_set_v3_50.xlsx` and stop with an ERROR (exit 1) when it is
missing; they still read a user's own `.xlsx` or `.csv`. Ships instead:
`paper_list2_2025.csv` (the 1,965 DOIs of the corpus, columns `DOI`, `Article Title`);
its titles come from public Crossref metadata, not from WoS. Titles feed the table
records (embeddings and BM25) and the judge's STUDY TITLE line, so exact reproduction
of the reported numbers also needs the WoS titles used in those runs (private
archive). Cluster paths now come
from `/work/$USER`. Removed files stay in the author's private archive. No reported
number changed.
`screening/download_papers.py` now reads a CSV without `LCA_prediction` as
all-positive, so RUNBOOK §3c runs on the shipped list (checked offline: 1,959
DOIs collected; the download was not re-run). `tools/measure_trim.py` exits 1
with an ERROR when no gold XML is found.

## Provenance

Numbers here are **immutable run records** — each row is what a named artifact says,
and stays true forever. *Current* status numbers live in the README, which is the
single mutable source; when results change, update the README and add a row here.
The cited artifacts are kept in the author's private archive and are not published:
they contain verbatim paper text or licensed data.

| Claim | Value | Artifact |
|---|---|---|
| Retrieval, 30-Q sweep optimum (a=0.4) | Hit@10 96.7% · MRR 0.666 · chunk-hit 40% | `eval_runs/results_20260623_121336_a_04.json` |
| Retrieval, full 50-Q set | Hit@10 92.0% · MRR 0.671 | `eval_runs/results_20260624_120336_v3.json` |
| End-to-end composite (trim + tables ON) | 47/50 = 94% (TASK2 36/38 · TASK1 11/12) | `gpu_28900.log` |
| Trim safety | every gold location survives trimming | `tools/measure_trim.py` output |
| Corpus | 1,959 / 1,965 screened list-2 XMLs | `logs/download_log_v2.csv` reconciliation |
| Reader bake-off, adopted config | composite 47/50 · 0 truncated · 29.2 s/q | `gpu_29103.log` |
| Adopted reader, judge + sections | judge 33/35 · section-hit 24/29 raw | `judged_expected_20260715_074948.xlsx` · `audit_locations` on `answers_20260715_074948.xlsx` |
| Rehearsal reproduction (fresh checkout, bare defaults) | 47/50 · 0 truncated · 31.3 s/q · locations 80/80 | `gpu_31898.log` |
| Section-hit, 31898 dump (offline recompute, 2026-10-24) | 24/29 raw (NO_GOLD qids 33–38) | `audit_locations` on `answers_20260727_043415.xlsx` (private archive) |
| Judge–expert agreement (recomputed 2026-10-24) | 29/34 = 85% (judge MATCH 28/34) | `eval/compare_judges.py` on the 34-label file (private archive) |
| BF16 reference run (H20), job 29353 | failed at server start: vLLM not ready in 10 min; non-matching server args | `gpu_29353.log` (private archive) |
| Retrieval, full 50-Q set, fresh Mac index (2026-10-24) | Hit@10 46/50 = 92.0%, MRR 0.671, identical for the 28 Jul code with the documented flags, the 2026-10-24 code with bare defaults, and the 2026-10-24 code at candidate_k 100 | Mac `eval_runs/` results JSON (gitignored, per machine) |
| Strict `relevant` parsing vs archived runs (2026-10-24) | 0 of 499 YES rows with an empty quote or assumption, so the old `bool("false")` bug never fired | the 7 archived `answers_*.xlsx` (private archive) |
