# RUNBOOK — LLM-LCI operations

**What to type, in order.** Read straight through from wherever §0 sends you.

Three documents, three jobs:

- `README.md` — what this is and how it works (current state)
- `RUNBOOK.md` — this file: what to type (procedures)
- `docs/pipeline.md` — what was tried and decided, and which artifact proves each
  number (append-only history; read it before changing anything)

Tested on macOS (locate-only) and Linux/DKUCC (everything).
Windows: untested; use WSL2 — vLLM is Linux-only regardless.

---

## §0 Start here — what do you already have?

Everything starts from a clone of this repo:

```
git clone https://github.com/David33706/LLM-LCI
```

| On this machine | Read |
|---|---|
| Everything works already | §5, §6 |
| Repo + the indexes available to copy in | §1 → §2 → §3a → §4 → §5 |
| Repo + `elsevier_xml/` only (no indexes) | §1 → §2 → §3b → §4 → §5 |
| **Repo only** | §1 → §2 → §3c → §3b → §3d → §4 → §5 |
| Nothing (new account, new machine) | §1 onward, in order |

A fresh public clone is the "Repo only" row. The corpus (`elsevier_xml/`,
`tables_xml/`) and the indexes built from it are licensed Elsevier content and
are not published, so you download the XMLs with your own Elsevier API key
(§3c) and build the indexes (§3b). Without them nothing runs. The eval set is
not published either; to evaluate, bring your own questions file (§7).

§7–§10 are for maintainers, not for first use.

---

## §1 Prerequisites

- A Slurm GPU cluster account with scratch space at `/work/$USER`. The scripts
  were written for DKUCC with a 48 GB GPU (L20; A40 for the Qwen2.5 baseline and
  judge). Pass your cluster's `--account` and `--partition` to `sbatch`/`srun`
  if it needs them; the commands below use `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION`.
- The repo (§0). GitHub is often unreachable from mainland networks; fallback is
  `rsync` from a machine that already has the repo.
- Every new cluster shell:

```
module load anaconda
source /work/$USER/venv-llmlci/bin/activate
```

- HF Hub is blocked on the cluster — these belong in `~/.bashrc`:

```
export HF_HOME=/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Three things that bite newcomers:

- **`/work` is scratch.** Files older than 75 days are purged automatically.
  Provenance logs and eval dumps must be archived off `/work`. The artifacts
  cited in `docs/pipeline.md` (run logs, answer dumps such as job 31898's
  `answers_20260727_043415.xlsx`, job 28900's dump and judged file) are kept in
  the author's private archive (not published: they contain verbatim paper text
  or licensed data). Indexes and models are rebuildable per §3b/§3d.
- **Cluster paths follow one work folder.** The scripts build model, venv, log
  and HF-cache paths from `WORK_ROOT="${LLMLCI_WORK:-/work/$USER}"`. Set
  `LLMLCI_WORK` to use another folder. `#SBATCH` lines cannot read environment
  variables, so `run_gpu.sbatch` logs to `/work/%u/logs/gpu_%j.log` (`%u` is your
  user name); with `LLMLCI_WORK` set, also pass
  `--output=$LLMLCI_WORK/logs/gpu_%j.log` to `sbatch`. Slurm opens that log
  before the script runs, so create the folder once: `mkdir -p /work/$USER/logs`.
- **The author's cluster checkout had no GitHub remote.** Code arrived by rsync
  from the authoring machine, so `git pull` did not work there. Jobs run the
  tree they were submitted from (`SLURM_SUBMIT_DIR`), and each log's first line
  names it.

## §2 Code + environment

Environments are pinned to the exact versions that produced the frozen numbers.

- Mac: `conda create -n llm-lci python=3.11`, then `conda activate llm-lci`,
  then `pip install -r requirements-mac.lock`
- Cluster, pipeline venv: `pip install -r requirements-cluster.lock`
- Cluster, vLLM venv (`/work/$USER/venv-vllm`):
  `pip install -r requirements-cluster-vllm.lock`

The locks are per-machine-class — the cluster locks contain Linux/CUDA wheels
that will not install on macOS. `requirements.txt` says what the project asks
for; the `.lock` files say what actually worked.

**Cluster pipeline lock — never executed from scratch.** It pins
`torch==2.12.1+cu126`, which is not on PyPI, so the install probably needs the
PyTorch index (untested):

```
pip install -r requirements-cluster.lock --extra-index-url https://download.pytorch.org/whl/cu126
```

The lock also pins both cu12 and cu13 NVIDIA libraries; expect to resolve that
by hand if pip complains. `requirements-gpu.txt` holds the torch CUDA notes for
GPU machines (replace the unpinned torch with the cu126 wheel, then check
`torch.version.cuda`).

**Mac: check torch before any ingest.** On 2026-10-24 the Mac env's
`libtorch_cpu.dylib` had been deleted, so `import torch` failed and ingest
logged every file as an error with a misleading "pip install
sentence-transformers" message. This line must print nothing and exit cleanly:

```
python -c "import torch, sentence_transformers"
```

If it fails, reinstall the pinned torch without touching anything else:

```
pip install --force-reinstall --no-deps torch==2.12.0
```

## §3 Data — get it or build it

These four directories are gitignored and per-machine:
`chroma_db/`, `bm25_index/`, `elsevier_xml/`, `tables_xml/`.

### §3a Copy indexes from a working machine (fastest)

```
rsync -av <source>:/path/to/LLM-LCI/chroma_db/    chroma_db/
rsync -av <source>:/path/to/LLM-LCI/bm25_index/   bm25_index/
rsync -av <source>:/path/to/LLM-LCI/elsevier_xml/ elsevier_xml/
rsync -av <source>:/path/to/LLM-LCI/tables_xml/   tables_xml/
```

Target state: `chroma_db` collection `papers` = **290,247** records,
`bm25_index/papers` = **290,247** chunks, `elsevier_xml/` = **1,959** XMLs
(count them with `ls elsevier_xml/*.xml | wc -l`; a bare `ls | wc -l` gives
1,960 because `download_log_v2.csv` sits in the same folder).

Local copies are fast. Pulling `chroma_db/` between machines is not — it is
~2.9 GB in one sqlite file, and a trans-Pacific ssh transfer can take hours. Add
`-P` to show progress and resume an interrupted transfer, and
`--exclude='.DS_Store'` when copying from macOS. On the same cluster, symlink
instead of copying.

Then go to §4.

### §3b Build the indexes from XMLs

*[execution-verified — June 2026 clean rebuild reproduced these counts;
re-exercised 2026-07-24 by the add_papers round-trip; rebuilt on the Mac
2026-10-24 (no wipe needed, no `chroma_db/` existed; `tables_xml/` restored
from an existing copy, not regenerated): 1,959 files, 0 errors, 280,253 prose chunks; 1,965 titles and 9,994 table records from 1,824 papers (280,253 → 290,247); BM25 290,247 chunks, all exact]*

Needs: §2 environment (run the torch check in §2 first), `elsevier_xml/`, and
`paper_list2_2025.csv` (in the repo). Check the corpus is complete — this must
print 1959:

```
ls elsevier_xml/*.xml | wc -l
```

Run under `tmux` (cluster) or `caffeinate -i` (Mac). Measured times: June 2026
Mac prose ingest 2,090 s; full build on a MacBook Air M4 16 GB, 2026-10-24:
about 38 min awake (prose ingest about 2,155 s, tables about 100 s, BM25 about 30 s; the raw 4,939 s includes a 2,652 s lid-closed sleep). BM25 alone ≈ 30 s.

**Mac: keep the lid open and the charger plugged in.** `caffeinate -i` does not
stop lid-close sleep. On 2026-10-24 a lid-closed sleep paused the build for
about 44 min (excluded from the time above).

**Adding only a few papers to an existing index? Do NOT use this section — use
§6.** This is for a full build.

**Step 1 — wipe any existing collection.** ChromaDB persists to disk and prose
chunk IDs contain a random UUID, so re-ingesting on top of a non-empty or
partially-built collection APPENDS orphan chunks instead of overwriting, silently
corrupting retrieval and eval.

```
python -c "import chromadb; c=chromadb.PersistentClient(path='chroma_db'); c.delete_collection('papers'); print('deleted')"
```

On a fresh machine with no `chroma_db/` this raises NotFoundError — that is
correct, there was nothing to delete. Continue.

**Step 2 — build, in this order** (BM25 last, after tables exist):

```
python -m tables.xml_tables --input-dir elsevier_xml --output-dir tables_xml
```

```
python ingest/ingest_dir.py --xml-dir elsevier_xml --collection papers
```

```
python ingest/index_tables.py --all --xlsx-dir tables_xml --xml-dir elsevier_xml --collection papers --wos paper_list2_2025.csv
```

```
python ingest/build_bm25_index.py --collection papers --out bm25_index/papers
```

**Acceptance — these counts ARE the test.** Chunking is deterministic, so a
correct rebuild lands on them exactly:

- `ingest_dir` → `chunks added: 280,253`
- `index_tables` → `count before: 280,253` → `count after: 290,247`
- `build_bm25_index` → `chunks indexed: 290,247`

Anything else (e.g. 301,405 or 311,399) means the collection was not empty —
wipe and rebuild. Wall time for the full build: about 38 min awake (prose ingest about 2,155 s, tables about 100 s, BM25 about 30 s; the raw 4,939 s includes a 2,652 s lid-closed sleep) (MacBook Air M4
16 GB, 2026-10-24, lid open, on power).

**Loud guards (since 2026-10-24).** Each prints an ERROR line and exits 1, so a
broken step stops the sequence instead of passing silently:

- `ingest_dir.py` without `--resume` on a collection that already has records:
  `ERROR: collection 'papers' already has N records; ... Wipe first (RUNBOOK §3b
  step 1) or pass --resume.`
- `ingest_dir.py --resume` when it cannot read the existing DOIs:
  `ERROR: --resume could not read existing DOIs from the collection: ...`
- `ingest_dir.py` when any file errored (after the summary):
  `ERROR: N files errored — the collection is incomplete; see <log path>`.
  Before this guard a broken torch logged every file as an error and still
  exited 0 with `chunks added: 0`.
- `index_tables.py` now defaults to `--wos paper_list2_2025.csv`. If titles are
  on and that file is missing or loads 0 titles:
  `ERROR: --wos paper_list2_2025.csv: file not found` (or `0 titles loaded`).
  Also exit 1 on a missing `--xlsx-dir`, on 0 table records built, and when the
  count delta differs from the number of new records.
- `build_bm25_index.py` exits 1 when `--chroma-dir` is missing
  (`ERROR: --chroma-dir not found: ... (run from the repo root)`) or the
  collection is empty (`ERROR: collection 'papers' is empty.`).

### §3c Download the corpus

*[recorded — May 2026; not re-executed since]*

Skip if you already have `elsevier_xml/`.

```
python screening/download_papers.py --input paper_list2_2025.csv --xml-dir elsevier_xml
```

**Notes on this command (found 2026-10-24):**

1. Fixed 2026-10-25. It used to crash with `KeyError: 'LCA_prediction'`: the
   script keeps rows with `LCA_prediction == 1`, and `paper_list2_2025.csv` has
   no such column. A CSV without that column now counts as all-positive (it
   prints `no LCA_prediction column, using every row`). Checked offline: it
   collects 1,959 DOIs from the shipped list. The download itself has not been
   re-run since the fix. The June download used a screened CSV (`DOI` plus an
   `LCA_prediction` column), which is kept in the author's private archive (not
   published: it contains licensed Web of Science records).
2. It keeps only `10.1016/` DOIs, so 6 of the 1,965 are never attempted
   (4× 10.3168/jds, J. Dairy Sci.; 2× 10.1053/j.ajkd). That is why the corpus
   is 1,959, not "6 unavailable".
3. It does not call `load_dotenv`. Set `ELSEVIER_API_KEY` to your own Elsevier
   API key in the environment first, without printing it, e.g.
   `set -a && source .env && set +a` (`.env` is gitignored).
4. It writes its log to `elsevier_xml/download_log_v2.csv` (inside
   `--xml-dir`), not `logs/`.

Constraints: set `ELSEVIER_API_KEY` to your own Elsevier API key. Full text
requires an institutional entitlement (IP-based), so run from your institution's
network; the entitlement comes from the network, not from the key. Since
2026-10-24 `utils/elsevier.py` redacts the `APIKey=` value in every printed URL
and error; the key is still sent as a URL parameter. Expect **1,959 of 1,965** XMLs; check with
`ls elsevier_xml/*.xml | wc -l` and `elsevier_xml/download_log_v2.csv`. Then §3b.

### §3d Models

*[recorded]*

Needed only for assumption questions (the reader). Locate questions need no model
beyond MiniLM. HF Hub is blocked on the cluster — download via ModelScope, or
fetch on a Mac and `rsync` across. End state:

- `/work/$USER/models/Qwen3.5-35B-A3B-FP8` — adopted reader
- `/work/$USER/models/Qwen/Qwen2___5-32B-Instruct-AWQ` — pinned judge + baseline

`run_gpu.sbatch` and `serve_vllm.sh` point at those exact paths.

## §4 Verify the install (no GPU needed)

```
python tools/ask.py "Which study contains the LCI for constructed wetlands for wastewater treatment?"
```

Expect `vector store: 290,247 chunks` and three titles + DOIs within seconds.
Any missing piece produces a loud ERROR naming exactly what to fix.

To check the index count on its own:

```
python -c "import chromadb; print(chromadb.PersistentClient(path='chroma_db').get_collection('papers').count())"
```

To check the eval set is the frozen one (read-only; 50 rows, 38/12 split,
`valid_dois` on 12 qids / 22 pairs). A stale copy without the `valid_dois`
column would silently change scoring; this check rejects it. It needs no
corpus or GPU, but it needs the author's private eval set, which is not
published:

```
python eval/validate_eval_set.py
```

It checks the private workbook `eval/LCA_RAG_eval_set_v3_50.xlsx` (gitignored)
by default. Expect a last line that starts with
`OK: eval/LCA_RAG_eval_set_v3_50.xlsx`. Any mismatch prints `ERROR: ...` lines
and exits 1. A CSV export of the same 50 questions (columns
`id,type,question,source_doi,valid_dois`) can be checked with `--eval-set`.
Without the file it prints `ERROR: eval set not found: ...` and exits 1. It
checks only the frozen 50-question set, so it rejects any other questions file.

## §5 Use it — ask a question

**One-shot, zero setup** (cluster login node) — for anyone who doesn't want to
manage a server:

```
./ask.sh "In LCA studies of MEA-based carbon capture, what CO2 capture efficiency is assumed?"
```

Each call boots its own server: queue + model load ≈ 3–6 min, then the answer.
If the job ends without writing its log, `ask.sh` now prints
`ERROR: job N finished (exit RC) but its log is missing: expected ...` and
exits non-zero.

**Locate questions need no GPU** — any machine with the indexes:

```
python tools/ask.py "Which study contains the LCI for X?"
```

≈ 2 seconds.

**Interactive session** — many questions, ≈ 30 s each after one load:

```
srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=03:00:00 --pty bash
```

```
bash serve_vllm.sh
```

```
source /work/$USER/venv-llmlci/bin/activate
```

```
python tools/ask.py "your question"
```

Repeat the last command freely, then `exit` — **that releases the GPU; do not
skip it.** `serve_vllm.sh` refuses to run on a node without a GPU and prints the
`srun` line above, so a wrong-node attempt fails clearly instead of crashing.

Since 2026-10-24 the cluster scripts also refuse a stale server:

- **Port 3000 busy.** `serve_vllm.sh` and `run_gpu.sbatch` exit 1 with
  `ERROR: something already answers on localhost:3000 — refusing to start a
  second server` if anything already answers there, because an older server
  would be mistaken for the new one. Stop it first (`pkill -f 'vllm serve'`).
- **Readiness means the right model.** Both scripts wait until `/v1/models`
  lists the served model name, not just any reply.
- **`tools/ask.py` preflight** checks the server's model id and
  `max_model_len` against `--model` and `--ctx`, and prints an ERROR on a
  mismatch.

Notes: `tools/ask.py` chdirs to the repo root itself, so a full path works from
any directory. Retrieval is phrasing-sensitive — two wordings of one question can
surface partly different candidate papers; that is retrieval behavior, not
nondeterminism.

## §6 Expand it — add new papers

```
python ingest/add_papers.py --xml-dir new_xmls
```

`new_xmls` is an example: use the folder that holds the new XMLs. Files must be
named by DOI, as downloaded: `10.1016_j.xxx.2025.12345.xml`. Since 2026-10-24
it first checks that `elsevier_xml/`, `tables_xml/`, `chroma_db/` and
`paper_list2_2025.csv` exist, and `--redo` with a DOI that has no file in the folder exits 1.

It skips DOIs already indexed (so §3b's wipe rule never applies here), runs the
same canonical builders on a staging folder, copies the XMLs and extracted tables
into `elsevier_xml/` and `tables_xml/` for future full rebuilds, asserts every
count, and aborts loudly on any mismatch — leaving its staging folder for
inspection. Crash recovery: `--redo <doi>` deletes that paper's records and
re-ingests it.

Two follow-ups it reminds you about:

- **Indexes are per-machine.** Run the same command on the other machine, or
  rsync `chroma_db/ bm25_index/ elsevier_xml/ tables_xml/`.
- **Optional:** append the new papers' `DOI` and `Article Title` (its two
  columns) to `paper_list2_2025.csv` so titles appear in locate mode and enrich table records
  (`--redo <doi>` to re-embed). Without it, locate mode shows
  "(title not in list)" — cosmetic only.

---

## §7 Reproduce the frozen numbers

Adopted configuration — bare defaults, deliberately:

```
sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8 --dump-answers
```

Expect composite **47/50** (TASK2 36/38 · TASK1 11/12), **0 truncated feeds**,
≈ 29–31 s/question. Log: `/work/$USER/logs/gpu_<jobid>.log`. The default eval
set is the author's private workbook `eval/LCA_RAG_eval_set_v3_50.xlsx`
(gitignored, not published); the composite score needs only its gold DOIs and
`valid_dois`. Without it the run prints `ERROR: eval set not found: ...` and
exits 1. Exact reproduction of these numbers needs that file.

**Your own questions.** Pass an `.xlsx` (sheet `Questions`) or a `.csv` with
columns `id,type,question,source_doi` (optional: `valid_dois`,
`expected_answer`, `source_location`). The scores are not comparable to the
frozen numbers:

```
sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8 --dump-answers --eval-set my_questions.csv
```

Read these lines in the log before trusting the numbers (new since 2026-10-24
unless noted):

- header: `vector store: 290247 chunks` (unchanged; an empty or unreadable store
  is now an ERROR and exit 1), `tokenizer: ...` (a WARNING means the len//4
  fallback is in use), `titles: N loaded from paper_list2_2025.csv` (a WARNING
  means 0 titles).
- SUMMARY: `reader errors: N (http A, parse B, other C); skipped papers:
  xml_missing D, empty_text E`. Before this line, failed reader calls vanished
  from `--all` runs. If N > 0 a WARNING lists the qids; those calls counted as
  not-YES. Re-run one with `--qid` to see its ERR lines. All other SUMMARY lines
  are unchanged.
- `--dump-answers` now also writes `eval_runs/manifest_<ts>.json` next to the
  dump (same timestamp): args, store count, tokenizer, titles, eval-set sha256,
  git commit, host, `SLURM_JOB_ID`. Archive it with the dump.

**Not yet re-run after the 2026-10-24 reader-parsing change.** `relevant` is
now parsed strictly: the string "false" counts as NO (it used to count as YES),
and a value that is not true/false/yes/no counts as an error. The frozen
baseline must be re-run before claiming the numbers are unchanged.

**Slow model load.** `run_gpu.sbatch` waits 10 min for the server by default.
For a slow load (e.g. BF16 weights over NFS: job 29353 failed with `ERROR: vLLM
did not become ready in 10 min`), raise the limit:

```
READY_TIMEOUT_MIN=30 sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8 --dump-answers
```

The scripts below run in your shell, not via Slurm, so activate the venv
first (`source /work/$USER/venv-llmlci/bin/activate`) — batch jobs do this
themselves, plain `python` does not. The judge needs a server, so it runs as
its own job (command below).

Then the outer layers, on the dump that run wrote (`eval_runs/answers_<ts>.xlsx`;
see each script's `--help` for exact flags). Both need the full private eval
workbook (`--qfile`; expected answers and section locations), which is not
published; without it they stop with an ERROR.

- `python eval/audit_locations.py --dump …` → section-hit (last measured 24/29
  raw, on both the 29103 and 31898 dumps, recomputed 2026-10-24)
- `python eval/judge_answers.py --dump …` → answer-match (last 33/35, judge job
  29269 on 29103's dump; agreement with Prof. Lam's labels 29/34 = 85%, a
  different number from the judge's own 28/34 MATCH on those labels)

Example with job 31898's dump (kept in the author's private archive; replace
the timestamp with your run's):

```
python eval/audit_locations.py --dump eval_runs/answers_20260727_043415.xlsx
```

```
MODEL=/work/$USER/models/Qwen/Qwen2___5-32B-Instruct-AWQ SERVED_NAME=qwen2.5-32b-awq MAX_LEN=32768 GPU_UTIL=0.9 DTYPE=float16 EXTRA_ARGS="" sbatch run_gpu.sbatch python eval/judge_answers.py --dump eval_runs/answers_20260727_043415.xlsx --wos paper_list2_2025.csv
```

Judge guards (since 2026-10-24):

- **Titles file required.** If `--wos` (default `paper_list2_2025.csv`) is
  missing or empty it prints `ERROR: titles file ... missing/empty` and exits 1,
  because the STUDY TITLE line is part of the judge's input. `--no-titles` opts
  out on purpose and prints a WARNING (not comparable to titled runs).
- **No silent overwrite.** If `judged_<target>_<ts>.xlsx` already exists for
  that dump it prints `ERROR: ... already exists; pass --overwrite to replace
  it`. Add `--overwrite` to re-judge the same dump. (In July, jobs 28481–28484
  all wrote one filename and only the last result survived.)
- **Strict verdicts.** A `match` value that is not a boolean or
  true/yes/match/false/no counts as ERR, not "no".

**The judge is pinned.** `judge_answers.py` defaults to `qwen2.5-32b-awq` — serve
it with the BASELINE override block in `run_gpu.sbatch`'s header comment. The
challenger never grades itself.

The previous reader's baseline (job 28900) is reproducible from that same
BASELINE block.

## §8 Change protocol — before adopting ANY change

1. Re-run §7's bare-default eval on the frozen eval set.
2. Compare all four layers against the recorded baseline: gold-fed →
   gold-in-answer → section-hit → answer-match. n=50 is small — treat few-point
   moves as noise; adopt on deterministic wins (e.g. truncation → 0) or clear
   gains. Worked example in `docs/pipeline.md`: prompt v2 rejected at 47→42.
3. Append ONE dated row to `docs/pipeline.md`. Update README's numbers only after
   adoption.
4. Judge candidates never replace the referee directly — calibrate with
   `eval/compare_judges.py` against Prof. Lam's validated labels first. The
   labels are kept in the author's private archive (not published).

**Load-bearing walls.** Changing these forks the measurement history; it may be
the right call, but say so in `docs/pipeline.md` and stop citing older numbers as
comparable:

- the eval workbook and its gold labels (including `valid_dois` multi-gold)
- scoring semantics (`gold_fed` is deliberately tagged-only)
- the chunk-ID / metadata schema (`doi::…` IDs, `doi` metadata key — both the
  add_papers dedupe and the wipe rule depend on it)
- the judge referee's identity **and its prompt wording**

## §9 Known limitations (state, not bugs)

- The reader returns ONE supporting quote per paper; some assumption questions
  need several (raised by Prof. Lam, July 2026 — affected qids to be named by his
  validation review). Changing the schema ripples into the judge and the audit.
- Eval qids 31–38 have no `source_location`, so they are outside the section-hit
  denominator. `audit_locations.py` prints only 33–38 as NO_GOLD because Q31 and
  Q32 have no tagged-gold YES rows to audit.
- Table-level TASK1 ("which table") is untested in the eval, although ask.py
  answers routinely cite `Table N` locations.
- Papers added after the screened list show "(title not in list)" in locate mode
  until appended to `paper_list2_2025.csv`.
- **Legal:** `elsevier_xml/` and `tables_xml/` are Elsevier full text obtained
  under a Duke institutional entitlement. They are gitignored and must never be
  pushed to any public repository or redistributed.

## §10 Future cloud deployment (documented, not built)

vLLM ships an official image. On any CUDA cloud box:

```
docker run --gpus all -v /path/to/models:/models -p 3000:8000 \
  vllm/vllm-openai:<PINNED_VERSION> \
  --model /models/Qwen3.5-35B-A3B-FP8 <flags from run_gpu.sbatch>
```

Pin a version tag; never `latest`. Docker is **not** usable on DKUCC (Slurm, no
root) — this path is for cloud GPUs only.

## Rehearsal record

Executed 2026-07-27/28 · Mac (M4 Air) + DKUCC L20.

- **§2–§4, Mac:** fresh clone into a new directory, environment built from
  `requirements-mac.lock`, indexes rsynced from the working copy. The locate
  question returned the correct top-3 (resconrec.108220 / cesys.100297 /
  cej.158905).
- **§7, cluster (job 31898):** 47/50 = 94% (TASK2 36/38 · TASK1 11/12),
  0 truncated feeds, 31.3 s/question, locations python-verified 80/80 — matches
  the frozen baseline. Judge and audit scripts verified via `--help`.
  Correction (2026-10-24): 31898 ran the canonical cluster repo, not a fresh
  checkout; only job 31964 ran from the fresh clone. 31898 scored with
  multi-gold on 12 questions (job 29103: 2).
- **Deviations found and fixed:**
  1. §4 promised a `vector store: N chunks` line that `ask.py` never printed —
     `preflight_index` now prints it.
  2. `run_gpu.sbatch` hardcoded the canonical repo path under `/work`, so a job submitted
     from another checkout silently ran the canonical repo. It now defaults to
     `SLURM_SUBMIT_DIR` and echoes the repo path; verified by job 31964 running
     from the rehearsal clone.
  3. §7's eval scripts need the venv activated manually — noted in §7.
     User-specific paths, the `/work` purge clock, and the missing GitHub remote
     on the cluster checkout are noted in §1.
- **Not re-executed:** cluster venv creation from lock (existing venv reused),
  §3c corpus download, §3d model download.

Executed 2026-10-24 · Mac (MacBook Air M4, 16 GB) only; no cluster runs.

- **§2, Mac:** fresh `git clone` into a new directory. The existing
  `llm-lci` env had lost `libtorch_cpu.dylib`, so `import torch` failed; fixed
  with `pip install --force-reinstall --no-deps torch==2.12.0` (no other
  package changed). The torch check is now in §2.
- **§3b, Mac:** corpus restored from an existing copy (`elsevier_xml/` 1,959 XMLs,
  `tables_xml/` 1,824 xlsx); indexes built with no wipe needed. Counts:
  1,959 files, 0 errors, 280,253 prose chunks; 1,965 titles and 9,994 table records from 1,824 papers (280,253 → 290,247); BM25 290,247 chunks, all exact. Wall time: about 38 min awake (prose ingest about 2,155 s, tables about 100 s, BM25 about 30 s; the raw 4,939 s includes a 2,652 s lid-closed sleep) (lid open, on power; a about 44 min
  lid-closed pause excluded).
- **§4, Mac:** smoke test `vector store: 290,247 chunks`, top 3 resconrec.2025.108220 / cesys.2025.100297 / cej.2024.158905, as expected. Retrieval eval on the fresh index:
  Hit@10 46/50 = 92.0%, MRR 0.671, identical for the 28 Jul code with the documented flags, the 2026-10-24 code with bare defaults, and the 2026-10-24 code at candidate_k 100.
- **Deviations found and fixed:** §3b "takes hours" replaced by measured times;
  XML count needs `*.xml` (the download log sits in the same folder); ingest
  now exits 1 on file errors instead of exiting 0 with `chunks added: 0`.
- **Found, not fixed:** the §3c download command (see §3c; its `KeyError` was
  fixed 2026-10-25).
