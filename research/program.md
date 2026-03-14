# kokoro-tts-swift CoreML Research

## Setup

1. **Agree on a run tag** based on today's date (e.g. `mar14`). Branch: `research/<tag>`.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - `scripts/stage_harness.py` — **READ-ONLY**.
   - `scripts/export_coreml.py` — the file you modify. Contains inlined `CustomSTFT` and `SineGen` that the harness imports directly.
   - `research/program.md` — this file.
4. **Verify environment**: `.venv/bin/python scripts/stage_harness.py` should run all stages.
5. **Establish baseline**: first run is always the harness as-is. Record the results.
6. **Confirm and go**.

## The harness

9 targets: 7 individual stages + 2 split models (split_A = stages 1-5, split_B = stage 6). 3 test sentences (short/medium/long), reports worst-case.

Per target: CPU Corr, CPU Cold/Warm, ANE Corr, ANE Cold/Warm. PASS if ANE Corr > 0.99.

Short sentence has ~0.02 noise from random phases. If medium/long PASS but short is WARN, the experiment is fine.

```bash
.venv/bin/python scripts/stage_harness.py
.venv/bin/python scripts/stage_harness.py --stage 6
.venv/bin/python scripts/stage_harness.py --json
```

## The goal

**Get the highest worst-case ANE Corr across all stages.**

Everything in `scripts/export_coreml.py` is fair game. All stages must stay PASS (ANE Corr > 0.99).

All else being equal, simpler is better.

## Architecture

`export_coreml.py` contains inlined `CustomSTFT` and `SineGen` from `.venv/kokoro/`. The harness imports these directly — modifications are traced natively. Do not monkey-patch — it breaks ANE tracing.

## Constraints

- Do not modify `scripts/stage_harness.py`.
- Do not modify files in `.venv/`.
- Do not install new packages.

## Logging results (MANDATORY)

**Log every experiment to `research/results.tsv`** (tab-separated). Do NOT commit this file.

```
commit	stage	cpu_corr	ane_corr	cpu_warm_ms	ane_warm_ms	status	description
```

- One row per stage per experiment.
- status: `keep`, `discard`, or `crash`
- Never skip logging.

## The experiment loop

LOOP FOREVER:

1. Look at the harness output. What's the bottleneck?
2. Form a hypothesis.
3. Modify `scripts/export_coreml.py`.
4. git commit.
5. Run: `.venv/bin/python scripts/stage_harness.py 2>&1 | tee research/run.log`
6. Record in results.tsv.
7. If correlation improved or maintained without regression: keep.
8. If worse: `git reset --hard HEAD~1`.

If a run crashes, fix trivial bugs and re-run, or skip the idea and move on.

**NEVER STOP**: do not pause to ask the human. Continue indefinitely until manually stopped.
