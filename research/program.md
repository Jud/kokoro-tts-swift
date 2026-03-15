# kokoro-tts-swift CoreML Research

## Setup

1. **Agree on a run tag** based on today's date (e.g. `mar14`). Branch: `research/<tag>`.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - `scripts/stage_harness.py`
   - `scripts/export_coreml.py` — the file you modify. Contains inlined `CustomSTFT` and `SineGen`.
   - `research/program.md` — this file.
4. **Verify environment**: `.venv/bin/python scripts/stage_harness.py` should run all stages.
5. **Establish baseline**: first run is always the harness as-is. Record the results.
6. **Confirm and go**.

## The harness

11 targets: 7 individual stages + 2 split models + 2 pipeline splits. 3 test sentences (short/medium/long), reports worst-case.

Per target: CPU Corr, CPU Cold/Warm, ANE Corr, ANE Cold/Warm. PASS if ANE Corr > 0.99.

Pipeline stages (10, 11) test two-model splits: frontend on CPU_ONLY, backend on ALL.

Audio correlation is measured on active samples only (ignores fixed-size padding).

```bash
.venv/bin/python scripts/stage_harness.py
.venv/bin/python scripts/stage_harness.py --stage 6
.venv/bin/python scripts/stage_harness.py --stage 10  # generator pipeline
.venv/bin/python scripts/stage_harness.py --stage 11  # decoder pipeline
.venv/bin/python scripts/stage_harness.py --json
```

## The goal

Maximize ANE Corr and minimize ANE latency across all stages. All stages must stay PASS (ANE Corr > 0.99).

Everything in `scripts/export_coreml.py` is fair game. `scripts/stage_harness.py` is **READ-ONLY**.

All else being equal, simpler is better.

## Architecture

`export_coreml.py` contains inlined `CustomSTFT` and `SineGen` from `.venv/kokoro/`. The harness imports these directly — modifications are traced natively.

Pipeline split modules (`GeneratorFrontEnd`, `GeneratorBackEnd`, `DecoderBackEnd`) enable testing CPU+ANE two-model pipelines.

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
- Run multiple times to verify — single-run results are unreliable due to phase-dependent variance.

## The experiment loop

LOOP FOREVER:

1. Look at the harness output. What's the bottleneck?
2. Form a hypothesis.
3. Modify `scripts/export_coreml.py`.
4. git commit.
5. Run: `.venv/bin/python scripts/stage_harness.py 2>&1 | tee research/run.log`
6. Record in results.tsv.
7. If improved without regression: keep.
8. If worse: `git reset --hard HEAD~1`.

If a run crashes, fix trivial bugs and re-run, or skip the idea and move on.

**NEVER STOP**: do not pause to ask the human. Continue indefinitely until manually stopped.
