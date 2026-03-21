# kokoro-coreml research

This is an experiment to have the LLM optimize CoreML model export fidelity.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `research/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `scripts/stage_harness.py` — fixed evaluation harness. Do not modify.
   - `scripts/export_coreml.py` — the file you modify. Contains CoreML export logic, pipeline split modules, ANE compatibility replacements, and patching.
   - `research/program.md` — this file.
4. **Verify environment**: `.venv/bin/python scripts/stage_harness.py` should run all stages.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment exports CoreML models for all three buckets, then compares each against the PyTorch reference using the `af_heart` voice across short, half, and filled sentence lengths. You launch it simply as: `.venv/bin/python scripts/stage_harness.py > run.log 2>&1`

**What you CAN do:**
- Modify `scripts/export_coreml.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify any file other than `scripts/export_coreml.py`. In particular, `scripts/reference.py` contains `CustomSTFT`, `SineGen`, and `patch_sinegen_for_export` which define the PyTorch reference — these are immutable.
- Install new packages or add dependencies.

**The goal is simple: get the lowest p99.9 while keeping Corr >= 0.99.** p99.9 is the 99.9th percentile of |CoreML - PyTorch| sample differences — it measures the severity of the worst recurring artifacts. Correlation measures waveform shape. A change that improves one bucket but regresses another is not acceptable.

Current baselines (af_heart, worst case across sentence lengths, compared against vanilla PyTorch):

| Bucket | Corr | p99.9 | Spk/s | Speed |
|--------|------|-------|-------|-------|
| kokoro_21_5s | 0.9968 | 0.0435 | 16 | 94ms |
| kokoro_24_10s | 0.9937 | 0.0809 | 85 | 124ms |
| kokoro_25_20s | 0.9941 | 0.0734 | 60 | 256ms |

PASS requires: Corr >= 0.99, Spk/s <= 50, Speed <= 300ms.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the harness as is.

## Output format

The harness runs all three buckets automatically and prints:

```
Test                             Corr   p99.9  Spk/s  Speed
--------------------------------------------------------------
kokoro_21_5s/short       PASS  0.9977  0.0453     16   94ms
kokoro_21_5s/half        PASS  0.9982  0.0302      1   94ms
kokoro_21_5s/filled      PASS  0.9985  0.0287      2   94ms
kokoro_21_5s/WORST       PASS  0.9977  0.0453     16   96ms
kokoro_24_10s/short      PASS  0.9966  0.0569     43  124ms
...
kokoro_25_20s/WORST      FAIL  0.9948  0.0834     65  254ms
```

Extract the key results:

```
grep "WORST" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

```
commit	bucket	corr	p999	spike_rate	speed_ms	status	description
```

- One row per bucket per experiment (the WORST case across sentence lengths).
- status: `keep`, `discard`, or `crash`

## The experiment loop

The experiment runs on a dedicated branch (e.g. `research/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `scripts/export_coreml.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `.venv/bin/python scripts/stage_harness.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "WORST" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If p99.9 improved (lower) without correlation regression (Corr stays >= 0.99): keep.
9. If p99.9 is equal or worse, or correlation dropped: `git reset --hard HEAD~1`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the ANE reference guide (https://github.com/hollance/neural-engine), re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
