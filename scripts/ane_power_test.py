#!/usr/bin/env python3
"""Test whether Kokoro CoreML models actually run on the ANE.

Loads the backend model with compute_units=ALL, runs inference in a loop
while sudo powermetrics captures ANE vs GPU power draw.

Usage:
    sudo .venv/bin/python scripts/ane_power_test.py
"""
import os
import signal
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models_export")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import coremltools as ct


def load_model(path, compute_units):
    print(f"  Loading {os.path.basename(path)} with {compute_units}...")
    return ct.models.MLModel(path, compute_units=compute_units)


def make_dummy_inputs(fe_model, be_model):
    """Run frontend to get realistic shaped inputs for backend."""
    n_tokens = 20  # short sentence
    fe_out = fe_model.predict({
        "input_ids": np.zeros((1, n_tokens), dtype=np.int32),
        "attention_mask": np.ones((1, n_tokens), dtype=np.int32),
        "ref_s": np.random.randn(1, 256).astype(np.float32),
        "speed": np.array([1.0], dtype=np.float32),
        "random_phases": np.zeros((1, 9), dtype=np.float32),
    })
    return {
        'asr': fe_out['asr'].astype(np.float32),
        'F0_pred': fe_out['F0_pred'].astype(np.float32),
        'N_pred': fe_out['N_pred'].astype(np.float32),
        's_content': np.random.randn(1, 128).astype(np.float32),
        'har': fe_out['har'].astype(np.float32),
    }


def run_power_test(be_model, inputs, n_iters=20, label=""):
    """Run backend inference in a loop, return timing."""
    # Warmup
    be_model.predict(inputs)
    be_model.predict(inputs)

    print(f"\n  [{label}] Running {n_iters} iterations...")
    times = []
    for i in range(n_iters):
        t0 = time.time()
        be_model.predict(inputs)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"  [{label}] Avg: {avg_ms:.1f}ms per inference")
    return avg_ms


def capture_powermetrics(duration_s, output_file):
    """Start powermetrics capture, return process."""
    cmd = [
        "sudo", "powermetrics",
        "--samplers", "gpu_power,ane_power",
        "--sample-rate", "500",  # 500ms intervals
        "-n", str(int(duration_s * 2)),  # number of samples
        "-o", output_file,
    ]
    print(f"  Starting powermetrics ({duration_s}s capture)...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)  # let it start
    return proc


def parse_powermetrics(output_file):
    """Parse powermetrics output for ANE and GPU power."""
    try:
        with open(output_file) as f:
            text = f.read()
    except FileNotFoundError:
        return None

    ane_powers = []
    gpu_powers = []

    for line in text.split('\n'):
        line_lower = line.lower().strip()
        # ANE power
        if 'ane power' in line_lower and 'mw' in line_lower:
            try:
                val = float(line_lower.split(':')[-1].strip().replace('mw', '').strip())
                ane_powers.append(val)
            except ValueError:
                pass
        # GPU power
        if 'gpu power' in line_lower and 'mw' in line_lower:
            try:
                val = float(line_lower.split(':')[-1].strip().replace('mw', '').strip())
                gpu_powers.append(val)
            except ValueError:
                pass

    return {
        'ane_avg_mw': np.mean(ane_powers) if ane_powers else 0,
        'gpu_avg_mw': np.mean(gpu_powers) if gpu_powers else 0,
        'ane_samples': len(ane_powers),
        'gpu_samples': len(gpu_powers),
        'raw_text': text[:2000],
    }


def main():
    if os.geteuid() != 0:
        print("This script needs sudo for powermetrics.")
        print("Run: sudo .venv/bin/python scripts/ane_power_test.py")
        sys.exit(1)

    fe_path = os.path.join(MODELS_DIR, "kokoro_frontend.mlpackage")
    be_path = os.path.join(MODELS_DIR, "kokoro_backend.mlpackage")

    if not os.path.exists(fe_path) or not os.path.exists(be_path):
        print(f"Models not found in {MODELS_DIR}")
        sys.exit(1)

    # Load frontend (CPU only, just to generate inputs)
    fe = load_model(fe_path, ct.ComputeUnit.CPU_ONLY)
    inputs = make_dummy_inputs(fe, None)
    del fe  # free memory

    configs = [
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ]

    results = {}
    for label, units in configs:
        print(f"\n{'='*60}")
        print(f"Testing compute_units={label}")
        print(f"{'='*60}")

        be = load_model(be_path, units)

        # Warmup outside measurement
        be.predict(inputs)
        be.predict(inputs)
        time.sleep(1)

        # Capture power during inference
        power_file = f"/tmp/kokoro_power_{label}.txt"
        power_proc = capture_powermetrics(15, power_file)

        time.sleep(1)  # baseline
        avg_ms = run_power_test(be, inputs, n_iters=30, label=label)
        time.sleep(1)  # cooldown

        # Wait for powermetrics to finish
        try:
            power_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            power_proc.terminate()
            power_proc.wait()

        power = parse_powermetrics(power_file)
        results[label] = {'avg_ms': avg_ms, 'power': power}

        del be
        time.sleep(2)  # let hardware settle

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<16} {'Avg ms':>8} {'ANE mW':>10} {'GPU mW':>10} {'Likely HW':>12}")
    print(f"{'-'*60}")
    for label, r in results.items():
        p = r['power']
        if p:
            ane = p['ane_avg_mw']
            gpu = p['gpu_avg_mw']
            hw = "ANE" if ane > gpu * 0.1 and ane > 50 else "GPU" if gpu > 50 else "CPU?"
            print(f"{label:<16} {r['avg_ms']:>7.1f} {ane:>9.0f} {gpu:>9.0f} {hw:>12}")
        else:
            print(f"{label:<16} {r['avg_ms']:>7.1f} {'N/A':>10} {'N/A':>10} {'???':>12}")

    # Print raw powermetrics excerpt for manual inspection
    print(f"\n{'='*60}")
    print("RAW POWERMETRICS (first config, first 2000 chars)")
    print(f"{'='*60}")
    first = list(results.values())[0]
    if first['power']:
        print(first['power']['raw_text'])


if __name__ == "__main__":
    main()
