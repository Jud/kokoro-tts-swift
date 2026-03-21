#!/usr/bin/env python3
"""Generate vanilla PyTorch reference audio — no patches, no padding.

Called as subprocess by stage_harness.py. Writes WAV files to output dir.

Usage:
    python gen_reference.py --output-dir /tmp/ref --voice af_heart --text "Hello world."
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--sentences-json", required=True,
                        help="JSON dict of {label: text}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences = json.loads(args.sentences_json)

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")

    for label, text in sentences.items():
        chunks = []
        for _, _, audio in pipeline(text, voice=args.voice, speed=1.0):
            chunks.append(audio.cpu().numpy())
        audio = np.concatenate(chunks)
        path = os.path.join(args.output_dir, f"{label}.wav")
        sf.write(path, audio, 24000)
        print(f"{label}: {len(audio)} samples ({len(audio)/24000:.1f}s)")


if __name__ == "__main__":
    main()
