#!/usr/bin/env python3
"""Verify that the large bucket fixes audio truncation from medium bucket overflow.

Demonstrates:
1. Medium bucket truncates tokens and loses audio for long text
2. Large bucket preserves all tokens and audio

Then runs full CoreML correlation test via verify_models.py approach.

Usage:
    .venv/bin/python scripts/verify_large_bucket.py
"""
import os
import sys
import threading

import numpy as np
import torch
import torch.nn as nn

nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)

import coremltools as ct
from coreml_ops import register_missing_torch_ops
register_missing_torch_ops()

sys.path.insert(0, os.path.dirname(__file__))
from export_coreml import BUCKETS, SAMPLES_PER_FRAME, patch_sinegen_for_export, CustomSTFT

# Text that overflows medium bucket (248 tokens > 242 max)
OVERFLOW_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, and the shells she sells are surely seashells. "
    "In a hole in the ground there lived a hobbit, not a nasty dirty wet hole "
    "filled with the ends of worms and an oozy smell."
)


def load_model():
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    model.eval()
    gen = model.decoder.generator
    gen.stft = CustomSTFT(
        filter_length=gen.stft.filter_length,
        hop_length=gen.stft.hop_length,
        win_length=gen.stft.win_length,
    )
    set_phases_fn = patch_sinegen_for_export(model)
    return pipeline, model, set_phases_fn


def tokenize(text, pipeline, model):
    phonemes, _ = pipeline.g2p(text)
    raw = list(filter(lambda i: i is not None,
        map(lambda p: model.vocab.get(p), phonemes)))
    return [0] + raw + [0]


def verify_bucket(pipeline, model, set_phases_fn, bucket_name, bucket_config,
                  token_ids, voice="af_heart"):
    """Adapted from verify_models.py — runs end-to-end CoreML vs PyTorch comparison."""
    max_tokens = bucket_config["max_tokens"]
    max_audio = bucket_config["max_audio"]
    max_frames = max_audio // SAMPLES_PER_FRAME

    seq_len = min(len(token_ids), max_tokens)
    truncated = len(token_ids) > max_tokens
    tokens_lost = max(0, len(token_ids) - max_tokens)

    # Prepare inputs
    ids = torch.zeros(1, max_tokens, dtype=torch.long)
    ids[0, :seq_len] = torch.tensor(token_ids[:seq_len])
    mask = torch.zeros(1, max_tokens, dtype=torch.long)
    mask[0, :seq_len] = 1
    speed = torch.tensor([1.0])

    voice_pack = pipeline.load_voice(voice)
    style = voice_pack[seq_len]
    if style.dim() == 1:
        style = style.unsqueeze(0)

    phases = torch.zeros(1, 9)
    set_phases_fn(model.decoder, phases)

    ids_np = ids.numpy().astype(np.int32)
    mask_np = mask.numpy().astype(np.int32)

    # PyTorch reference
    with torch.no_grad():
        input_lengths = mask.sum(dim=1).long()
        text_mask = (mask == 0)

        bert_dur = model.bert(ids, attention_mask=mask)
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = style[:, 128:]
        s_content = style[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * mask.long()

        import torch.nn.functional as F
        cumsum = torch.cumsum(pred_dur, dim=-1)
        total_frames = int(cumsum[0, -1].item())
        frame_indices = torch.arange(max_frames).unsqueeze(0)
        starts = F.pad(cumsum[:, :-1], (1, 0))
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model.decoder(asr, F0_pred, N_pred, s_content)

    py_audio_len = total_frames * SAMPLES_PER_FRAME
    py_audio = audio.squeeze().numpy()[:py_audio_len]

    # CoreML — use pre-exported models if available, otherwise skip CoreML comparison
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_export")
    fe_path = os.path.join(base, f"{bucket_name}_frontend.mlpackage")
    be_path = os.path.join(base, f"{bucket_name}_backend.mlpackage")

    if not os.path.exists(fe_path) or not os.path.exists(be_path):
        return {
            "bucket": bucket_name,
            "total_tokens": len(token_ids),
            "tokens_used": seq_len,
            "tokens_lost": tokens_lost,
            "truncated": truncated,
            "py_audio_len": py_audio_len,
            "py_peak": float(np.max(np.abs(py_audio))),
            "buffer_size": max_audio,
            "buffer_overflow": py_audio_len > max_audio,
            "samples_lost": max(0, py_audio_len - max_audio),
            "coreml": False,
        }

    # Run CoreML frontend on separate thread
    coreml_result = {}

    def run_frontend():
        fe = ct.models.MLModel(fe_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        coreml_result["fe_out"] = fe.predict({
            "input_ids": ids_np, "attention_mask": mask_np,
            "ref_s": style.numpy().astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
            "random_phases": phases.numpy().astype(np.float32),
        })

    t1 = threading.Thread(target=run_frontend)
    t1.start()
    t1.join()
    fe_out = coreml_result["fe_out"]

    def run_backend():
        be = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.ALL)
        coreml_result["be_out"] = be.predict({
            "asr": fe_out["asr"].astype(np.float32),
            "F0_pred": fe_out["F0_pred"].astype(np.float32),
            "N_pred": fe_out["N_pred"].astype(np.float32),
            "s_content": style[:, :128].numpy().astype(np.float32),
            "har": fe_out["har"].astype(np.float32),
        })

    t2 = threading.Thread(target=run_backend)
    t2.start()
    t2.join()
    be_out = coreml_result["be_out"]

    cm_audio_len = int(fe_out["audio_length_samples"].flatten()[0])
    cm_audio = be_out["audio"].flatten()[:min(cm_audio_len, max_audio)]

    # Correlation (PyTorch vs CoreML, trimmed to active region)
    ml = min(len(py_audio), len(cm_audio))
    py_f = py_audio[:ml].astype(np.float64)
    cm_f = cm_audio[:ml].astype(np.float64)
    corr = float(np.corrcoef(py_f, cm_f)[0, 1]) if np.std(py_f) > 1e-10 and np.std(cm_f) > 1e-10 else 0.0

    return {
        "bucket": bucket_name,
        "total_tokens": len(token_ids),
        "tokens_used": seq_len,
        "tokens_lost": tokens_lost,
        "truncated": truncated,
        "py_audio_len": py_audio_len,
        "py_peak": float(np.max(np.abs(py_audio))),
        "cm_audio_len": cm_audio_len,
        "cm_actual_len": len(cm_audio),
        "cm_peak": float(np.max(np.abs(cm_audio))),
        "buffer_size": max_audio,
        "buffer_overflow": cm_audio_len > max_audio,
        "samples_lost": max(0, cm_audio_len - max_audio),
        "correlation": corr,
        "coreml": True,
    }


def main():
    print("Loading model...")
    pipeline, model, set_phases_fn = load_model()

    token_ids = tokenize(OVERFLOW_TEXT, pipeline, model)
    print(f"\nText: \"{OVERFLOW_TEXT[:80]}...\"")
    print(f"Tokens: {len(token_ids)} (medium max: 242, large max: 510)\n")

    for bucket_name in ["kokoro_24_10s", "kokoro_25_20s"]:
        bucket = BUCKETS[bucket_name]
        print(f"{'=' * 70}")
        print(f"  {bucket_name} (max_tokens={bucket['max_tokens']}, "
              f"max_audio={bucket['max_audio']})")
        print(f"{'=' * 70}")

        r = verify_bucket(pipeline, model, set_phases_fn, bucket_name, bucket, token_ids)

        status_tok = f"({r['tokens_lost']} LOST)" if r['truncated'] else "(all preserved)"
        print(f"  Tokens:    {r['total_tokens']} -> {r['tokens_used']} {status_tok}")
        print(f"  PyTorch:   {r['py_audio_len']} samples ({r['py_audio_len']/24000:.1f}s), "
              f"peak={r['py_peak']:.3f}")

        if r['buffer_overflow']:
            print(f"  OVERFLOW:  predicted {r.get('cm_audio_len', r['py_audio_len'])} samples, "
                  f"buffer {r['buffer_size']} -> {r['samples_lost']} samples "
                  f"({r['samples_lost']/24000:.1f}s) LOST")
        else:
            headroom = r['buffer_size'] - r.get('cm_audio_len', r['py_audio_len'])
            print(f"  Buffer:    OK (headroom: {headroom} samples, {headroom/24000:.1f}s)")

        if r['coreml']:
            print(f"  CoreML:    {r['cm_actual_len']} samples ({r['cm_actual_len']/24000:.1f}s), "
                  f"peak={r['cm_peak']:.3f}")
            print(f"  Corr:      {r['correlation']:.4f}")
        else:
            print(f"  CoreML:    (exported models not found, skipping CoreML test)")

        verdict = "PASS" if not r['truncated'] and not r['buffer_overflow'] else "FAIL"
        if r['coreml'] and r['correlation'] < 0.99:
            verdict = "FAIL (low correlation)"
        print(f"  Verdict:   {verdict}")
        print()


if __name__ == "__main__":
    main()
