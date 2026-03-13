# Kokoro TTS: Reference Comparison

Comparison of our CoreML Swift implementation against the reference Python
implementation at [hexgrad/Kokoro-TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS).

**Date:** 2026-03-13

---

## Bugs Found & Fixed

### Session 1: Core Fixes (TTS→STT roundtrip 3/10 → 10/10)

| Bug | Impact | Fix |
|-----|--------|-----|
| BOS/EOS tokens were 1,2 instead of 0,0 | Garbled short utterances — model saw phoneme symbols `$`,`^` instead of boundary markers | `Tokenizer.swift`: bosId=0, eosId=0 |
| Generic style vector used for all inputs | Wrong prosody — cosine similarity between generic and short-input vector was only 0.47 | `VoiceStore.swift`: load length-indexed embeddings; `KokoroEngine.swift`: call `embedding(for:tokenCount:)` per sentence |
| Short inputs routed to 242-token bucket | 97% padding for 5-token inputs | `ModelBucket.select`: removed minTokens gate, route to smallest fitting bucket |

### Session 2: Reference Alignment

| ID | Bug | Fix |
|----|-----|-----|
| D10 | Voice embedding off-by-2: `tokenIds.count` includes BOS+EOS but VoiceStore expects phoneme count | `KokoroEngine.swift:223`: pass `tokenCount - 2` |
| D3 | Text-level sentence splitting loses cross-sentence phonetic context | Phonemize-then-chunk pipeline: phonemize full text first, chunk phoneme stream at punctuation using waterfall search (`!.?…` → `:;` → `,—` → space) |
| D3b | Hard truncation in Tokenizer at max length | Backward punctuation search for cleaner truncation boundary |
| D8 | CamelCase fallback for OOV words instead of BART neural G2P | Added 752K-param BART model (d_model=128, 1 enc/dec layer). Pure Accelerate inference, 3 MB safetensors weights from `PeterReid/graphemes_to_phonemes_en_us` |

---

## Remaining Discrepancies

### D1: Bundled Vocabulary — FALSE ALARM

The bundled `vocab_index.json` was verified correct (Kokoro native vocabulary,
114 entries). It uses the nested `{"vocab": {...}}` format which the Tokenizer
handles since the initial fix session.

### D2: hopSize — CORRECT at 600

`hopSize = 600` is correct for our end-to-end CoreML models. The reference
PyTorch model has Decoder 2x upsampling × Generator 300x = 600 total. The
earlier claim of hopSize=300 was based on the Generator alone (ignoring the
Decoder's 2x contribution). Verified by `audio_length_samples` model output.

### D4: Post-Processing Not in Reference

**Severity: LOW — intentional quality additions**

Our `postProcess()` applies HP filter, presence EQ, and normalization not in
the reference. These are intentional for voice assistant quality. The
`rawAudio` parameter allows bypassing for A/B testing.

### D7: Max Token Limit (242 vs 510)

**Severity: LOW — inherent to CoreML model conversion**

Our largest bucket supports 242 tokens vs reference 510. Hard physical limits
baked into compiled CoreML models. With phoneme-level chunking (D3 fix),
the impact is minimized.

### D9: ɾ→T Replacement — FALSE ALARM

The `ɾ→T` replacement in `EnglishG2P.swift` matches the reference. The misaki
G2P library performs this upstream in its phonemization pipeline. Our
implementation correctly mirrors this behavior.

---

## Verified Correct (Matches Reference)

- BOS/EOS tokens = 0
- Style vector selection: `pack[len(ps)-1]` with correct phoneme count (minus BOS/EOS)
- Attention mask polarity: 1=valid, 0=padded
- Phoneme replacements: `ɾ→T`, `ʔ→t`
- Unknown phoneme characters silently dropped
- Random phases externalized as `[1,9]` input (CoreML conversion artifact)
- Speed injection via graph modification (mathematically equivalent)
- Phonemize-then-chunk pipeline (phonemize full text, chunk phoneme stream)
- BART neural G2P for OOV words (deterministic greedy decoding matches Python)
- hopSize = 600 (Decoder 2x × Generator 300x)

---

## Model Inventory

Verified from `model.mil` and `metadata.json` on disk at
`~/Library/Application Support/com.operator/models/kokoro/`:

| Model | Tokens | Audio Buffer | Speed | iOS | Status |
|-------|--------|-------------|-------|-----|--------|
| `kokoro_21_5s` | 124 | 175,800 (7.3s) | Yes | 16+ | **Active** (small) |
| `kokoro_24_10s` | 242 | 240,000 (10.0s) | No¹ | 17+ | **Active** (medium) |
| `kokoro_21_10s` | 249 | 372,600 (15.5s) | Yes | 16+ | **On disk, unused** |

¹ Speed input not detected at runtime; model uses `pred_dur` output instead of
`pred_dur_clamped`. Original model without speed injection.

Reference PyTorch model: 510 tokens max (512 `max_position_embeddings` − 2).
