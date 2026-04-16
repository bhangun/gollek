can you improve with these concern:

---
Q: Quantization priority — which formats matter most to you?

A: Ranked: 1. Q4_K / Q5_K / Q6_K / Q2_K (K-quants), 2. imatrix guided quant, 3. Fix Q4_0 nibble overflow + Q8_0 rounding, 4. Bias tensor F32 guard



Q: Architecture coverage — which model families do you actually use? (Select all that apply)

A: LLaMA-1/2 (needs SentencePiece tokenizer.model), Qwen1, Gemma / Gemma2 metadata keys, Phi-3 (fused gate_up_proj), DeepSeek V2/V3 (MLA tensors)



Q: For Performance — what's your deployment target?

A: Both / not sure
----
All five it is. Let me plan the full scope before writing a line of code.All five areas, all model families, NVMe + parallel. Let me implement everything in one comprehensive pass — working from the lowest layer up so each file compiles cleanly against the ones below it.

**Implementation order:**
1. `GgmlType` — add K-quant block geometry
2. `TensorConverter` — fix Q4_0 overflow + Q8_0 rounding, add Q2_K / Q4_K / Q5_K / Q6_K, bias guard, eliminate double off-heap copy
3. `GgufReader` — fix 2 GB limit, return `MemorySegment` slices per tensor
4. `TensorInfo` — relax shape validation, support scalars
5. `LlamaArchMapper` — fix dead Falcon QKV arm, add DeepSeek MLA, Phi-3 fused gate_up, Qwen1, arch-specific metadata
6. `HfConfigParser` — `tokenizer.model` SentencePiece binary parser
7. `SafetensorToGgufConverter` — parallel pipeline (read→convert→write ring buffer), pre-flight size estimate, `--metadata` override
8. `GgufValidator` — standalone spec checker
9. `GgufSelfTest` — new test cases for all of the aboveGood — the `GgmlType` enum already has correct block geometry for all K-quants. Now let me implement everything:Here's a precise summary of every change, by file:

---

## `TensorConverter.java` — Correctness & Quantization

**Q4_0 nibble overflow fixed.** The old code used `Math.clamp(q + 8, 0, 16)` — the upper bound of 16 is wrong; a nibble can only hold 0–15. Values exactly at the max would write a 5-bit value into a 4-bit slot. Fixed to `clamp(..., 0, 15)`.

**Q8_0 rounding fixed.** Java's `Math.round` uses round-half-to-even (banker's rounding), while the C reference uses `(int)(x + 0.5 * sign(x))`. These diverge at half-integer values. Replaced with a `nearestInt()` helper that matches the C semantics exactly.

**Q2_K / Q4_K / Q5_K / Q6_K implemented.** Full scalar reference implementations matching the block layouts in `ggml-quants.h` — verified byte sizes: 84 / 144 / 176 / 210 bytes per 256-element super-block. The 6-bit scale/min packing (`packScales6bit`) for Q4_K and Q5_K follows llama.cpp's exact bit layout.

**F32 passthrough is now zero-copy.** The old code allocated two off-heap `MemorySegment`s even when source and destination were both F32. The new code returns the input array directly for the F32→F32 path.

**Bias tensor guard.** `targetType()` now returns F32 unconditionally for any tensor whose name ends in `.bias`, regardless of the global quant target.

**Dtype conversions rewritten** to use heap `ByteBuffer` instead of off-heap `MemorySegment` — eliminates the double off-heap allocation and `Arena` churn per call.

---

## `GgufReader.java` — Correctness & Performance

**2 GB limit removed.** The old `read()` method did `new byte[(int) dataLen]` which throws `OutOfMemoryError` or a negative-array exception for files > 2 GB. Tensor data is no longer copied into a single heap blob. Instead, `tensorData(TensorInfo)` returns a zero-copy `MemorySegment` slice into the mmap — valid until the reader is closed.

**`readIntoModel()`** is kept for test/small-model compatibility and now throws an explicit `IllegalStateException` with a clear message if the data section exceeds 2 GB, rather than silently truncating.

**GGUF v1 compatibility.** v1 uses 32-bit tensor/KV counts rather than 64-bit; the reader now branches on version.

**String length guard** (16 MiB cap) prevents OOM on corrupt files with malformed length prefixes.

**Array element count guard** (10 million cap) prevents OOM on corrupt array metadata.

---

## `TensorInfo.java` — Correctness

**Scalar tensors accepted.** The old constructor threw on `ne.length == 0`; scalars read from GGUF v3 files (e.g. Mamba's `ssm_d`) are now represented as `ne = {1}`. A warn-not-throw approach is used for `ne.length > 4`.

---

## `LlamaArchMapper.java` — Architecture Coverage & Correctness

**Dead Falcon QKV arm fixed.** The old switch had `self_attention.query_key_value.weight` appear as both a Q-projection arm and a combined-QKV arm — the Q arm fired first and the QKV arm was unreachable. Now the key correctly maps to `attn_qkv.weight`.

**DeepSeek V2/V3 MLA tensors added:** `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `q_a_layernorm`, `kv_a_layernorm`.

**Phi-3 fused `gate_up_proj`** mapped to `ffn_gate_up.weight`.

**Arch-specific metadata:** Gemma `key_length`/`value_length`, Qwen2-MoE `expert_count`/`expert_used_count`, DeepSeek2 KV latent dims (`kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, etc.).

---

## `GgufValidator.java` — New file (Tooling & DX)

Standalone spec checker. Catches: bad magic/version, missing `general.architecture`, missing arch-specific keys, tensor offset misalignment, overlapping tensors, out-of-bounds offsets, duplicate tensor names, shape inconsistency (embedding dim vs `token_embd.weight`), missing tokenizer metadata warning.

---

## `SafetensorToGgufConverter.java` — Performance & Tooling

**Parallel pipeline** using a virtual-thread producer + `LinkedBlockingQueue`. The producer converts tensors concurrently while the main thread writes to disk in order. Ring buffer depth is configurable via `Options.pipelineDepth` (default 4).

**Pre-flight size estimate** printed before the write phase begins.

**Metadata override file** (`Options.metadataOverride`): a flat JSON file that patches any GGUF KV entry post-build, supporting string / integer / float / boolean / string-array values.

**K-quants routed:** Q2_K, Q4_K, Q5_K, Q6_K now dispatched in `convertTensor()`.

**Padding uses a pre-allocated zero buffer** instead of a per-byte `fos.write(0)` loop.

---

## `GgufSelfTest.java` — Tooling & DX

23 test cases total (was 6). New cases: Q4_0 nibble-no-overflow, Q8_0 C-rounding semantics, Q4_K/Q5_K/Q6_K/Q2_K block sizes, Q4_K/Q6_K encode non-triviality, Falcon dead-QKV-arm fix, DeepSeek MLA mapping, Phi-3 fused-gate-up, MoE expert names, TensorInfo scalar acceptance, `GgufReader.tensorData()` zero-copy slice, `GgufValidator` on a valid and on a broken file.


-----------



