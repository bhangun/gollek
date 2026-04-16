Now let me fetch the actual `convert_hf_to_gguf.py` from llama.cpp to understand the real bindings:Now I have a thorough understanding of the codebase and the llama.cpp conversion architecture. Let me build all the improvements and add `SafetensorToGgufConverter`. Here's the plan:

**Improvements to existing code:**
1. `LlamaArchMapper` — add all missing tensor mappings (QKV combined for many models, MLP variants, Qwen/Gemma/Phi-specific tensors, attention bias patterns, tied embeddings handling)
2. `HfToGgufConverter` — fix the two-pass offset recompute bug (tensor type can change F32 fallback but descriptor already registered), add proper shard index file support (`model.safetensors.index.json`), streaming blob write instead of one giant heap array
3. `SafetensorsReader` — add `tensorSegment` zero-copy path in the converter, handle the `__metadata__` skip properly
4. `HfConfigParser` — add `head_dim`, `sliding_window`, `rope_scaling`, `attention_bias`, `partial_rotary_factor` fields
5. **New** `SafetensorToGgufConverter` — high-level API that wraps everything with proper shard discovery via the index JSON, progress callbacks, streaming outputNow copy all files to outputs:Here's a summary of everything delivered and what changed:

---

## What was delivered

### New: `SafetensorToGgufConverter.java`
The real high-level converter with actual bindings, not stubs:

- **`Options.Builder` API** — fluent builder with `inputDir`, `outputFile`, `quantType`, `modelVersion`, `verbose`, and a `BiConsumer<Integer,Integer> onProgress` progress callback
- **`model.safetensors.index.json` support** — reads the HuggingFace shard-weight index directly instead of scanning every shard, which is critical for sharded 70B+ models
- **Streaming disk write** — serializes the GGUF header to a temp file, then streams each tensor directly from its mmap'd shard to the output file; heap usage is bounded by the *largest single tensor*, not the whole model
- **F64→F32 conversion** — covers models that store weights in float64
- **`generation_config.json` support** — picks up `pad_token_id` and `chat_template` metadata
- **Block-alignment validation** — catches element-count mismatches before quantization and falls back to F32 with a warning instead of crashing

---

### Improved: `LlamaArchMapper.java`
Tensor name mapping rewritten from the actual `gguf-py/gguf/tensor_mapping.py`:

- **60+ new tensor mappings** — GPT-2 `c_attn`/`c_proj`, Bloom `dense_h_to_4h`, Falcon `query_key_value`/`dense`, Phi `fc1`/`fc2`, Mamba `mixer.*`, StarCoder2, Baichuan, InternLM2
- **MoE expert tensors** — dynamic routing for `block_sparse_moe.experts.N.*` (Mixtral), `mlp.experts.N.*` (DeepSeek, Qwen2-MoE), and `mlp.shared_experts.*` (DeepSeek shared experts)
- **RoPE scaling metadata** — `original_max_position_embeddings`, `low_freq_factor`, `high_freq_factor` (LLaMA-3.1 style)
- **20+ new architecture names** — Jamba, Mamba, OLMo, Orion, Arctic, ChatGLM, Solar, Granite, etc.

---

### Improved: `HfConfigParser.java`
- Added `slidingWindow`, `attentionBias`, `partialRotaryFactor`, `headDim` fields to `ModelConfig`
- Multiple key aliases for `rms_norm_eps` (`layer_norm_epsilon`, `layer_norm_eps`) and `rope_theta` (`rotary_emb_base`)
- Better tokenizer parsing: handles BOS/EOS tokens that are strings *or* `{content: …}` objects, detects LLaMA-3 `<|begin_of_text|>`/`<|eot_id|>` patterns, fixes null gaps in vocab arrays

---

### Improved: `HfToGgufConverter.java`
- **Fixed two-pass offset bug** — type decisions and offset calculations now happen in a single consistent pass; the descriptor registered in step 5 always matches the data written in step 6
- **Shard index JSON** — falls back to scanning only when the index file is absent
- **Tied-embedding skip** — skips `lm_head.weight` when `tie_word_embeddings = true`
- Cached shard header reads to avoid re-opening the same file for every tensor lookup

---

### Improved: `GgufConverterMain.java`
- Added `safetensors` subcommand (routes to `SafetensorToGgufConverter`) with `--progress` flag that renders a live `\r` progress bar
- `convert` retained for backwards-compat