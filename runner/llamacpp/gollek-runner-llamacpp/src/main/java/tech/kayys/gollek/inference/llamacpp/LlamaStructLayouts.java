package tech.kayys.gollek.inference.llamacpp;

import java.lang.foreign.*;

/**
 * Shared FFM struct layout definitions for llama.cpp C structs.
 *
 * <p>All layouts are package-private constants used by the binding components.
 * They mirror the exact memory layout of the corresponding C structs so that
 * {@link java.lang.foreign.MemorySegment} field access is correct on all platforms.
 */
final class LlamaStructLayouts {

    private LlamaStructLayouts() {}

    /** {@code llama_model_params} — controls model loading (GPU layers, mmap, etc.). */
    static final MemoryLayout MODEL_PARAMS = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("devices"),
            ValueLayout.ADDRESS.withName("tensor_buft_overrides"),
            ValueLayout.JAVA_INT.withName("n_gpu_layers"),
            ValueLayout.JAVA_INT.withName("split_mode"),
            ValueLayout.JAVA_INT.withName("main_gpu"),
            MemoryLayout.paddingLayout(4),
            ValueLayout.ADDRESS.withName("tensor_split"),
            ValueLayout.ADDRESS.withName("progress_callback"),
            ValueLayout.ADDRESS.withName("progress_callback_user_data"),
            ValueLayout.ADDRESS.withName("kv_overrides"),
            ValueLayout.JAVA_BYTE.withName("vocab_only"),
            ValueLayout.JAVA_BYTE.withName("use_mmap"),
            ValueLayout.JAVA_BYTE.withName("use_direct_io"),
            ValueLayout.JAVA_BYTE.withName("use_mlock"),
            ValueLayout.JAVA_BYTE.withName("check_tensors"),
            ValueLayout.JAVA_BYTE.withName("use_extra_bufts"),
            ValueLayout.JAVA_BYTE.withName("no_host"),
            ValueLayout.JAVA_BYTE.withName("no_alloc")).withName("llama_model_params");

    /** {@code llama_context_params} — controls inference context (threads, RoPE, KV cache, etc.). */
    static final StructLayout CONTEXT_PARAMS = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_ctx"),
            ValueLayout.JAVA_INT.withName("n_batch"),
            ValueLayout.JAVA_INT.withName("n_ubatch"),
            ValueLayout.JAVA_INT.withName("n_seq_max"),
            ValueLayout.JAVA_INT.withName("n_threads"),
            ValueLayout.JAVA_INT.withName("n_threads_batch"),
            ValueLayout.JAVA_INT.withName("rope_scaling_type"),
            ValueLayout.JAVA_INT.withName("pooling_type"),
            ValueLayout.JAVA_INT.withName("attention_type"),
            ValueLayout.JAVA_INT.withName("flash_attn_type"),
            ValueLayout.JAVA_FLOAT.withName("rope_freq_base"),
            ValueLayout.JAVA_FLOAT.withName("rope_freq_scale"),
            ValueLayout.JAVA_FLOAT.withName("yarn_ext_factor"),
            ValueLayout.JAVA_FLOAT.withName("yarn_attn_factor"),
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_fast"),
            ValueLayout.JAVA_FLOAT.withName("yarn_beta_slow"),
            ValueLayout.JAVA_INT.withName("yarn_orig_ctx"),
            ValueLayout.JAVA_FLOAT.withName("defrag_thold"),
            ValueLayout.ADDRESS.withName("cb_eval"),
            ValueLayout.ADDRESS.withName("cb_eval_user_data"),
            ValueLayout.JAVA_INT.withName("type_k"),
            ValueLayout.JAVA_INT.withName("type_v"),
            ValueLayout.ADDRESS.withName("abort_callback"),
            ValueLayout.ADDRESS.withName("abort_callback_data"),
            ValueLayout.JAVA_BYTE.withName("embeddings"),
            ValueLayout.JAVA_BYTE.withName("offload_kqv"),
            ValueLayout.JAVA_BYTE.withName("no_perf"),
            ValueLayout.JAVA_BYTE.withName("op_offload"),
            ValueLayout.JAVA_BYTE.withName("swa_full"),
            ValueLayout.JAVA_BYTE.withName("kv_unified"),
            MemoryLayout.paddingLayout(2),
            ValueLayout.ADDRESS.withName("samplers"),
            ValueLayout.JAVA_LONG.withName("n_samplers")).withName("llama_context_params");

    /** {@code llama_batch} — token batch submitted to {@code llama_decode}. */
    static final MemoryLayout BATCH = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_tokens"),
            MemoryLayout.paddingLayout(4),
            ValueLayout.ADDRESS.withName("token"),
            ValueLayout.ADDRESS.withName("embd"),
            ValueLayout.ADDRESS.withName("pos"),
            ValueLayout.ADDRESS.withName("n_seq_id"),
            ValueLayout.ADDRESS.withName("seq_id"),
            ValueLayout.ADDRESS.withName("logits")).withName("llama_batch");

    /** {@code llama_sampler_chain_params} — single-byte struct controlling perf tracking. */
    static final MemoryLayout SAMPLER_CHAIN_PARAMS = MemoryLayout.structLayout(
            ValueLayout.JAVA_BYTE.withName("no_perf"));
}
