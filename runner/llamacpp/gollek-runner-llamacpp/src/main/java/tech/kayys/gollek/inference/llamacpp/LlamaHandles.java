package tech.kayys.gollek.inference.llamacpp;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Holds all {@link MethodHandle} downcalls to the llama.cpp native library and
 * provides the {@link #link} / {@link #linkOptional} helpers used during construction.
 *
 * <p>This class is package-private. Consumers use {@link LlamaCppBinding} as the
 * public facade.
 *
 * <p>Required handles are linked eagerly; optional handles (LoRA, Mirostat, grammar,
 * etc.) are left {@code null} when the symbol is absent from the loaded library.
 */
final class LlamaHandles {

    private static final Logger log = Logger.getLogger(LlamaHandles.class);

    // ── Backend ───────────────────────────────────────────────────────────────
    final MethodHandle backendInit;
    final MethodHandle backendFree;

    // ── Model / context defaults (shim-first, then standard) ─────────────────
    final MethodHandle modelDefaultParams;       // optional — struct-return variant
    final MethodHandle contextDefaultParams;     // optional — struct-return variant
    final MethodHandle gollekModelDefaultParamsInto;  // shim: fills existing segment
    final MethodHandle gollekContextDefaultParamsInto;
    final MethodHandle gollekLogDisable;

    // ── Model lifecycle ───────────────────────────────────────────────────────
    final MethodHandle loadModelFromFile;
    final MethodHandle initFromModel;
    final MethodHandle freeModel;
    final MethodHandle freeContext;

    // ── KV cache ─────────────────────────────────────────────────────────────
    final MethodHandle getMemory;
    final MethodHandle memoryClear;

    // ── Vocab / metadata ─────────────────────────────────────────────────────
    final MethodHandle modelGetVocab;
    final MethodHandle modelMetaValStr;
    final MethodHandle vocabEos;
    final MethodHandle vocabBos;
    final MethodHandle vocabNTokens;
    final MethodHandle vocabIsEog;

    // ── Tokenization ─────────────────────────────────────────────────────────
    final MethodHandle tokenize;
    final MethodHandle tokenToPiece;
    final MethodHandle detokenize;

    // ── Inference ────────────────────────────────────────────────────────────
    final MethodHandle decode;
    final MethodHandle getLogits;
    final MethodHandle getLogitsIth;
    final MethodHandle nCtx;

    // ── Batch ────────────────────────────────────────────────────────────────
    final MethodHandle batchInit;   // optional
    final MethodHandle batchFree;

    // ── Sampler chain ─────────────────────────────────────────────────────────
    final MethodHandle samplerChainDefaultParams; // optional
    final MethodHandle samplerChainInit;
    final MethodHandle samplerChainAdd;
    final MethodHandle samplerInitGreedy;
    final MethodHandle samplerInitTopK;
    final MethodHandle samplerInitTopP;           // optional
    final MethodHandle samplerInitMinP;           // optional
    final MethodHandle samplerInitTemp;
    final MethodHandle samplerInitDist;
    final MethodHandle samplerSample;             // optional
    final MethodHandle samplerFree;
    final MethodHandle samplerInitPenalties;
    final MethodHandle samplerInitMirostat;
    final MethodHandle samplerInitMirostatV2;
    final MethodHandle samplerInitGrammar;
    final MethodHandle samplerInitTypical;        // optional

    // ── Embeddings ───────────────────────────────────────────────────────────
    final MethodHandle nEmbd;
    final MethodHandle getEmbeddings;
    final MethodHandle getEmbeddingsIth;

    // ── LoRA adapters (all optional) ─────────────────────────────────────────
    final MethodHandle adapterLoraInit;
    final MethodHandle setAdapterLora;
    final MethodHandle rmAdapterLora;
    final MethodHandle clearAdapterLora;
    final MethodHandle adapterLoraFree;

    // ── Verbosity flag (set after construction) ───────────────────────────────
    boolean verbose = false;

    LlamaHandles(SymbolLookup lookup) {
        Linker linker = Linker.nativeLinker();

        backendInit  = link(linker, lookup, "llama_backend_init",  FunctionDescriptor.ofVoid());
        backendFree  = link(linker, lookup, "llama_backend_free",  FunctionDescriptor.ofVoid());

        modelDefaultParams   = linkOpt(linker, lookup, "llama_model_default_params",
                FunctionDescriptor.of(LlamaStructLayouts.MODEL_PARAMS));
        contextDefaultParams = linkOpt(linker, lookup, "llama_context_default_params",
                FunctionDescriptor.of(LlamaStructLayouts.CONTEXT_PARAMS));
        gollekModelDefaultParamsInto   = linkOpt(linker, lookup, "gollek_llama_model_default_params_into",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        gollekContextDefaultParamsInto = linkOpt(linker, lookup, "gollek_llama_context_default_params_into",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        gollekLogDisable = linkOpt(linker, lookup, "gollek_llama_log_disable", FunctionDescriptor.ofVoid());

        loadModelFromFile = link(linker, lookup, "llama_model_load_from_file",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LlamaStructLayouts.MODEL_PARAMS));
        initFromModel = link(linker, lookup, "llama_init_from_model",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LlamaStructLayouts.CONTEXT_PARAMS));
        freeModel   = link(linker, lookup, "llama_free_model",   FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        freeContext = link(linker, lookup, "llama_free",         FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        getMemory    = link(linker, lookup, "llama_get_memory",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        memoryClear  = link(linker, lookup, "llama_memory_clear",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

        modelGetVocab    = link(linker, lookup, "llama_model_get_vocab",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        modelMetaValStr  = link(linker, lookup, "llama_model_meta_val_str",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        vocabEos         = link(linker, lookup, "llama_vocab_eos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        vocabBos         = link(linker, lookup, "llama_vocab_bos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        vocabNTokens     = link(linker, lookup, "llama_vocab_n_tokens",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        vocabIsEog       = link(linker, lookup, "llama_vocab_is_eog",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        tokenize     = link(linker, lookup, "llama_tokenize",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));
        tokenToPiece = link(linker, lookup, "llama_token_to_piece",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
        detokenize   = link(linker, lookup, "llama_detokenize",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));

        decode       = link(linker, lookup, "llama_decode",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, LlamaStructLayouts.BATCH));
        getLogits    = link(linker, lookup, "llama_get_logits",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        getLogitsIth = link(linker, lookup, "llama_get_logits_ith",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        nCtx         = link(linker, lookup, "llama_n_ctx",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        batchInit = linkOpt(linker, lookup, "llama_batch_init",
                FunctionDescriptor.of(LlamaStructLayouts.BATCH, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        batchFree = link(linker, lookup, "llama_batch_free",
                FunctionDescriptor.ofVoid(LlamaStructLayouts.BATCH));

        samplerChainDefaultParams = linkOpt(linker, lookup, "llama_sampler_chain_default_params",
                FunctionDescriptor.of(LlamaStructLayouts.SAMPLER_CHAIN_PARAMS));
        samplerChainInit  = link(linker, lookup, "llama_sampler_chain_init",
                FunctionDescriptor.of(ValueLayout.ADDRESS, LlamaStructLayouts.SAMPLER_CHAIN_PARAMS));
        samplerChainAdd   = link(linker, lookup, "llama_sampler_chain_add",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        samplerInitGreedy = link(linker, lookup, "llama_sampler_init_greedy",
                FunctionDescriptor.of(ValueLayout.ADDRESS));
        samplerInitTopK   = link(linker, lookup, "llama_sampler_init_top_k",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        samplerInitTopP   = linkOpt(linker, lookup, "llama_sampler_init_top_p",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));
        samplerInitMinP   = linkOpt(linker, lookup, "llama_sampler_init_min_p",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));
        samplerInitTemp   = link(linker, lookup, "llama_sampler_init_temp",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT));
        samplerInitDist   = link(linker, lookup, "llama_sampler_init_dist",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        samplerSample     = linkOpt(linker, lookup, "llama_sampler_sample",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT));
        samplerFree       = link(linker, lookup, "llama_sampler_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        samplerInitPenalties = link(linker, lookup, "llama_sampler_init_penalties",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT,
                        ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));
        samplerInitMirostat = link(linker, lookup, "llama_sampler_init_mirostat",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));
        samplerInitMirostatV2 = link(linker, lookup, "llama_sampler_init_mirostat_v2",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));
        samplerInitGrammar = link(linker, lookup, "llama_sampler_init_grammar",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        samplerInitTypical = linkOpt(linker, lookup, "llama_sampler_init_typical",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));

        nEmbd           = link(linker, lookup, "llama_n_embd",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        getEmbeddings   = link(linker, lookup, "llama_get_embeddings",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        getEmbeddingsIth = link(linker, lookup, "llama_get_embeddings_ith",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        adapterLoraInit  = linkOpt(linker, lookup, "llama_adapter_lora_init",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        setAdapterLora   = linkOpt(linker, lookup, "llama_set_adapter_lora",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_FLOAT));
        rmAdapterLora    = linkOpt(linker, lookup, "llama_rm_adapter_lora",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        clearAdapterLora = linkOpt(linker, lookup, "llama_clear_adapter_lora",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        adapterLoraFree  = linkOpt(linker, lookup, "llama_adapter_lora_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    }

    // ── Linking helpers ───────────────────────────────────────────────────────

    /** Links a required symbol; logs a warning if absent. */
    MethodHandle link(Linker linker, SymbolLookup lookup, String name, FunctionDescriptor desc) {
        return doLink(linker, lookup, name, desc, true);
    }

    /** Links an optional symbol; logs at debug level if absent. */
    MethodHandle linkOpt(Linker linker, SymbolLookup lookup, String name, FunctionDescriptor desc) {
        return doLink(linker, lookup, name, desc, false);
    }

    private MethodHandle doLink(Linker linker, SymbolLookup lookup,
            String name, FunctionDescriptor desc, boolean required) {
        try {
            return lookup.find(name)
                    .map(addr -> {
                        log.debugf("Linking native function: %s", name);
                        return linker.downcallHandle(addr, desc);
                    })
                    .orElseGet(() -> {
                        String msg = "Native function '" + name + "' not found.";
                        if (required) log.warn(msg); else log.debug(msg);
                        return null;
                    });
        } catch (Throwable t) {
            String msg = "Failed to link" + (required ? "" : " optional") + " native function '" + name + "': " + t.getMessage();
            if (required && verbose) log.warn(msg); else log.debug(msg);
            return null;
        }
    }

    /** Throws {@link IllegalStateException} if the handle is {@code null}. */
    void require(MethodHandle handle, String name) {
        if (handle == null) {
            throw new IllegalStateException(
                    "Native function '" + name + "' not available in the loaded library.");
        }
    }
}
