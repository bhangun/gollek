
package tech.kayys.gollek.inference.gguf;

import java.io.InputStream;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.jboss.logging.Logger;

/**
 * JDK 21+ Foreign Function & Memory API binding for llama.cpp
 * This provides direct access to native llama.cpp functions
 */
public class LlamaCppBinding {

    private static final Logger log = Logger.getLogger(LlamaCppBinding.class);
    private static final String LIB_BASE_NAME = "llama";

    private final SymbolLookup symbolLookup;
    private final Arena arena;
    private final java.util.concurrent.ConcurrentHashMap<Long, Arena> manualBatchArenas = new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.concurrent.ConcurrentHashMap<Long, ManualBatchBuffers> manualBatchBuffers = new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.concurrent.atomic.AtomicBoolean backendInitialized = new java.util.concurrent.atomic.AtomicBoolean(
            false);

    // Function handles
    private final MethodHandle llama_backend_init;
    private final MethodHandle llama_backend_free;
    private final MethodHandle llama_model_default_params;
    private final MethodHandle llama_context_default_params;
    private final MethodHandle gollek_llama_model_default_params_into;
    private final MethodHandle gollek_llama_context_default_params_into;
    private final MethodHandle gollek_llama_log_disable;
    private final MethodHandle llama_load_model_from_file;
    private final MethodHandle llama_new_context_with_model;
    private final MethodHandle llama_free_model;
    private final MethodHandle llama_get_memory;
    private final MethodHandle llama_memory_clear;
    private final MethodHandle llama_free;
    private final MethodHandle llama_model_get_vocab;
    private final MethodHandle llama_model_meta_val_str;
    private final MethodHandle llama_tokenize;
    private final MethodHandle llama_token_to_piece;
    private final MethodHandle llama_detokenize;
    private final MethodHandle llama_decode;
    private final MethodHandle llama_get_logits;
    private final MethodHandle llama_get_logits_ith;
    private final MethodHandle llama_vocab_eos;
    private final MethodHandle llama_vocab_bos;
    private final MethodHandle llama_batch_init;
    private final MethodHandle llama_batch_free;
    private final MethodHandle llama_n_ctx;
    private final MethodHandle llama_vocab_n_tokens;
    private final MethodHandle llama_sampler_chain_default_params;
    private final MethodHandle llama_sampler_chain_init;
    private final MethodHandle llama_sampler_chain_add;
    private final MethodHandle llama_sampler_init_greedy;
    private final MethodHandle llama_sampler_init_top_k;
    private final MethodHandle llama_sampler_init_top_p;
    private final MethodHandle llama_sampler_init_min_p;
    private final MethodHandle llama_sampler_init_temp;
    private final MethodHandle llama_sampler_init_dist;
    private final MethodHandle llama_sampler_sample;
    private final MethodHandle llama_sampler_free;

    // Extended sampler handles
    private final MethodHandle llama_sampler_init_penalties;
    private final MethodHandle llama_sampler_init_mirostat;
    private final MethodHandle llama_sampler_init_mirostat_v2;
    private final MethodHandle llama_sampler_init_grammar;
    private final MethodHandle llama_sampler_init_typical;
    private final MethodHandle llama_vocab_is_eog;

    // Struct layouts
    private static final MemoryLayout LLAMA_MODEL_PARAMS_LAYOUT = MemoryLayout.structLayout(
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

    private static final StructLayout LLAMA_CONTEXT_PARAMS_LAYOUT = MemoryLayout.structLayout(
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

    private static final MemoryLayout LLAMA_BATCH_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("n_tokens"),
            MemoryLayout.paddingLayout(4), // Aligns next address pointer to 8 bytes
            ValueLayout.ADDRESS.withName("token"),
            ValueLayout.ADDRESS.withName("embd"),
            ValueLayout.ADDRESS.withName("pos"),
            ValueLayout.ADDRESS.withName("n_seq_id"),
            ValueLayout.ADDRESS.withName("seq_id"),
            ValueLayout.ADDRESS.withName("logits")).withName("llama_batch");

    private static final MemoryLayout LLAMA_SAMPLER_CHAIN_PARAMS_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_BYTE.withName("no_perf"));

    private static final class ManualBatchBuffers {
        private final MemorySegment token;
        private final MemorySegment pos;
        private final MemorySegment nSeqId;
        private final MemorySegment seqIdPtr;
        private final MemorySegment logits;

        private ManualBatchBuffers(
                MemorySegment token,
                MemorySegment pos,
                MemorySegment nSeqId,
                MemorySegment seqIdPtr,
                MemorySegment logits) {
            this.token = token;
            this.pos = pos;
            this.nSeqId = nSeqId;
            this.seqIdPtr = seqIdPtr;
            this.logits = logits;
        }
    }

    private LlamaCppBinding(SymbolLookup symbolLookup) {

        this.symbolLookup = symbolLookup;
        this.arena = Arena.ofShared();

        // Link function handles
        Linker linker = Linker.nativeLinker();

        this.llama_backend_init = linkFunction(linker, "llama_backend_init",
                FunctionDescriptor.ofVoid());

        this.llama_backend_free = linkFunction(linker, "llama_backend_free",
                FunctionDescriptor.ofVoid());

        this.llama_model_default_params = linkOptionalFunction(linker, "llama_model_default_params",
                FunctionDescriptor.of(LLAMA_MODEL_PARAMS_LAYOUT));

        this.llama_context_default_params = linkOptionalFunction(linker, "llama_context_default_params",
                FunctionDescriptor.of(LLAMA_CONTEXT_PARAMS_LAYOUT));

        this.gollek_llama_model_default_params_into = linkFunction(
                linker,
                "gollek_llama_model_default_params_into",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        this.gollek_llama_context_default_params_into = linkFunction(
                linker,
                "gollek_llama_context_default_params_into",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        this.gollek_llama_log_disable = linkFunction(
                linker,
                "gollek_llama_log_disable",
                FunctionDescriptor.ofVoid());

        this.llama_load_model_from_file = linkFunction(linker, "llama_model_load_from_file",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LLAMA_MODEL_PARAMS_LAYOUT));
        this.llama_new_context_with_model = linkFunction(linker, "llama_init_from_model",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LLAMA_CONTEXT_PARAMS_LAYOUT));
        this.llama_free_model = linkFunction(linker, "llama_free_model",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // llama_kv_cache_clear is deprecated/removed in favor of llama_memory_clear
        this.llama_get_memory = linkFunction(linker, "llama_get_memory",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        this.llama_memory_clear = linkFunction(linker, "llama_memory_clear",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

        this.llama_free = linkFunction(linker, "llama_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // Get vocab pointer from model (new API)
        this.llama_model_get_vocab = linkFunction(linker, "llama_model_get_vocab",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        // Metadata API
        this.llama_model_meta_val_str = linkFunction(linker, "llama_model_meta_val_str",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // Updated tokenize: now takes vocab pointer, not model
        // llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special,
        // parse_special)
        this.llama_tokenize = linkFunction(linker, "llama_tokenize",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN,
                        ValueLayout.JAVA_BOOLEAN));

        // Updated token_to_piece: now takes vocab pointer and has lstrip and special
        // params
        // llama_token_to_piece(vocab, token, buf, length, lstrip, special)
        this.llama_token_to_piece = linkFunction(linker, "llama_token_to_piece",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN));
        this.llama_detokenize = linkFunction(linker, "llama_detokenize",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN,
                        ValueLayout.JAVA_BOOLEAN));

        this.llama_decode = linkFunction(linker, "llama_decode",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, LLAMA_BATCH_LAYOUT));

        this.llama_get_logits = linkFunction(linker, "llama_get_logits",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        this.llama_get_logits_ith = linkFunction(linker, "llama_get_logits_ith",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // Updated token functions: now take vocab pointer
        this.llama_vocab_eos = linkFunction(linker, "llama_vocab_eos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        this.llama_vocab_bos = linkFunction(linker, "llama_vocab_bos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        this.llama_batch_init = linkOptionalFunction(linker, "llama_batch_init",
                FunctionDescriptor.of(LLAMA_BATCH_LAYOUT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT));

        this.llama_batch_free = linkFunction(linker, "llama_batch_free",
                FunctionDescriptor.ofVoid(LLAMA_BATCH_LAYOUT));

        this.llama_n_ctx = linkFunction(linker, "llama_n_ctx",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // Replaced llama_n_vocab with llama_vocab_n_tokens
        this.llama_vocab_n_tokens = linkFunction(linker, "llama_vocab_n_tokens",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        // Sampler API
        this.llama_sampler_chain_default_params = linkOptionalFunction(linker, "llama_sampler_chain_default_params",
                FunctionDescriptor.of(LLAMA_SAMPLER_CHAIN_PARAMS_LAYOUT));

        this.llama_sampler_chain_init = linkFunction(linker, "llama_sampler_chain_init",
                FunctionDescriptor.of(ValueLayout.ADDRESS, LLAMA_SAMPLER_CHAIN_PARAMS_LAYOUT));

        this.llama_sampler_chain_add = linkFunction(linker, "llama_sampler_chain_add",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        this.llama_sampler_init_greedy = linkFunction(linker, "llama_sampler_init_greedy",
                FunctionDescriptor.of(ValueLayout.ADDRESS));

        this.llama_sampler_init_top_k = linkFunction(linker, "llama_sampler_init_top_k",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        this.llama_sampler_init_top_p = linkOptionalFunction(linker, "llama_sampler_init_top_p",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));

        this.llama_sampler_init_min_p = linkOptionalFunction(linker, "llama_sampler_init_min_p",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));

        this.llama_sampler_init_temp = linkFunction(linker, "llama_sampler_init_temp",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT));

        this.llama_sampler_init_dist = linkFunction(linker, "llama_sampler_init_dist",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        this.llama_sampler_sample = linkOptionalFunction(linker, "llama_sampler_sample",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT));

        this.llama_sampler_free = linkFunction(linker, "llama_sampler_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // Extended samplers
        // llama_sampler_init_penalties(int32_t penalty_last_n, float penalty_repeat,
        // float penalty_freq, float penalty_present)
        this.llama_sampler_init_penalties = linkFunction(linker, "llama_sampler_init_penalties",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT,
                        ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // llama_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float
        // eta, int32_t m)
        this.llama_sampler_init_mirostat = linkFunction(linker, "llama_sampler_init_mirostat",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // llama_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta)
        this.llama_sampler_init_mirostat_v2 = linkFunction(linker, "llama_sampler_init_mirostat_v2",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT,
                        ValueLayout.JAVA_FLOAT));

        // llama_sampler_init_grammar(const llama_vocab* vocab, const char* grammar_str,
        // const char* grammar_root)
        this.llama_sampler_init_grammar = linkFunction(linker, "llama_sampler_init_grammar",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS));

        // llama_sampler_init_typical(float p, size_t min_keep)
        this.llama_sampler_init_typical = linkOptionalFunction(linker, "llama_sampler_init_typical",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_LONG));

        // llama_vocab_is_eog(const llama_vocab* vocab, llama_token token) -> bool
        this.llama_vocab_is_eog = linkFunction(linker, "llama_vocab_is_eog",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    }

    private MethodHandle linkFunction(Linker linker, String name, FunctionDescriptor descriptor) {
        return linkFunction(linker, name, descriptor, true);
    }

    private MethodHandle linkOptionalFunction(Linker linker, String name, FunctionDescriptor descriptor) {
        return linkFunction(linker, name, descriptor, false);
    }

    private MethodHandle linkFunction(Linker linker, String name, FunctionDescriptor descriptor, boolean required) {
        try {
            return symbolLookup.find(name)
                    .map(addr -> {
                        log.debugf("Linking native function: %s with descriptor: %s", name, descriptor);
                        return linker.downcallHandle(addr, descriptor);
                    })
                    .orElseGet(() -> {
                        String msg = String.format("Native function '%s' not found. Some features may be disabled.",
                                name);
                        if (required) {
                            log.warn(msg);
                        } else {
                            log.debug(msg);
                        }
                        return null;
                    });
        } catch (Throwable t) {
            if (required) {
                log.warnf("Failed to link native function '%s' (%s). Function will be treated as unavailable.",
                        name, t.getMessage());
            } else {
                log.debugf("Failed to link optional native function '%s' (%s).", name, t.getMessage());
            }
            return null;
        }
    }

    /**
     * Load the native library and create binding instance with default (quiet)
     * logging.
     */
    public static LlamaCppBinding load() {
        return load(false);
    }

    public static LlamaCppBinding load(GGUFProviderConfig config) {
        boolean verbose = config != null && config.verboseLogging();
        Optional<String> explicitLibPath = config == null ? Optional.empty() : config.nativeLibraryPath();
        Optional<String> explicitLibDir = config == null ? Optional.empty() : config.nativeLibraryDir();
        return load(verbose, explicitLibPath, explicitLibDir);
    }

    /**
     * Load the native library and create binding instance.
     * 
     * @param verbose if true, show all llama.cpp debug output; if false, suppress
     *                verbose logs
     */
    public static LlamaCppBinding load(boolean verbose) {
        return load(verbose, Optional.empty(), Optional.empty());
    }

    private static LlamaCppBinding load(boolean verbose, Optional<String> explicitLibPath,
            Optional<String> explicitLibDir) {
        try {
            Path loadedFrom = loadNativeLibrary(explicitLibPath, explicitLibDir, verbose);

            // Create symbol lookup
            SymbolLookup symbolLookup = SymbolLookup.loaderLookup();

            // Suppress verbose native logs BEFORE any llama.cpp calls
            if (!verbose) {
                suppressNativeLogsStatic(symbolLookup);
            }

            LlamaCppBinding binding = new LlamaCppBinding(symbolLookup);

            // Initialize backend
            binding.backendInit();

            if (!verbose) {
                log.info("Loaded llama.cpp native library (quiet mode)");
            } else {
                log.infof("Loaded llama.cpp native library: %s", loadedFrom);
            }
            return binding;

        } catch (Throwable e) {
            e.printStackTrace();
            System.err.println("FATAL: Failed to load llama.cpp library: " + e.getMessage());
            throw new RuntimeException("Failed to load llama.cpp library", e);
        }
    }

    private static Path loadNativeLibrary(Optional<String> explicitLibPath, Optional<String> explicitLibDir,
            boolean verbose) throws Exception {
        String mappedLibName = System.mapLibraryName(LIB_BASE_NAME);
        List<String> attempts = new ArrayList<>();
        Optional<String> configuredLibPath = explicitLibPath.isPresent() && !explicitLibPath.get().isBlank()
                ? explicitLibPath
                : optionalEnv("GOLEK_LLAMA_LIB_PATH");
        Optional<String> configuredLibDir = explicitLibDir.isPresent() && !explicitLibDir.get().isBlank()
                ? explicitLibDir
                : optionalEnv("GOLEK_LLAMA_LIB_DIR");

        // 1) Explicit absolute library file path.
        if (configuredLibPath.isPresent() && !configuredLibPath.get().isBlank()) {
            Path candidate = Path.of(configuredLibPath.get()).toAbsolutePath();
            attempts.add("explicit library path: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(candidate.getParent(), candidate.getFileName().toString(), verbose);
                return candidate;
            }
        }

        // 2) Explicit directory containing libllama + libggml* files.
        if (configuredLibDir.isPresent() && !configuredLibDir.get().isBlank()) {
            Path dir = Path.of(configuredLibDir.get()).toAbsolutePath();
            Path candidate = dir.resolve(mappedLibName);
            attempts.add("explicit library dir: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(dir, mappedLibName, verbose);
                return candidate;
            }
        }

        // 3) Standard linker path (e.g. java.library.path / DYLD_LIBRARY_PATH).
        try {
            System.loadLibrary(LIB_BASE_NAME);
            attempts.add("System.loadLibrary(" + LIB_BASE_NAME + ")");
            return Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        } catch (Throwable t) {
            attempts.add("System.loadLibrary(" + LIB_BASE_NAME + ") failed: " + t.getMessage());
        }

        // 4) Common runtime filesystem locations around executable/current dir.
        for (Path dir : candidateRuntimeDirs()) {
            Path candidate = dir.resolve(mappedLibName);
            attempts.add("runtime dir: " + candidate);
            if (Files.exists(candidate)) {
                loadWithDependencies(dir, mappedLibName, verbose);
                return candidate;
            }
        }

        // 5) Extract all relevant libs from resources and load from temp.
        Path extractedMain = extractAndLoadFromResources(verbose);
        if (extractedMain != null) {
            attempts.add("resource extraction: " + extractedMain);
            return extractedMain;
        }

        throw new RuntimeException(
                "Unable to locate/load llama.cpp native library. Attempts: " + String.join(" | ", attempts));
    }

    /**
     * Static method to suppress native logs before binding instance is created.
     */
    private static void suppressNativeLogsStatic(SymbolLookup symbolLookup) {
        try {
            var shimLogDisableAddr = symbolLookup.find("gollek_llama_log_disable");
            if (shimLogDisableAddr.isPresent()) {
                Linker nativeLinker = Linker.nativeLinker();
                MethodHandle shimDisable = nativeLinker.downcallHandle(
                        shimLogDisableAddr.get(),
                        FunctionDescriptor.ofVoid());
                shimDisable.invoke();
                return;
            }

            var logSetAddr = symbolLookup.find("llama_log_set");
            if (logSetAddr.isPresent()) {
                Linker nativeLinker = Linker.nativeLinker();

                // Create a no-op callback using upcall stub
                // Signature: void callback(int level, const char* text, void* user_data)
                MethodHandle noOpHandler = MethodHandles.lookup().findStatic(
                        LlamaCppBinding.class,
                        "noOpLogCallback",
                        MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class));

                FunctionDescriptor callbackDesc = FunctionDescriptor.ofVoid(
                        ValueLayout.JAVA_INT, // enum ggml_log_level (int)
                        ValueLayout.ADDRESS, // const char* text
                        ValueLayout.ADDRESS // void* user_data
                );

                // Create upcall stub in global arena (lives as long as JVM)
                MemorySegment noOpCallback = nativeLinker.upcallStub(
                        noOpHandler,
                        callbackDesc,
                        Arena.global());

                MethodHandle llama_log_set = nativeLinker.downcallHandle(
                        logSetAddr.get(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

                // Set our no-op callback
                llama_log_set.invoke(noOpCallback, MemorySegment.NULL);
            }
        } catch (Throwable e) {
            // Silently ignore - verbose logging is just a convenience
        }
    }

    /**
     * No-op log callback that discards all log messages.
     */
    @SuppressWarnings("unused")
    private static void noOpLogCallback(int level, MemorySegment text, MemorySegment userData) {
        // Intentionally empty - discards all log output
    }

    /**
     * Suppress verbose llama.cpp debug output.
     * Sets a no-op log callback to silence Metal/CUDA pipeline compilation
     * messages.
     */
    public void suppressNativeLogs() {
        try {
            if (gollek_llama_log_disable != null) {
                gollek_llama_log_disable.invoke();
                log.debug("Suppressed verbose llama.cpp native logging via shim");
                return;
            }
        } catch (Throwable t) {
            log.debugf("Shim-based native log suppression failed: %s", t.getMessage());
        }
        suppressNativeLogsStatic(symbolLookup);
        log.debug("Suppressed verbose llama.cpp native logging");
    }

    private static Path extractAndLoadFromResources(boolean verbose) throws Exception {
        String mainLib = System.mapLibraryName(LIB_BASE_NAME);
        Path tempDir = Files.createTempDirectory("llama-cpp");
        List<Path> extracted = new ArrayList<>();

        for (String fileName : nativeDependencyFileNames()) {
            Path extractedPath = extractLibraryFromResources(fileName, tempDir);
            if (extractedPath != null) {
                extracted.add(extractedPath);
            }
        }

        if (extracted.isEmpty()) {
            return null;
        }

        if (verbose) {
            log.infof("Extracted native libraries to %s", tempDir);
        }
        loadWithDependencies(tempDir, mainLib, verbose);
        return tempDir.resolve(mainLib);
    }

    private static Path extractLibraryFromResources(String fileName, Path targetDir) throws Exception {
        for (String prefix : nativeResourcePrefixes()) {
            String resourcePath = prefix + fileName;
            try (InputStream is = getResourceAsStream(resourcePath)) {
                if (is != null) {
                    Path target = targetDir.resolve(fileName);
                    Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
                    if (!isWindows()) {
                        target.toFile().setExecutable(true);
                    }
                    return target;
                }
            }
        }
        return null;
    }

    private static List<String> nativeResourcePrefixes() {
        String cudaPath = System.getenv("CUDA_PATH");
        boolean hasCuda = cudaPath != null && !cudaPath.isBlank();
        List<String> prefixes = new ArrayList<>();
        if (hasCuda) {
            prefixes.add("/gollek-gguf/native-libs/cuda/");
        } else {
            prefixes.add("/gollek-gguf/native-libs/cpu/");
        }
        prefixes.add("/gollek-gguf/native-libs/");
        if (hasCuda) {
            prefixes.add("/native-libs/cuda/");
        } else {
            prefixes.add("/native-libs/cpu/");
        }
        prefixes.add("/native-libs/");
        return prefixes;
    }

    private static void loadWithDependencies(Path libraryDir, String mainLibName, boolean verbose) {
        List<String> errors = new ArrayList<>();
        String shimName = shimLibraryFileName();
        for (String dep : dependencyLoadOrder(libraryDir)) {
            if (dep.equals(mainLibName) || dep.equals("lib" + LIB_BASE_NAME + nativeLibExt())
                    || dep.startsWith("lib" + LIB_BASE_NAME + ".")) {
                continue;
            }
            if (dep.equals(shimName)) {
                continue;
            }
            Path depPath = libraryDir.resolve(dep);
            if (!Files.exists(depPath)) {
                continue;
            }
            try {
                System.load(depPath.toAbsolutePath().toString());
                if (verbose) {
                    log.infof("Loaded native dependency: %s", depPath);
                }
            } catch (UnsatisfiedLinkError e) {
                errors.add(dep + ": " + e.getMessage());
            }
        }

        Path mainPath = libraryDir.resolve(mainLibName);
        System.load(mainPath.toAbsolutePath().toString());

        Path shimPath = libraryDir.resolve(shimName);
        if (Files.exists(shimPath)) {
            try {
                System.load(shimPath.toAbsolutePath().toString());
                if (verbose) {
                    log.infof("Loaded native shim: %s", shimPath);
                }
            } catch (UnsatisfiedLinkError e) {
                errors.add(shimName + ": " + e.getMessage());
            }
        }

        if (!errors.isEmpty()) {
            log.debugf("Some optional native dependency loads failed before main library load: %s",
                    String.join(" | ", errors));
        }
    }

    private static List<String> dependencyLoadOrder(Path libraryDir) {
        Set<String> names = new LinkedHashSet<>(nativeDependencyFileNames());
        try (var paths = Files.list(libraryDir)) {
            paths.filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(nativeLibExt()))
                    .filter(name -> name.startsWith("libggml") || name.startsWith("ggml"))
                    .sorted()
                    .forEach(names::add);
        } catch (IOException ignored) {
            // Keep static dependency order.
        }
        return new ArrayList<>(names);
    }

    private static List<Path> candidateRuntimeDirs() {
        List<Path> dirs = new ArrayList<>();
        optionalEnv("GOLEK_LLAMA_LIB_PATH").map(Path::of).map(Path::toAbsolutePath).map(Path::normalize)
                .map(Path::getParent).ifPresent(dirs::add);
        optionalEnv("GOLEK_LLAMA_LIB_DIR").map(Path::of).map(Path::toAbsolutePath).map(Path::normalize)
                .ifPresent(dirs::add);
        dirs.add(Path.of(".").toAbsolutePath().normalize());
        dirs.add(Path.of("native-libs").toAbsolutePath().normalize());
        dirs.add(Path.of("lib").toAbsolutePath().normalize());
        dirs.add(Path.of(System.getProperty("user.home"), ".gollek", "native-libs").toAbsolutePath().normalize());

        ProcessHandle.current().info().command().ifPresent(command -> {
            Path binary = Path.of(command).toAbsolutePath().normalize();
            Path parent = binary.getParent();
            if (parent != null) {
                dirs.add(parent);
                dirs.add(parent.resolve("native-libs"));
                dirs.add(parent.resolve("lib"));
            }
        });

        // Development workspace paths (walk up from cwd).
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        for (int i = 0; i < 8 && current != null; i++) {
            dirs.add(current.resolve("extension/format/gguf/gollek-ext-format-gguf/target/llama-cpp/lib"));
            dirs.add(current.resolve("inference/format/gguf/source/llama-cpp/llama.cpp/build/bin"));
            current = current.getParent();
        }
        return dirs;
    }

    private static List<String> nativeDependencyFileNames() {
        String ext = nativeLibExt();
        List<String> names = new ArrayList<>();
        if (isWindows()) {
            names.add("ggml-base" + ext);
            names.add("ggml-cpu" + ext);
            names.add("ggml-blas" + ext);
            names.add("ggml-cuda" + ext);
            names.add("ggml-metal" + ext);
            names.add("ggml" + ext);
            names.add("llama" + ext);
            names.add("libggml-base" + ext);
            names.add("libggml-cpu" + ext);
            names.add("libggml-blas" + ext);
            names.add("libggml-cuda" + ext);
            names.add("libggml-metal" + ext);
            names.add("libggml" + ext);
            names.add("libllama" + ext);
        } else {
            names.add("libggml-base" + ext);
            names.add("libggml-cpu" + ext);
            names.add("libggml-blas" + ext);
            names.add("libggml-cuda" + ext);
            names.add("libggml-cuda.0" + ext);
            names.add("libggml-vulkan" + ext);
            names.add("libggml-vulkan.0" + ext);
            names.add("libggml-metal" + ext);
            names.add("libggml-metal.0" + ext);
            names.add("libggml-opencl" + ext);
            names.add("libggml-opencl.0" + ext);
            names.add("libggml" + ext);
            names.add("libggml.0" + ext);
            names.add("libggml-base.0" + ext);
            names.add("libggml-cpu.0" + ext);
            names.add("libggml-blas.0" + ext);
            names.add("libllama" + ext);
            names.add("libllama.0" + ext);
            names.add(shimLibraryFileName());
        }
        return names;
    }

    private static Optional<String> optionalEnv(String key) {
        String value = System.getenv(key);
        if (value == null || value.isBlank()) {
            return Optional.empty();
        }
        return Optional.of(value.trim());
    }

    private static String shimLibraryFileName() {
        if (isWindows()) {
            return "gollek_llama_shim" + nativeLibExt();
        }
        return "libgollek_llama_shim" + nativeLibExt();
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().contains("win");
    }

    private static boolean isMac() {
        return System.getProperty("os.name").toLowerCase().contains("mac");
    }

    private static String nativeLibExt() {
        if (isWindows()) {
            return ".dll";
        }
        if (isMac()) {
            return ".dylib";
        }
        return ".so";
    }

    private static InputStream getResourceAsStream(String path) {
        // 1. Try LlamaCppBinding classloader (absolute)
        InputStream is = LlamaCppBinding.class.getResourceAsStream(path);
        if (is != null)
            return is;

        // 2. Try LlamaCppBinding classloader (relative to root)
        String noSlash = path.startsWith("/") ? path.substring(1) : path;
        is = LlamaCppBinding.class.getClassLoader().getResourceAsStream(noSlash);
        if (is != null)
            return is;

        // 3. Try Thread context classloader
        ClassLoader tcl = Thread.currentThread().getContextClassLoader();
        if (tcl != null) {
            is = tcl.getResourceAsStream(noSlash);
            if (is != null)
                return is;
        }

        return null;
    }

    // ===================================================================
    // Public API Methods
    // ===================================================================

    public void backendInit() {
        if (backendInitialized.compareAndSet(false, true)) {
            try {
                checkHandle(llama_backend_init, "llama_backend_init");
                log.info("Initializing llama.cpp backend...");
                llama_backend_init.invoke();
            } catch (Throwable e) {
                backendInitialized.set(false);
                throw new RuntimeException("Failed to initialize llama backend", e);
            }
        }
    }

    public void backendFree() {
        if (backendInitialized.compareAndSet(true, false)) {
            try {
                if (llama_backend_free != null) {
                    log.info("Freeing llama.cpp backend");
                    llama_backend_free.invoke();
                }
            } catch (Throwable e) {
                log.error("Failed to free llama backend", e);
            }
        }
    }

    private void checkHandle(MethodHandle handle, String name) {
        if (handle == null) {
            throw new IllegalStateException(
                    "Native function " + name + " not found. The native library may be incompatible or incomplete.");
        }
    }

    public MemorySegment getDefaultModelParams() {
        try {
            MemorySegment params = arena.allocate(LLAMA_MODEL_PARAMS_LAYOUT);
            if (gollek_llama_model_default_params_into != null) {
                gollek_llama_model_default_params_into.invoke(params);
            } else if (llama_model_default_params != null) {
                MemorySegment defaults = (MemorySegment) llama_model_default_params.invoke((SegmentAllocator) arena);
                MemorySegment.copy(defaults, 0, params, 0, LLAMA_MODEL_PARAMS_LAYOUT.byteSize());
            } else {
                // Conservative defaults if native default-params helper is unavailable in
                // native
                // image linking.
                setModelParam(params, "n_gpu_layers", 0);
                setModelParam(params, "split_mode", 0);
                // No GPU devices available in many native CPU-only environments.
                setModelParam(params, "main_gpu", -1);
                setModelParam(params, "use_mmap", true);
                setModelParam(params, "use_direct_io", false);
                setModelParam(params, "use_mlock", false);
                setModelParam(params, "check_tensors", false);
                setModelParam(params, "use_extra_bufts", false);
                setModelParam(params, "no_host", false);
                setModelParam(params, "no_alloc", false);
            }
            return params;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get default model params", e);
        }
    }

    public MemorySegment getDefaultContextParams() {
        try {
            MemorySegment params = arena.allocate(LLAMA_CONTEXT_PARAMS_LAYOUT);
            if (gollek_llama_context_default_params_into != null) {
                gollek_llama_context_default_params_into.invoke(params);
            } else if (llama_context_default_params != null) {
                MemorySegment defaults = (MemorySegment) llama_context_default_params.invoke((SegmentAllocator) arena);
                MemorySegment.copy(defaults, 0, params, 0, LLAMA_CONTEXT_PARAMS_LAYOUT.byteSize());
            } else {
                // Conservative defaults if native default-params helper is unavailable in
                // native
                // image linking.
                setContextParam(params, "n_ctx", 4096);
                setContextParam(params, "n_batch", 512);
                setContextParam(params, "n_ubatch", 512);
                setContextParam(params, "n_seq_max", 1);
                setContextParam(params, "n_threads", Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
                setContextParam(params, "n_threads_batch", Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
                setContextParam(params, "rope_scaling_type", 0);
                // llama.cpp default for causal LLM context is "none" (-1), not 0.
                setContextParam(params, "pooling_type", -1);
                setContextParam(params, "attention_type", 0);
                setContextParam(params, "flash_attn_type", 0);
                setContextParam(params, "embeddings", false);
                setContextParam(params, "offload_kqv", false);
                setContextParam(params, "no_perf", false);
                setContextParam(params, "op_offload", false);
                setContextParam(params, "swa_full", false);
                setContextParam(params, "kv_unified", false);
                setContextParam(params, "samplers", MemorySegment.NULL);
                setContextParam(params, "n_samplers", 0L);
            }
            return params;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get default context params", e);
        }
    }

    public MemorySegment loadModel(String path, MemorySegment modelParams) {
        try {
            checkHandle(llama_load_model_from_file, "llama_load_model_from_file");
            MemorySegment pathSegment = arena.allocateFrom(path);
            MemorySegment model = (MemorySegment) llama_load_model_from_file.invoke(pathSegment, modelParams);

            if (model.address() == 0) {
                throw new RuntimeException("Failed to load model from: " + path);
            }

            return model;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }

    public MemorySegment createContext(MemorySegment model, MemorySegment contextParams) {
        try {
            checkHandle(llama_new_context_with_model, "llama_new_context_with_model");
            MemorySegment context = (MemorySegment) llama_new_context_with_model.invoke(model, contextParams);

            if (context.address() == 0) {
                throw new RuntimeException("Failed to create context");
            }

            return context;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create context", e);
        }
    }

    public void freeModel(MemorySegment model) {
        try {
            llama_free_model.invoke(model);
        } catch (Throwable e) {
            log.error("Failed to free model", e);
        }
    }

    public void freeContext(MemorySegment context) {
        try {
            llama_free.invoke(context);
        } catch (Throwable e) {
            log.error("Failed to free context", e);
        }
    }

    public int[] tokenize(MemorySegment model, String text, boolean addBos, boolean special) {
        try (Arena localArena = Arena.ofConfined()) {
            // Get vocab from model (new API)
            MemorySegment vocab = getVocab(model);

            // Convert text to UTF-8 C-string
            MemorySegment textSegment = localArena.allocateFrom(text);
            // Get actual byte length (excluding null terminator)
            int textLenBytes = (int) textSegment.byteSize() - 1;

            // First call to get token count (negative value indicates needed buffer size)
            int tokenCount = (int) llama_tokenize.invoke(
                    vocab, textSegment, textLenBytes,
                    MemorySegment.NULL, 0, addBos, special);

            // Handle negative return value (buffer too small)
            int bufferSize = tokenCount < 0 ? -tokenCount : tokenCount + 32;

            // Allocate buffer and tokenize
            MemorySegment tokensBuffer = localArena.allocate(ValueLayout.JAVA_INT, bufferSize);

            int actualCount = (int) llama_tokenize.invoke(
                    vocab, textSegment, textLenBytes,
                    tokensBuffer, bufferSize, addBos, special);

            if (actualCount < 0) {
                throw new RuntimeException("Tokenization failed with error code: " + actualCount);
            }

            // Copy to Java array
            int[] tokens = new int[actualCount];
            for (int i = 0; i < actualCount; i++) {
                tokens[i] = tokensBuffer.getAtIndex(ValueLayout.JAVA_INT, i);
            }

            // DEBUG LOGGING
            if (log.isDebugEnabled()) {
                StringBuilder sb = new StringBuilder("Tokenized: '");
                if (text.length() > 50)
                    sb.append(text.substring(0, 50)).append("...");
                else
                    sb.append(text);
                sb.append("' -> [");
                for (int t : tokens)
                    sb.append(t).append(", ");
                sb.append("]");
                log.debug(sb.toString());
            }

            return tokens;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to tokenize text", e);
        }
    }

    public String tokenToPiece(MemorySegment model, int token) {
        try (Arena localArena = Arena.ofConfined()) {
            MemorySegment vocab = getVocab(model);
            MemorySegment tokenBuf = localArena.allocate(ValueLayout.JAVA_INT, 1);
            tokenBuf.setAtIndex(ValueLayout.JAVA_INT, 0, token);

            MemorySegment buffer = localArena.allocate(ValueLayout.JAVA_BYTE, 256);
            int length;

            if (llama_token_to_piece != null) {
                length = (int) llama_token_to_piece.invoke(vocab, token, buffer, 256, 0, false);
                if (length < 0 || length > 256) {
                    int needed = length > 256 ? length : Math.abs(length);
                    buffer = localArena.allocate(ValueLayout.JAVA_BYTE, needed);
                    length = (int) llama_token_to_piece.invoke(vocab, token, buffer, needed, 0, false);
                }
            } else {
                // Fallback for libraries that only expose llama_detokenize.
                length = (int) llama_detokenize.invoke(vocab, tokenBuf, 1, buffer, 256, false, true);
                if (length < 0 || length > 256) {
                    int needed = length > 256 ? length : Math.abs(length);
                    buffer = localArena.allocate(ValueLayout.JAVA_BYTE, needed);
                    length = (int) llama_detokenize.invoke(vocab, tokenBuf, 1, buffer, needed, false, true);
                }
            }

            if (length <= 0)
                return "";

            byte[] bytes = new byte[length];
            MemorySegment.copy(buffer, 0, MemorySegment.ofArray(bytes), 0, length);
            return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to decode token", e);
        }
    }

    public int decode(MemorySegment context, MemorySegment batch) {
        try {
            return (int) llama_decode.invoke(context, batch);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to decode", e);
        }
    }

    public MemorySegment getLogits(MemorySegment context) {
        try {
            return (MemorySegment) llama_get_logits.invoke(context);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get logits", e);
        }
    }

    public MemorySegment getLogitsIth(MemorySegment context, int index) {
        try {
            return (MemorySegment) llama_get_logits_ith.invoke(context, index);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get logits at index", e);
        }
    }

    /**
     * Clear the KV cache
     */
    public void kvCacheClear(MemorySegment context) {
        try {
            checkHandle(llama_get_memory, "llama_get_memory");
            checkHandle(llama_memory_clear, "llama_memory_clear");

            MemorySegment memory = (MemorySegment) llama_get_memory.invoke(context);
            llama_memory_clear.invoke(memory, true);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to clear KV cache", e);
        }
    }

    /**
     * Get vocab pointer from model. Required for tokenization in new API.
     */
    public MemorySegment getVocab(MemorySegment model) {
        try {
            checkHandle(llama_model_get_vocab, "llama_model_get_vocab");
            return (MemorySegment) llama_model_get_vocab.invoke(model);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get vocab from model", e);
        }
    }

    public int getEosToken(MemorySegment model) {
        try {
            MemorySegment vocab = getVocab(model);
            return (int) llama_vocab_eos.invoke(vocab);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get EOS token", e);
        }
    }

    public int getBosToken(MemorySegment model) {
        try {
            MemorySegment vocab = getVocab(model);
            return (int) llama_vocab_bos.invoke(vocab);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get BOS token", e);
        }
    }

    public MemorySegment batchInit(int nTokens, int embd, int nSeqMax) {
        try {
            Arena batchArena = Arena.ofConfined();
            MemorySegment batch = batchArena.allocate(LLAMA_BATCH_LAYOUT);
            MemorySegment token = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment pos = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment nSeqId = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment logits = batchArena.allocate(ValueLayout.JAVA_BYTE, nTokens);
            MemorySegment seqIdFlat = batchArena.allocate(ValueLayout.JAVA_INT, nTokens);
            MemorySegment seqIdPtr = batchArena.allocate(ValueLayout.ADDRESS, nTokens);

            for (int i = 0; i < nTokens; i++) {
                seqIdPtr.setAtIndex(ValueLayout.ADDRESS, i,
                        seqIdFlat.asSlice((long) i * ValueLayout.JAVA_INT.byteSize(), ValueLayout.JAVA_INT.byteSize()));
            }

            batch.set(ValueLayout.JAVA_INT,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")), nTokens);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("token")), token);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("embd")), MemorySegment.NULL);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("pos")), pos);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_seq_id")), nSeqId);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("seq_id")), seqIdPtr);
            batch.set(ValueLayout.ADDRESS,
                    LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("logits")), logits);
            manualBatchArenas.put(batch.address(), batchArena);
            manualBatchBuffers.put(batch.address(), new ManualBatchBuffers(token, pos, nSeqId, seqIdPtr, logits));
            return batch;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize batch", e);
        }
    }

    public void batchFree(MemorySegment batch) {
        try {
            manualBatchBuffers.remove(batch.address());
            Arena batchArena = manualBatchArenas.remove(batch.address());
            if (batchArena != null) {
                batchArena.close();
                return;
            }
            if (llama_batch_free != null) {
                llama_batch_free.invoke(batch);
            }
        } catch (Throwable e) {
            log.error("Failed to free batch", e);
        }
    }

    public int getContextSize(MemorySegment context) {
        try {
            return (int) llama_n_ctx.invoke(context);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get context size", e);
        }
    }

    public int getVocabSize(MemorySegment model) {
        try {
            MemorySegment vocab = getVocab(model);
            return (int) llama_vocab_n_tokens.invoke(vocab);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get vocab size", e);
        }
    }

    // Sampler Methods

    public MemorySegment createSamplerChain() {
        try {
            MemorySegment params = (MemorySegment) llama_sampler_chain_default_params
                    .invokeExact((SegmentAllocator) arena);
            // We need to pass the struct by value.
            // Since invokeExact with SegmentAllocator returns the struct segment, we can't
            // use it directly in the next call if it expects by value...
            // Wait, standard downcall for struct-by-value arg expects a MemorySegment
            // containing the struct.
            // For return value struct, it needs an allocator.
            // Let's use invoke with allocator correctly.
            // Actually, if we use just invoke on a function returning struct, does it
            // require allocator?
            // Yes, standard Linker requires SegmentAllocator as first arg for functions
            // returning struct.

            // However, we can construct cleaner bindings if we do it manually.
            // Simplified: manually init params
            MemorySegment paramsSeg = arena.allocate(LLAMA_SAMPLER_CHAIN_PARAMS_LAYOUT);
            paramsSeg.set(ValueLayout.JAVA_BYTE, 0, (byte) 0); // no_perf = false

            return (MemorySegment) llama_sampler_chain_init.invoke(paramsSeg);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create sampler chain", e);
        }
    }

    public void addGreedySampler(MemorySegment chain) {
        try {
            MemorySegment greedy = (MemorySegment) llama_sampler_init_greedy.invoke();
            llama_sampler_chain_add.invoke(chain, greedy);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add greedy sampler", e);
        }
    }

    public void addTopKSampler(MemorySegment chain, int k) {
        try {
            MemorySegment sampler = (MemorySegment) llama_sampler_init_top_k.invoke(k);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add top-k sampler", e);
        }
    }

    public void addTopPSampler(MemorySegment chain, float p, long minKeep) {
        try {
            MemorySegment sampler = (MemorySegment) llama_sampler_init_top_p.invoke(p, minKeep);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add top-p sampler", e);
        }
    }

    public void addMinPSampler(MemorySegment chain, float p, long minKeep) {
        try {
            MemorySegment sampler = (MemorySegment) llama_sampler_init_min_p.invoke(p, minKeep);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add min-p sampler", e);
        }
    }

    public void addTempSampler(MemorySegment chain, float temp) {
        try {
            MemorySegment sampler = (MemorySegment) llama_sampler_init_temp.invoke(temp);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add temp sampler", e);
        }
    }

    public void addDistSampler(MemorySegment chain, int seed) {
        try {
            MemorySegment sampler = (MemorySegment) llama_sampler_init_dist.invoke(seed);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add dist sampler", e);
        }
    }

    public int sample(MemorySegment sampler, MemorySegment context, int index) {
        try {
            return (int) llama_sampler_sample.invoke(sampler, context, index);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to sample", e);
        }
    }

    public void freeSampler(MemorySegment sampler) {
        try {
            llama_sampler_free.invoke(sampler);
        } catch (Throwable e) {
            log.error("Failed to free sampler", e);
        }
    }

    // --- Extended Sampler Methods ---

    /**
     * Add penalties sampler (repeat, frequency, presence) to the chain.
     * 
     * @param penaltyLastN     last n tokens to penalize (0 = disable, -1 = context
     *                         size)
     * @param repeatPenalty    1.0 = disabled
     * @param frequencyPenalty 0.0 = disabled
     * @param presencePenalty  0.0 = disabled
     */
    public void addPenaltiesSampler(MemorySegment chain, int penaltyLastN,
            float repeatPenalty, float frequencyPenalty, float presencePenalty) {
        try {
            checkHandle(llama_sampler_init_penalties, "llama_sampler_init_penalties");
            MemorySegment sampler = (MemorySegment) llama_sampler_init_penalties.invoke(
                    penaltyLastN, repeatPenalty, frequencyPenalty, presencePenalty);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add penalties sampler", e);
        }
    }

    /**
     * Add Mirostat v1 sampler to the chain.
     * 
     * @param nVocab vocabulary size
     * @param seed   RNG seed
     * @param tau    target cross-entropy
     * @param eta    learning rate
     * @param m      number of tokens for s_hat estimation
     */
    public void addMirostatSampler(MemorySegment chain, int nVocab, int seed,
            float tau, float eta, int m) {
        try {
            checkHandle(llama_sampler_init_mirostat, "llama_sampler_init_mirostat");
            MemorySegment sampler = (MemorySegment) llama_sampler_init_mirostat.invoke(
                    nVocab, seed, tau, eta, m);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add mirostat sampler", e);
        }
    }

    /**
     * Add Mirostat v2 sampler to the chain.
     * 
     * @param seed RNG seed
     * @param tau  target cross-entropy
     * @param eta  learning rate
     */
    public void addMirostatV2Sampler(MemorySegment chain, int seed, float tau, float eta) {
        try {
            checkHandle(llama_sampler_init_mirostat_v2, "llama_sampler_init_mirostat_v2");
            MemorySegment sampler = (MemorySegment) llama_sampler_init_mirostat_v2.invoke(seed, tau, eta);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add mirostat v2 sampler", e);
        }
    }

    /**
     * Add grammar (GBNF) sampler to the chain for constrained output.
     * 
     * @param model       the model (used to get vocab)
     * @param grammarStr  GBNF grammar string
     * @param grammarRoot root rule name (typically "root")
     */
    public void addGrammarSampler(MemorySegment chain, MemorySegment model,
            String grammarStr, String grammarRoot) {
        try {
            checkHandle(llama_sampler_init_grammar, "llama_sampler_init_grammar");
            MemorySegment vocab = getVocab(model);
            MemorySegment grammarStrSeg = arena.allocateFrom(grammarStr);
            MemorySegment grammarRootSeg = arena.allocateFrom(grammarRoot);
            MemorySegment sampler = (MemorySegment) llama_sampler_init_grammar.invoke(
                    vocab, grammarStrSeg, grammarRootSeg);
            if (sampler == null || sampler.equals(MemorySegment.NULL)) {
                throw new RuntimeException(
                        "Grammar parse failed for: " + grammarStr.substring(0, Math.min(80, grammarStr.length())));
            }
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add grammar sampler", e);
        }
    }

    /**
     * Add typical sampling to the chain.
     * 
     * @param p       typical p value
     * @param minKeep minimum tokens to keep
     */
    public void addTypicalSampler(MemorySegment chain, float p, long minKeep) {
        try {
            checkHandle(llama_sampler_init_typical, "llama_sampler_init_typical");
            MemorySegment sampler = (MemorySegment) llama_sampler_init_typical.invoke(p, minKeep);
            llama_sampler_chain_add.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add typical sampler", e);
        }
    }

    /**
     * Check if a token is an end-of-generation token (EOS, EOT, etc.).
     */
    public boolean isEndOfGeneration(MemorySegment model, int tokenId) {
        try {
            checkHandle(llama_vocab_is_eog, "llama_vocab_is_eog");
            MemorySegment vocab = getVocab(model);
            return (int) llama_vocab_is_eog.invoke(vocab, tokenId) != 0;
        } catch (Throwable e) {
            // Fallback: just check EOS
            return tokenId == getEosToken(model);
        }
    }

    // Batch Manipulation Helpers

    public void setBatchSize(MemorySegment batch, int nTokens) {
        batch.set(ValueLayout.JAVA_INT, 0, nTokens);
    }

    public void setBatchToken(MemorySegment batch, int index, int token, int pos, int seqId, boolean outputLogits) {
        ManualBatchBuffers manual = manualBatchBuffers.get(batch.address());
        if (manual != null) {
            manual.token.setAtIndex(ValueLayout.JAVA_INT, index, token);
            manual.pos.setAtIndex(ValueLayout.JAVA_INT, index, pos);
            manual.nSeqId.setAtIndex(ValueLayout.JAVA_INT, index, 1);
            MemorySegment seqIdsForToken = manual.seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                    .reinterpret(ValueLayout.JAVA_INT.byteSize());
            seqIdsForToken.set(ValueLayout.JAVA_INT, 0, seqId);
            manual.logits.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (outputLogits ? 1 : 0));
            return;
        }

        // Access pointers from struct
        // Offset 8: token pointer (n_tokens 4 bytes + 4 bytes padding)
        MemorySegment tokenPtr = batch.get(ValueLayout.ADDRESS, 8)
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        tokenPtr.setAtIndex(ValueLayout.JAVA_INT, index, token);

        // Offset 24: pos pointer (8 + 8 + 8) -> 0:n_tokens(4+4), 8:token, 16:embd,
        // 24:pos
        MemorySegment posPtr = batch.get(ValueLayout.ADDRESS, 24)
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        posPtr.setAtIndex(ValueLayout.JAVA_INT, index, pos);

        // Offset 32: n_seq_id pointer
        MemorySegment nSeqIdPtr = batch.get(ValueLayout.ADDRESS, 32)
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        nSeqIdPtr.setAtIndex(ValueLayout.JAVA_INT, index, 1); // We assume 1 sequence per token for now

        // Offset 40: seq_id pointer (pointer to pointer)
        MemorySegment seqIdPtr = batch.get(ValueLayout.ADDRESS, 40)
                .reinterpret((long) (index + 1) * ValueLayout.ADDRESS.byteSize());
        MemorySegment seqIdsForToken = seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                .reinterpret(ValueLayout.JAVA_INT.byteSize());
        seqIdsForToken.set(ValueLayout.JAVA_INT, 0, seqId);

        // Offset 48: logits pointer
        MemorySegment logitsPtr = batch.get(ValueLayout.ADDRESS, 48)
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_BYTE.byteSize());
        logitsPtr.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (outputLogits ? 1 : 0));
    }
    // ===================================================================
    // Struct Accessors
    // ===================================================================

    public void setModelParam(MemorySegment params, String name, Object value) {
        setParam(LLAMA_MODEL_PARAMS_LAYOUT, params, name, value);
    }

    public void setContextParam(MemorySegment params, String name, Object value) {
        setParam(LLAMA_CONTEXT_PARAMS_LAYOUT, params, name, value);
    }

    private void setParam(MemoryLayout layout, MemorySegment segment, String name, Object value) {
        long offset = layout.byteOffset(MemoryLayout.PathElement.groupElement(name));
        if (value instanceof Integer i) {
            segment.set(ValueLayout.JAVA_INT, offset, i);
        } else if (value instanceof Long l) {
            segment.set(ValueLayout.JAVA_LONG, offset, l);
        } else if (value instanceof Float f) {
            segment.set(ValueLayout.JAVA_FLOAT, offset, f);
        } else if (value instanceof Boolean b) {
            segment.set(ValueLayout.JAVA_BYTE, offset, (byte) ((Boolean) value ? 1 : 0));
        } else if (value instanceof MemorySegment ms) {
            segment.set(ValueLayout.ADDRESS, offset, ms);
        } else {
            throw new IllegalArgumentException("Unsupported type: " + value.getClass());
        }
    }

    public void setBatchToken(MemorySegment batch, int index, int token) {
        ManualBatchBuffers manual = manualBatchBuffers.get(batch.address());
        if (manual != null) {
            manual.token.setAtIndex(ValueLayout.JAVA_INT, index, token);
            return;
        }
        MemorySegment tokenPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("token")))
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        tokenPtr.setAtIndex(ValueLayout.JAVA_INT, index, token);
    }

    public void setBatchPos(MemorySegment batch, int index, int pos) {
        ManualBatchBuffers manual = manualBatchBuffers.get(batch.address());
        if (manual != null) {
            manual.pos.setAtIndex(ValueLayout.JAVA_INT, index, pos);
            return;
        }
        MemorySegment posPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("pos")))
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        posPtr.setAtIndex(ValueLayout.JAVA_INT, index, pos);
    }

    public void setBatchSeqId(MemorySegment batch, int index, int seqId) {
        ManualBatchBuffers manual = manualBatchBuffers.get(batch.address());
        if (manual != null) {
            manual.nSeqId.setAtIndex(ValueLayout.JAVA_INT, index, 1);
            MemorySegment seqIdPtr = manual.seqIdPtr.getAtIndex(ValueLayout.ADDRESS, index)
                    .reinterpret(ValueLayout.JAVA_INT.byteSize());
            seqIdPtr.set(ValueLayout.JAVA_INT, 0, seqId);
            return;
        }
        MemorySegment nSeqIdPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_seq_id")))
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_INT.byteSize());
        nSeqIdPtr.setAtIndex(ValueLayout.JAVA_INT, index, 1);

        MemorySegment seqIdPtrPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("seq_id")))
                .reinterpret((long) (index + 1) * ValueLayout.ADDRESS.byteSize());
        MemorySegment seqIdPtr = seqIdPtrPtr.getAtIndex(ValueLayout.ADDRESS, index)
                .reinterpret(ValueLayout.JAVA_INT.byteSize());
        seqIdPtr.set(ValueLayout.JAVA_INT, 0, seqId);
    }

    public void setBatchLogits(MemorySegment batch, int index, boolean enable) {
        ManualBatchBuffers manual = manualBatchBuffers.get(batch.address());
        if (manual != null) {
            manual.logits.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (enable ? 1 : 0));
            return;
        }
        MemorySegment logitsPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("logits")))
                .reinterpret((long) (index + 1) * ValueLayout.JAVA_BYTE.byteSize());
        logitsPtr.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (enable ? 1 : 0));
    }

    public void setBatchNTokens(MemorySegment batch, int nTokens) {
        batch.set(ValueLayout.JAVA_INT,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")), nTokens);
    }

    public int getBatchNTokensCount(MemorySegment batch) {
        return batch.get(ValueLayout.JAVA_INT,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_tokens")));
    }

    public Arena getArena() {
        return arena;
    }

    public String getModelMetadata(MemorySegment model, String key) {
        try {
            checkHandle(llama_model_meta_val_str, "llama_model_meta_val_str");

            MemorySegment keySegment = arena.allocateFrom(key);

            // First call to get size (buf_size = 0)
            int size = (int) llama_model_meta_val_str.invoke(model, keySegment, MemorySegment.NULL, 0L);

            if (size < 0) {
                // Key not found
                return null;
            }

            // Allocate buffer (+1 for null terminator just in case, though size usually
            // includes it for C strings logic or we handle Java string creation)
            // llama_model_meta_val_str returns length of string.
            MemorySegment buf = arena.allocate(size + 1);

            int result = (int) llama_model_meta_val_str.invoke(model, keySegment, buf, (long) (size + 1));

            if (result < 0) {
                return null;
            }

            return buf.getString(0);
        } catch (Throwable e) {
            log.warnf("Failed to get metadata key %s: %s", key, e.getMessage());
            return null;
        }
    }

    public void close() {
        backendFree();
        arena.close();
    }
}
