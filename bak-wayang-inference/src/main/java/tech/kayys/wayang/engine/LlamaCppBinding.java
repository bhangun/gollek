package tech.kayys.golek.engine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class LlamaCppBinding {
    private static final Logger log = LoggerFactory.getLogger(LlamaCppBinding.class);

    private final SymbolLookup lookup;
    private final Linker linker;

    // Core functions
    private final MethodHandle llamaBackendInit;
    private final MethodHandle llamaBackendFree;
    private final MethodHandle llamaModelDefaultParams;
    private final MethodHandle llamaLoadModelFromFile;
    private final MethodHandle llamaFreeModel;
    private final MethodHandle llamaContextDefaultParams;
    private final MethodHandle llamaNewContextWithModel;
    private final MethodHandle llamaFreeContext;

    // Model info
    private final MethodHandle llamaNVocab;
    private final MethodHandle llamaNCtxTrain;
    private final MethodHandle llamaNEmbd;
    private final MethodHandle llamaModelDesc;
    private final MethodHandle llamaModelSize;
    private final MethodHandle llamaModelNParams;

    // Tokenization
    private final MethodHandle llamaTokenize;
    private final MethodHandle llamaTokenToPiece;
    private final MethodHandle llamaTokenBos;
    private final MethodHandle llamaTokenEos;
    private final MethodHandle llamaTokenNl;
    private final MethodHandle llamaAddBosToken;
    private final MethodHandle llamaAddEosToken;

    // Inference
    private final MethodHandle llamaDecode;
    private final MethodHandle llamaGetLogitsIth;
    private final MethodHandle llamaGetEmbeddingsIth;
    private final MethodHandle llamaGetEmbeddingsSeq;

    // KV cache
    private MethodHandle llamaKvCacheClear;
    private MethodHandle llamaKvCacheSeqRm;
    private MethodHandle llamaKvCacheSeqCp;
    private MethodHandle llamaKvCacheSeqKeep;
    private MethodHandle llamaKvCacheSeqAdd;
    private MethodHandle llamaKvCacheSeqDiv;
    private MethodHandle llamaKvCacheDefrag; // Remove final
    private MethodHandle llamaKvCacheUpdate; // Remove final

    // State save/load
    private final MethodHandle llamaStateGetSize;
    private final MethodHandle llamaStateSaveFile;
    private final MethodHandle llamaStateLoadFile;

    // Performance
    private final MethodHandle llamaPerfContextPrint;
    private final MethodHandle llamaPerfContextReset;

    // Memory management
    private final MethodHandle llamaBatchInit;
    private final MethodHandle llamaBatchFree;

    // Platform detection
    private static final String OS_NAME = System.getProperty("os.name").toLowerCase();
    private static final boolean IS_MACOS = OS_NAME.contains("mac") || OS_NAME.contains("darwin");

    // Core function names - define once, resolve per platform
    private static final List<String> CORE_FUNCTIONS = List.of(
            "llama_backend_init",
            "llama_backend_free",
            "llama_load_model_from_file",
            "llama_free_model",
            "llama_model_default_params",
            "llama_context_default_params",
            "llama_new_context_with_model",
            "llama_free_context",
            "llama_n_vocab",
            "llama_n_ctx_train",
            "llama_n_embd",
            "llama_model_desc",
            "llama_model_size",
            "llama_model_n_params",
            "llama_tokenize",
            "llama_token_to_piece",
            "llama_token_bos",
            "llama_token_eos",
            "llama_token_nl",
            "llama_add_bos_token",
            "llama_add_eos_token",
            "llama_decode",
            "llama_get_logits_ith",
            "llama_get_embeddings_ith",
            "llama_get_embeddings_seq",
            "llama_kv_cache_clear",
            "llama_kv_cache_seq_rm",
            "llama_kv_cache_seq_cp",
            "llama_kv_cache_seq_keep",
            "llama_kv_cache_seq_add",
            "llama_kv_cache_seq_div",
            "llama_kv_cache_defrag",
            "llama_kv_cache_update",
            "llama_state_get_size",
            "llama_state_save_file",
            "llama_state_load_file",
            "llama_perf_context_print",
            "llama_perf_context_reset",
            "llama_batch_init",
            "llama_batch_free");

    // C++ mangled names for when C API isn't available
    private static final Map<String, String> CPP_MANGLED_NAMES = Map.ofEntries(
            Map.entry("llama_kv_cache_clear", "__ZN14llama_kv_cache5clearEb"),
            Map.entry("llama_kv_cache_seq_rm", "__ZN14llama_kv_cache6seq_rmEiii"),
            Map.entry("llama_kv_cache_seq_cp", "__ZN14llama_kv_cache6seq_cpEiiii"),
            Map.entry("llama_kv_cache_seq_keep", "__ZN14llama_kv_cache8seq_keepEi"),
            Map.entry("llama_kv_cache_seq_add", "__ZN14llama_kv_cache7seq_addEiiii"),
            Map.entry("llama_kv_cache_seq_div", "__ZN14llama_kv_cache7seq_divEiiii"));

    private static List<String> getPossibleFunctionNames(String baseName) {
        List<String> names = new ArrayList<>();
        if (IS_MACOS) {
            names.add("_" + baseName);
        }
        names.add(baseName);
        return names;
    }

    public LlamaCppBinding(String libraryPath) {
        log.info("Loading llama.cpp library on {}: {}", OS_NAME, libraryPath);

        if (!Files.exists(Path.of(libraryPath))) {
            throw new RuntimeException("Library not found: " + libraryPath);
        }

        System.load(libraryPath);
        this.linker = Linker.nativeLinker();
        this.lookup = SymbolLookup.loaderLookup();

        try {
            // Core functions
            this.llamaBackendInit = findFunction("llama_backend_init",
                    FunctionDescriptor.ofVoid());
            this.llamaBackendFree = findFunction("llama_backend_free",
                    FunctionDescriptor.ofVoid());

            // Test multiple signatures for model loading functions
            this.llamaModelDefaultParams = findFunctionWithSignatures(
                    "llama_model_default_params",
                    new FunctionDescriptor[] {
                            FunctionDescriptor.of(ValueLayout.ADDRESS),
                            FunctionDescriptor.of(LlamaStructs.MODEL_PARAMS_LAYOUT)
                    });

            this.llamaLoadModelFromFile = findFunctionWithSignatures(
                    "llama_load_model_from_file",
                    new FunctionDescriptor[] {
                            // Try common signatures
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS),
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                    LlamaStructs.MODEL_PARAMS_LAYOUT),
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS),
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS)
                    });

            this.llamaFreeModel = findFunction("llama_free_model",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // Test context functions with multiple signatures
            this.llamaContextDefaultParams = findFunctionWithSignatures("llama_model_default_params",
                    new FunctionDescriptor[] {
                            FunctionDescriptor.of(ValueLayout.ADDRESS),
                            FunctionDescriptor.of(LlamaStructs.MODEL_PARAMS_LAYOUT)
                    });

            this.llamaNewContextWithModel = findFunctionWithSignatures(
                    "llama_new_context_with_model",
                    new FunctionDescriptor[] {
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS),
                            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                    LlamaStructs.CONTEXT_PARAMS_LAYOUT)
                    });

            this.llamaFreeContext = findFunction("llama_free",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // Model info
            this.llamaNVocab = findFunction("llama_n_vocab",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaNCtxTrain = findFunction("llama_n_ctx_train",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaNEmbd = findFunction("llama_n_embd",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaModelDesc = findFunction("llama_model_desc",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT));
            this.llamaModelSize = findFunction("llama_model_size",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            this.llamaModelNParams = findFunction("llama_model_n_params",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

            // Tokenization
            this.llamaTokenize = findFunction("llama_tokenize",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN));
            this.llamaTokenToPiece = findFunction("llama_token_to_piece",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaTokenBos = findFunction("llama_token_bos",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaTokenEos = findFunction("llama_token_eos",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaTokenNl = findFunction("llama_token_nl",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            this.llamaAddBosToken = findFunction("llama_add_bos_token",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS));
            this.llamaAddEosToken = findFunction("llama_add_eos_token",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS));

            // Inference
            this.llamaDecode = findFunction("llama_decode",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            this.llamaGetLogitsIth = findFunction("llama_get_logits_ith",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaGetEmbeddingsIth = findFunction("llama_get_embeddings_ith",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaGetEmbeddingsSeq = findFunction("llama_get_embeddings_seq",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // Batch management
            this.llamaBatchInit = findFunction("llama_batch_init",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));
            this.llamaBatchFree = findFunction("llama_batch_free",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // Performance
            this.llamaPerfContextPrint = findFunction("llama_perf_context_print",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            this.llamaPerfContextReset = findFunction("llama_perf_context_reset",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // State management
            this.llamaStateGetSize = findFunction("llama_state_get_size",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            this.llamaStateSaveFile = findFunction("llama_state_save_file",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_BOOLEAN));
            this.llamaStateLoadFile = findFunction("llama_state_load_file",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_BOOLEAN));

            // KV Cache - Use C++ mangled names since C API doesn't exist
            try {
                this.llamaKvCacheClear = findFunction("__ZN14llama_kv_cache5clearEb",
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));
                log.info("Found KV cache clear function");
            } catch (Exception e) {
                log.warn("KV cache clear not available: {}", e.getMessage());
                this.llamaKvCacheClear = null;
            }

            try {
                this.llamaKvCacheSeqRm = findFunction("__ZN14llama_kv_cache6seq_rmEiii",
                        FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
                log.info("Found KV cache seq_rm function");
            } catch (Exception e) {
                log.warn("KV cache seq_rm not available: {}", e.getMessage());
                this.llamaKvCacheSeqRm = null;
            }

            // Initialize other KV cache functions as null for now
            this.llamaKvCacheSeqCp = null;
            this.llamaKvCacheSeqKeep = null;
            this.llamaKvCacheSeqAdd = null;
            this.llamaKvCacheSeqDiv = null;
            this.llamaKvCacheDefrag = null;
            this.llamaKvCacheUpdate = null;

            log.info("Llama.cpp bindings initialized successfully");

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize llama.cpp bindings", e);
        }
    }

    private MethodHandle findFunctionWithSignatures(String name, FunctionDescriptor[] descriptors) {
        for (FunctionDescriptor descriptor : descriptors) {
            try {
                MethodHandle handle = findFunction(name, descriptor);
                log.info("Found function {} with signature: {}", name, descriptor);
                return handle;
            } catch (Exception e) {
                log.debug("Signature {} failed for {}: {}", descriptor, name, e.getMessage());
            }
        }
        throw new RuntimeException("No valid signature found for function: " + name);
    }

    private MethodHandle findFunction(String name, FunctionDescriptor descriptor) {
        List<String> namesToTry = getPossibleFunctionNames(name);

        for (String tryName : namesToTry) {
            Optional<MemorySegment> addr = lookup.find(tryName);
            if (addr.isPresent()) {
                log.debug("Found function: {} as {}", name, tryName);
                return linker.downcallHandle(addr.get(), descriptor);
            }
        }

        throw new RuntimeException("Function not found: " + name + " (tried: " + namesToTry + ")");
    }

    // Debug method to check core functions
    private void debugAvailableCoreFunctions() {
        if (!log.isDebugEnabled())
            return;

        String[] coreFunctions = {
                "llama_backend_init", "llama_backend_free",
                "llama_model_default_params", "llama_load_model_from_file", "llama_free_model",
                "llama_context_default_params", "llama_new_context_with_model", "llama_free",
                "llama_n_vocab", "llama_n_ctx_train", "llama_n_embd",
                "llama_tokenize", "llama_token_to_piece",
                "llama_decode", "llama_get_logits_ith", "llama_get_embeddings_ith",
                "llama_batch_init", "llama_batch_free"
        };

        log.debug("=== Checking core functions ===");
        for (String func : coreFunctions) {
            List<String> namesToTry = getPossibleFunctionNames(func);
            boolean found = false;
            for (String name : namesToTry) {
                if (lookup.find(name).isPresent()) {
                    log.debug("✓ FOUND: {}", name);
                    found = true;
                    break;
                }
            }
            if (!found) {
                log.debug("✗ MISSING: {}", func);
            }
        }
    }

    private long countInitializedMethods() {
        // Count all declared fields in this class (each MethodHandle is a field)
        return this.getClass().getDeclaredFields().length;
    }

    // Core API with improved error handling
    public int backendInit() throws Throwable {
        log.debug("Initializing llama.cpp backend");
        try {
            llamaBackendInit.invoke();
            return 0;
        } catch (Throwable e) {
            log.error("Backend initialization failed", e);
            throw e;
        }
    }

    public void backendFree() throws Throwable {
        log.debug("Freeing llama.cpp backend");
        try {
            llamaBackendFree.invoke();
        } catch (Throwable e) {
            log.error("Backend free failed", e);
            throw e;
        }
    }

    public MemorySegment loadModel(Arena arena, String path, int gpuLayers, boolean useMmap, boolean useMlock)
            throws Throwable {
        log.debug("Loading model: {} (gpu_layers: {}, mmap: {}, mlock: {})",
                path, gpuLayers, useMmap, useMlock);

        if (!Files.exists(Path.of(path))) {
            throw new RuntimeException("Model file not found: " + path);
        }

        // Get default params as struct (not address)
        MemorySegment params = (MemorySegment) llamaModelDefaultParams.invoke();

        // Create a copy in our arena to modify
        MemorySegment paramsCopy = arena.allocate(LlamaStructs.MODEL_PARAMS_LAYOUT);
        MemorySegment.copy(params, 0, paramsCopy, 0, LlamaStructs.MODEL_PARAMS_LAYOUT.byteSize());

        // Set model parameters
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.MODEL_PARAMS_GPU_LAYERS_OFFSET, gpuLayers);
        paramsCopy.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.MODEL_PARAMS_USE_MMAP_OFFSET, useMmap);
        paramsCopy.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.MODEL_PARAMS_USE_MLOCK_OFFSET, useMlock);

        MemorySegment pathSeg = arena.allocateFrom(path, StandardCharsets.UTF_8);
        MemorySegment model = (MemorySegment) llamaLoadModelFromFile.invoke(pathSeg, paramsCopy);

        if (model.address() == 0) {
            throw new RuntimeException("Failed to load model: " + path);
        }

        log.info("Model loaded successfully: {}", path);
        return model;
    }

    // Updated createContext method
    public MemorySegment createContext(Arena arena, MemorySegment model, int ctxSize, int batchSize,
            int threads, int seed, float ropeBase, float ropeScale,
            boolean embeddings, boolean flashAttn) throws Throwable {
        log.debug("Creating context: ctx={}, batch={}, threads={}, embeddings={}",
                ctxSize, batchSize, threads, embeddings);

        // Get default params as struct (not address)
        MemorySegment params = (MemorySegment) llamaContextDefaultParams.invoke();

        // Create a copy in our arena to modify
        MemorySegment paramsCopy = arena.allocate(LlamaStructs.CONTEXT_PARAMS_LAYOUT);
        MemorySegment.copy(params, 0, paramsCopy, 0, LlamaStructs.CONTEXT_PARAMS_LAYOUT.byteSize());

        // Set context parameters
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_SEED_OFFSET, seed);
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_CTX_OFFSET, ctxSize);
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_BATCH_OFFSET, batchSize);
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_THREADS_OFFSET, threads);
        paramsCopy.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_THREADS_BATCH_OFFSET, threads);
        paramsCopy.set(ValueLayout.JAVA_FLOAT, LlamaStructs.CONTEXT_PARAMS_ROPE_FREQ_BASE_OFFSET, ropeBase);
        paramsCopy.set(ValueLayout.JAVA_FLOAT, LlamaStructs.CONTEXT_PARAMS_ROPE_FREQ_SCALE_OFFSET, ropeScale);
        paramsCopy.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.CONTEXT_PARAMS_EMBEDDINGS_OFFSET, embeddings);
        paramsCopy.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.CONTEXT_PARAMS_FLASH_ATTN_OFFSET, flashAttn);

        MemorySegment context = (MemorySegment) llamaNewContextWithModel.invoke(model, paramsCopy);

        if (context.address() == 0) {
            throw new RuntimeException("Failed to create context with size: " + ctxSize);
        }

        log.info("Context created successfully");
        return context;
    }

    public void freeModel(MemorySegment model) throws Throwable {
        if (model != null && model.address() != 0) {
            log.debug("Freeing model");
            try {
                llamaFreeModel.invoke(model);
            } catch (Throwable e) {
                log.error("Model free failed", e);
                throw e;
            }
        }
    }

    public void freeContext(MemorySegment ctx) throws Throwable {
        if (ctx != null && ctx.address() != 0) {
            log.debug("Freeing context");
            try {
                llamaFreeContext.invoke(ctx);
            } catch (Throwable e) {
                log.error("Context free failed", e);
                throw e;
            }
        }
    }

    // Model info
    public int nVocab(MemorySegment model) throws Throwable {
        return (int) llamaNVocab.invoke(model);
    }

    public int nCtxTrain(MemorySegment model) throws Throwable {
        return (int) llamaNCtxTrain.invoke(model);
    }

    public int nEmbd(MemorySegment model) throws Throwable {
        return (int) llamaNEmbd.invoke(model);
    }

    public String modelDesc(Arena arena, MemorySegment model, int bufferSize) throws Throwable {
        MemorySegment buffer = arena.allocate(bufferSize);
        int len = (int) llamaModelDesc.invoke(model, buffer, bufferSize);

        if (len <= 0) {
            return "Unknown";
        }

        return buffer.getString(0, StandardCharsets.UTF_8);
    }

    public String modelDesc(Arena arena, MemorySegment model) throws Throwable {
        return modelDesc(arena, model, 256);
    }

    public long modelSize(MemorySegment model) throws Throwable {
        return (long) llamaModelSize.invoke(model);
    }

    public long modelNParams(MemorySegment model) throws Throwable {
        return (long) llamaModelNParams.invoke(model);
    }

    // Tokenization with improved buffer handling
    public int[] tokenize(Arena arena, MemorySegment model, String text, boolean addBos, boolean special)
            throws Throwable {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }

        MemorySegment textSeg = arena.allocateFrom(text, StandardCharsets.UTF_8);
        int textLen = text.length();

        // Estimate token count - usually 1-2 tokens per word, plus some buffer
        int maxTokensEstimate = Math.max(textLen / 2 + 10, 64);
        MemorySegment tokens = arena.allocate(ValueLayout.JAVA_INT, maxTokensEstimate);

        int n = (int) llamaTokenize.invoke(model, textSeg, textLen, tokens, maxTokensEstimate, addBos, special);

        if (n < 0) {
            // Negative return means buffer too small, retry with larger buffer
            maxTokensEstimate = -n;
            tokens = arena.allocate(ValueLayout.JAVA_INT, maxTokensEstimate);
            n = (int) llamaTokenize.invoke(model, textSeg, textLen, tokens, maxTokensEstimate, addBos, special);

            if (n < 0) {
                throw new RuntimeException("Tokenization failed with code: " + n);
            }
        }

        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = tokens.getAtIndex(ValueLayout.JAVA_INT, i);
        }

        log.debug("Tokenized {} characters into {} tokens", textLen, n);
        return result;
    }

    public String tokenToString(Arena arena, MemorySegment model, int token) throws Throwable {
        // Start with reasonable buffer size
        int bufferSize = 64;
        MemorySegment buffer = arena.allocate(bufferSize);

        int len = (int) llamaTokenToPiece.invoke(model, token, buffer, bufferSize);

        if (len == 0) {
            return "";
        }

        if (len < 0) {
            // Buffer too small, retry with larger buffer
            bufferSize = -len;
            buffer = arena.allocate(bufferSize);
            len = (int) llamaTokenToPiece.invoke(model, token, buffer, bufferSize);

            if (len <= 0) {
                return "";
            }
        }

        // Handle the case where len might be larger than bufferSize (shouldn't happen
        // but safe)
        int actualLen = Math.min(len, bufferSize);
        byte[] bytes = new byte[actualLen];
        MemorySegment.copy(buffer, ValueLayout.JAVA_BYTE, 0, bytes, 0, actualLen);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public int tokenBos(MemorySegment model) throws Throwable {
        return (int) llamaTokenBos.invoke(model);
    }

    public int tokenEos(MemorySegment model) throws Throwable {
        return (int) llamaTokenEos.invoke(model);
    }

    public int tokenNl(MemorySegment model) throws Throwable {
        return (int) llamaTokenNl.invoke(model);
    }

    public boolean addBosToken(MemorySegment model) throws Throwable {
        return (boolean) llamaAddBosToken.invoke(model);
    }

    public boolean addEosToken(MemorySegment model) throws Throwable {
        return (boolean) llamaAddEosToken.invoke(model);
    }

    // Inference with error checking
    public int decode(MemorySegment ctx, MemorySegment batch) throws Throwable {
        int result = (int) llamaDecode.invoke(ctx, batch);
        if (result != 0) {
            log.warn("Decode returned non-zero status: {}", result);
        }
        return result;
    }

    public MemorySegment getLogitsIth(MemorySegment ctx, int i) throws Throwable {
        MemorySegment logits = (MemorySegment) llamaGetLogitsIth.invoke(ctx, i);
        if (logits.address() == 0) {
            throw new RuntimeException("Failed to get logits for index: " + i);
        }
        return logits;
    }

    public MemorySegment getEmbeddingsIth(MemorySegment ctx, int i) throws Throwable {
        MemorySegment embeddings = (MemorySegment) llamaGetEmbeddingsIth.invoke(ctx, i);
        if (embeddings.address() == 0) {
            throw new RuntimeException("Failed to get embeddings for index: " + i);
        }
        return embeddings;
    }

    public MemorySegment getEmbeddingsSeq(MemorySegment ctx, int seqId) throws Throwable {
        MemorySegment embeddings = (MemorySegment) llamaGetEmbeddingsSeq.invoke(ctx, seqId);
        if (embeddings.address() == 0) {
            throw new RuntimeException("Failed to get embeddings for sequence: " + seqId);
        }
        return embeddings;
    }

    // KV cache
    private MethodHandle findFunctionOptional(String name, FunctionDescriptor descriptor) {
        try {
            return findFunction(name, descriptor);
        } catch (Exception e) {
            log.debug("Optional function {} not available: {}", name, e.getMessage());
            return null;
        }
    }

    // Update methods to handle null MethodHandles
    public void kvCacheClear(MemorySegment ctx) throws Throwable {
        if (llamaKvCacheClear != null) {
            llamaKvCacheClear.invoke(ctx, true);
        } else {
            throw new UnsupportedOperationException("llama_kv_cache_clear not available");
        }
    }

    public void kvCacheDefrag(MemorySegment ctx) throws Throwable {
        if (llamaKvCacheDefrag != null) {
            llamaKvCacheDefrag.invoke(ctx);
        }
        // Silently ignore if not available - it's optional
    }

    public void kvCacheUpdate(MemorySegment ctx) throws Throwable {
        if (llamaKvCacheUpdate != null) {
            llamaKvCacheUpdate.invoke(ctx);
        }
        // Silently ignore if not available - it's optional
    }

    public boolean kvCacheSeqRm(MemorySegment ctx, int seqId, int p0, int p1) throws Throwable {
        return (boolean) llamaKvCacheSeqRm.invoke(ctx, seqId, p0, p1);
    }

    public void kvCacheSeqCp(MemorySegment ctx, int seqIdSrc, int seqIdDst, int p0, int p1) throws Throwable {
        llamaKvCacheSeqCp.invoke(ctx, seqIdSrc, seqIdDst, p0, p1);
    }

    public void kvCacheSeqKeep(MemorySegment ctx, int seqId) throws Throwable {
        llamaKvCacheSeqKeep.invoke(ctx, seqId);
    }

    public void kvCacheSeqAdd(MemorySegment ctx, int seqId, int p0, int p1, int delta) throws Throwable {
        llamaKvCacheSeqAdd.invoke(ctx, seqId, p0, p1, delta);
    }

    public void kvCacheSeqDiv(MemorySegment ctx, int seqId, int p0, int p1, int divisor) throws Throwable {
        llamaKvCacheSeqDiv.invoke(ctx, seqId, p0, p1, divisor);
    }

    // State management with proper error handling
    public long stateGetSize(MemorySegment ctx) throws Throwable {
        return (long) llamaStateGetSize.invoke(ctx);
    }

    public boolean stateSaveFile(MemorySegment ctx, String path) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pathSeg = arena.allocateFrom(path, StandardCharsets.UTF_8);
            return (boolean) llamaStateSaveFile.invoke(ctx, pathSeg, false);
        }
    }

    public boolean stateLoadFile(MemorySegment ctx, String path) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment pathSeg = arena.allocateFrom(path, StandardCharsets.UTF_8);
            return (boolean) llamaStateLoadFile.invoke(ctx, pathSeg, false);
        }
    }

    // Performance
    public void perfContextPrint(MemorySegment ctx) throws Throwable {
        llamaPerfContextPrint.invoke(ctx);
    }

    public void perfContextReset(MemorySegment ctx) throws Throwable {
        llamaPerfContextReset.invoke(ctx);
    }

    // Batch management
    public MemorySegment batchInit(Arena arena, int n_tokens, int embd, int n_seq_max) throws Throwable {
        return (MemorySegment) llamaBatchInit.invoke(n_tokens, embd, n_seq_max);
    }

    public void batchFree(MemorySegment batch) throws Throwable {
        llamaBatchFree.invoke(batch);
    }

    // Utility method to check if library is properly loaded
    public boolean isLoaded() {
        try {
            // Try to call a simple function to verify the library is working
            llamaBackendInit.invoke();
            return true;
        } catch (Throwable e) {
            return false;
        }
    }

    // Utility method to debug available functions
    public void debugAvailableFunctions() {
        log.debug("=== Available Functions ===");
        for (String funcName : CORE_FUNCTIONS) {
            List<String> namesToTry = getPossibleFunctionNames(funcName);
            for (String name : namesToTry) {
                Optional<MemorySegment> addr = lookup.find(name);
                if (addr.isPresent()) {
                    log.debug("FOUND: {} -> {}", funcName, name);
                    break;
                }
            }
        }

        // Also check C++ mangled names
        log.debug("=== C++ Functions ===");
        for (Map.Entry<String, String> entry : CPP_MANGLED_NAMES.entrySet()) {
            Optional<MemorySegment> addr = lookup.find(entry.getValue());
            if (addr.isPresent()) {
                log.debug("FOUND C++: {} -> {}", entry.getKey(), entry.getValue());
            }
        }
    }
}