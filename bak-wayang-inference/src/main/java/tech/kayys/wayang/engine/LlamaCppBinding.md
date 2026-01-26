package tech.kayys.golek.engine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

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
    private final MethodHandle llamaKvCacheClear;
    private final MethodHandle llamaKvCacheSeqRm;
    private final MethodHandle llamaKvCacheSeqCp;
    private final MethodHandle llamaKvCacheSeqKeep;
    private final MethodHandle llamaKvCacheSeqAdd;
    private final MethodHandle llamaKvCacheSeqDiv;
    private final MethodHandle llamaKvCacheDefrag;
    private final MethodHandle llamaKvCacheUpdate;

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

    public LlamaCppBinding(String libraryPath) {
        log.info("Loading llama.cpp library: {}", libraryPath);

        if (!Files.exists(Path.of(libraryPath))) {
            throw new RuntimeException("Library not found: " + libraryPath);
        }

        System.load(libraryPath);
        this.linker = Linker.nativeLinker();
        this.lookup = SymbolLookup.loaderLookup();

        try {
            // Core
            this.llamaBackendInit = findFunction("llama_backend_init",
                    FunctionDescriptor.ofVoid());
            this.llamaBackendFree = findFunction("llama_backend_free",
                    FunctionDescriptor.ofVoid());
            this.llamaModelDefaultParams = findFunction("llama_model_default_params",
                    FunctionDescriptor.of(LlamaStructs.MODEL_PARAMS_LAYOUT));
            this.llamaLoadModelFromFile = findFunction("llama_load_model_from_file",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LlamaStructs.MODEL_PARAMS_LAYOUT));
            this.llamaFreeModel = findFunction("llama_free_model",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            this.llamaContextDefaultParams = findFunction("llama_context_default_params",
                    FunctionDescriptor.of(LlamaStructs.CONTEXT_PARAMS_LAYOUT));
            this.llamaNewContextWithModel = findFunction("llama_new_context_with_model",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            LlamaStructs.CONTEXT_PARAMS_LAYOUT));
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
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaModelSize = findFunction("llama_model_size",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            this.llamaModelNParams = findFunction("llama_model_n_params",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

            // Tokenization - FIXED: Use correct function names and descriptors
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
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, LlamaStructs.BATCH_LAYOUT));
            this.llamaGetLogitsIth = findFunction("llama_get_logits_ith",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaGetEmbeddingsIth = findFunction("llama_get_embeddings_ith",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaGetEmbeddingsSeq = findFunction("llama_get_embeddings_seq",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // KV cache
            this.llamaKvCacheClear = findFunction("llama_kv_cache_clear",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            this.llamaKvCacheSeqRm = findFunction("llama_kv_cache_seq_rm",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.llamaKvCacheSeqCp = findFunction("llama_kv_cache_seq_cp",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.llamaKvCacheSeqKeep = findFunction("llama_kv_cache_seq_keep",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.llamaKvCacheSeqAdd = findFunction("llama_kv_cache_seq_add",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.llamaKvCacheSeqDiv = findFunction("llama_kv_cache_seq_div",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.llamaKvCacheDefrag = findFunction("llama_kv_cache_defrag",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            this.llamaKvCacheUpdate = findFunction("llama_kv_cache_update",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // State
            this.llamaStateGetSize = findFunction("llama_state_get_size",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            this.llamaStateSaveFile = findFunction("llama_state_save_file",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));
            this.llamaStateLoadFile = findFunction("llama_state_load_file",
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

            // Performance
            this.llamaPerfContextPrint = findFunction("llama_perf_context_print",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            this.llamaPerfContextReset = findFunction("llama_perf_context_reset",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            // Batch management
            this.llamaBatchInit = findFunction("llama_batch_init",
                    FunctionDescriptor.of(LlamaStructs.BATCH_LAYOUT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));
            this.llamaBatchFree = findFunction("llama_batch_free",
                    FunctionDescriptor.ofVoid(LlamaStructs.BATCH_LAYOUT));

            log.info("Llama.cpp bindings initialized successfully with {} functions",
                    countInitializedMethods());

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize llama.cpp bindings", e);
        }
    }

    private MethodHandle findFunction(String name, FunctionDescriptor descriptor) {
        return lookup.find(name)
                .map(addr -> linker.downcallHandle(addr, descriptor))
                .orElseThrow(() -> new RuntimeException("Function not found: " + name));
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

        MemorySegment params = arena.allocate(LlamaStructs.MODEL_PARAMS_LAYOUT);
        MemorySegment defaultParams = (MemorySegment) llamaModelDefaultParams.invoke();

        // Copy default params and then modify
        MemorySegment.copy(defaultParams, 0, params, 0, LlamaStructs.MODEL_PARAMS_LAYOUT.byteSize());

        // Set model parameters - using correct offsets based on llama.h struct
        params.set(ValueLayout.JAVA_INT, LlamaStructs.MODEL_PARAMS_GPU_LAYERS_OFFSET, gpuLayers);
        params.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.MODEL_PARAMS_USE_MMAP_OFFSET, useMmap);
        params.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.MODEL_PARAMS_USE_MLOCK_OFFSET, useMlock);

        MemorySegment pathSeg = arena.allocateFrom(path, StandardCharsets.UTF_8);
        MemorySegment model = (MemorySegment) llamaLoadModelFromFile.invoke(pathSeg, params);

        if (model.address() == 0) {
            throw new RuntimeException("Failed to load model: " + path);
        }

        log.info("Model loaded successfully: {}", path);
        return model;
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

    public MemorySegment createContext(Arena arena, MemorySegment model, int ctxSize, int batchSize,
            int threads, int seed, float ropeBase, float ropeScale,
            boolean embeddings, boolean flashAttn) throws Throwable {
        log.debug("Creating context: ctx={}, batch={}, threads={}, embeddings={}",
                ctxSize, batchSize, threads, embeddings);

        MemorySegment params = arena.allocate(LlamaStructs.CONTEXT_PARAMS_LAYOUT);
        MemorySegment defaultParams = (MemorySegment) llamaContextDefaultParams.invoke();
        MemorySegment.copy(defaultParams, 0, params, 0, LlamaStructs.CONTEXT_PARAMS_LAYOUT.byteSize());

        // Set context parameters using correct offsets
        params.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_SEED_OFFSET, seed);
        params.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_CTX_OFFSET, ctxSize);
        params.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_BATCH_OFFSET, batchSize);
        params.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_THREADS_OFFSET, threads);
        params.set(ValueLayout.JAVA_INT, LlamaStructs.CONTEXT_PARAMS_N_THREADS_BATCH_OFFSET, threads);
        params.set(ValueLayout.JAVA_FLOAT, LlamaStructs.CONTEXT_PARAMS_ROPE_FREQ_BASE_OFFSET, ropeBase);
        params.set(ValueLayout.JAVA_FLOAT, LlamaStructs.CONTEXT_PARAMS_ROPE_FREQ_SCALE_OFFSET, ropeScale);
        params.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.CONTEXT_PARAMS_EMBEDDINGS_OFFSET, embeddings);
        params.set(ValueLayout.JAVA_BOOLEAN, LlamaStructs.CONTEXT_PARAMS_FLASH_ATTN_OFFSET, flashAttn);

        MemorySegment context = (MemorySegment) llamaNewContextWithModel.invoke(model, params);

        if (context.address() == 0) {
            throw new RuntimeException("Failed to create context with size: " + ctxSize);
        }

        log.info("Context created successfully");
        return context;
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
    public void kvCacheClear(MemorySegment ctx) throws Throwable {
        llamaKvCacheClear.invoke(ctx);
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

    public void kvCacheDefrag(MemorySegment ctx) throws Throwable {
        llamaKvCacheDefrag.invoke(ctx);
    }

    public void kvCacheUpdate(MemorySegment ctx) throws Throwable {
        llamaKvCacheUpdate.invoke(ctx);
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
}