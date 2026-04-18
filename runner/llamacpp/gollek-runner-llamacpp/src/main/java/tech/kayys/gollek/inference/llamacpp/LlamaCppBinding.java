package tech.kayys.gollek.inference.llamacpp;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Public facade for the llama.cpp FFM binding.
 *
 * <p>Delegates to focused collaborators, each with a single responsibility:
 * <ul>
 *   <li>{@link LlamaHandles} — all {@link java.lang.invoke.MethodHandle} downcalls and linking</li>
 *   <li>{@link LlamaBatchOps} — batch allocation, population, and deallocation</li>
 *   <li>{@link LlamaSamplerOps} — sampler chain construction and token sampling</li>
 *   <li>{@link LlamaNativeLoader} — native library discovery and loading</li>
 *   <li>{@link LlamaStructLayouts} — shared FFM struct layout constants</li>
 * </ul>
 *
 * <h3>Lifecycle</h3>
 * <ol>
 *   <li>Call {@link #load(LlamaCppProviderConfig)} (or {@link #load()}) to obtain an instance.</li>
 *   <li>Use the instance for inference — thread-safe for concurrent reads.</li>
 *   <li>Call {@link #close()} to free the backend and shared {@link Arena}.</li>
 * </ol>
 *
 * @see LlamaCppProvider
 * @see LlamaCppSession
 */
public class LlamaCppBinding {

    private static final Logger log = Logger.getLogger(LlamaCppBinding.class);

    final LlamaHandles h;
    final LlamaBatchOps batch;
    final LlamaSamplerOps sampler;

    private final Arena arena;
    private final AtomicBoolean backendInitialized = new AtomicBoolean(false);

    private LlamaCppBinding(LlamaHandles handles) {
        this.h       = handles;
        this.arena   = Arena.ofShared();
        this.batch   = new LlamaBatchOps(handles);
        this.sampler = new LlamaSamplerOps(handles, arena);
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /** Loads with default settings (quiet mode, no explicit path). */
    public static LlamaCppBinding load() { return load(false); }

    /** Loads using provider configuration for verbosity and library path. */
    public static LlamaCppBinding load(LlamaCppProviderConfig config) {
        boolean verbose = config != null && config.verboseLogging();
        Optional<String> libPath = config == null ? Optional.empty() : config.nativeLibraryPath();
        Optional<String> libDir  = config == null ? Optional.empty() : config.nativeLibraryDir();
        return doLoad(verbose, libPath, libDir);
    }

    /** Loads with explicit verbosity flag. */
    public static LlamaCppBinding load(boolean verbose) {
        return doLoad(verbose, Optional.empty(), Optional.empty());
    }

    private static LlamaCppBinding doLoad(boolean verbose, Optional<String> libPath, Optional<String> libDir) {
        SymbolLookup lookup = LlamaNativeLoader.load(verbose, libPath, libDir);
        LlamaHandles handles = new LlamaHandles(lookup);
        handles.verbose = verbose;
        LlamaCppBinding binding = new LlamaCppBinding(handles);
        binding.backendInit();
        return binding;
    }

    // ── Backend lifecycle ─────────────────────────────────────────────────────

    /** Initialises the llama.cpp backend (ggml backends, Metal/CUDA device setup). Idempotent. */
    public void backendInit() {
        if (backendInitialized.compareAndSet(false, true)) {
            try {
                h.require(h.backendInit, "llama_backend_init");
                log.info("Initializing llama.cpp backend...");
                h.backendInit.invoke();
            } catch (Throwable e) {
                backendInitialized.set(false);
                throw new RuntimeException("Failed to initialize llama backend", e);
            }
        }
    }

    /** Frees the llama.cpp backend resources. Idempotent. */
    public void backendFree() {
        if (backendInitialized.compareAndSet(true, false)) {
            try {
                if (h.backendFree != null) { log.info("Freeing llama.cpp backend"); h.backendFree.invoke(); }
            } catch (Throwable e) { log.error("Failed to free llama backend", e); }
        }
    }

    /** Suppresses verbose native log output (Metal/CUDA pipeline messages). */
    public void suppressNativeLogs() {
        try {
            if (h.gollekLogDisable != null) { h.gollekLogDisable.invoke(); return; }
        } catch (Throwable ignored) {}
        LlamaNativeLoader.suppressNativeLogs(SymbolLookup.loaderLookup());
    }

    // ── Default params ────────────────────────────────────────────────────────

    /**
     * Returns a segment containing default {@code llama_model_params}.
     * Uses the Gollek shim if available, otherwise falls back to the standard
     * llama.cpp function or hard-coded conservative defaults.
     */
    public MemorySegment getDefaultModelParams() {
        try {
            MemorySegment params = arena.allocate(LlamaStructLayouts.MODEL_PARAMS);
            if (h.gollekModelDefaultParamsInto != null) {
                h.gollekModelDefaultParamsInto.invoke(params);
            } else if (h.modelDefaultParams != null) {
                MemorySegment def = (MemorySegment) h.modelDefaultParams.invoke((SegmentAllocator) arena);
                MemorySegment.copy(def, 0, params, 0, LlamaStructLayouts.MODEL_PARAMS.byteSize());
            } else {
                setModelParam(params, "n_gpu_layers", 0);
                setModelParam(params, "split_mode", 0);
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
        } catch (Throwable e) { throw new RuntimeException("Failed to get default model params", e); }
    }

    /**
     * Returns a segment containing default {@code llama_context_params}.
     * Uses the Gollek shim if available, otherwise falls back to the standard
     * llama.cpp function or hard-coded conservative defaults.
     */
    public MemorySegment getDefaultContextParams() {
        try {
            MemorySegment params = arena.allocate(LlamaStructLayouts.CONTEXT_PARAMS);
            if (h.gollekContextDefaultParamsInto != null) {
                h.gollekContextDefaultParamsInto.invoke(params);
            } else if (h.contextDefaultParams != null) {
                MemorySegment def = (MemorySegment) h.contextDefaultParams.invoke((SegmentAllocator) arena);
                MemorySegment.copy(def, 0, params, 0, LlamaStructLayouts.CONTEXT_PARAMS.byteSize());
            } else {
                int threads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
                setContextParam(params, "n_ctx", 4096);
                setContextParam(params, "n_batch", 512);
                setContextParam(params, "n_ubatch", 512);
                setContextParam(params, "n_seq_max", 1);
                setContextParam(params, "n_threads", threads);
                setContextParam(params, "n_threads_batch", threads);
                setContextParam(params, "rope_scaling_type", 0);
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
        } catch (Throwable e) { throw new RuntimeException("Failed to get default context params", e); }
    }

    // ── Model / context lifecycle ─────────────────────────────────────────────

    public MemorySegment loadModel(String path, MemorySegment modelParams) {
        try {
            h.require(h.loadModelFromFile, "llama_model_load_from_file");
            MemorySegment pathSeg = arena.allocateFrom(path);
            MemorySegment model = (MemorySegment) h.loadModelFromFile.invoke(pathSeg, modelParams);
            if (model.address() == 0) throw new RuntimeException("Failed to load model from: " + path);
            return model;
        } catch (Throwable e) { throw new RuntimeException("Failed to load model", e); }
    }

    public MemorySegment createContext(MemorySegment model, MemorySegment contextParams) {
        try {
            h.require(h.initFromModel, "llama_init_from_model");
            MemorySegment ctx = (MemorySegment) h.initFromModel.invoke(model, contextParams);
            if (ctx.address() == 0) throw new RuntimeException("Failed to create context");
            return ctx;
        } catch (Throwable e) { throw new RuntimeException("Failed to create context", e); }
    }

    public void freeModel(MemorySegment model) {
        try { h.freeModel.invoke(model); } catch (Throwable e) { log.error("Failed to free model", e); }
    }

    public void freeContext(MemorySegment context) {
        try { h.freeContext.invoke(context); } catch (Throwable e) { log.error("Failed to free context", e); }
    }

    // ── Vocab / metadata ─────────────────────────────────────────────────────

    public MemorySegment getVocab(MemorySegment model) {
        try {
            h.require(h.modelGetVocab, "llama_model_get_vocab");
            return (MemorySegment) h.modelGetVocab.invoke(model);
        } catch (Throwable e) { throw new RuntimeException("Failed to get vocab", e); }
    }

    public int getEosToken(MemorySegment model) {
        try { return (int) h.vocabEos.invoke(getVocab(model)); }
        catch (Throwable e) { throw new RuntimeException("Failed to get EOS token", e); }
    }

    public int getBosToken(MemorySegment model) {
        try { return (int) h.vocabBos.invoke(getVocab(model)); }
        catch (Throwable e) { throw new RuntimeException("Failed to get BOS token", e); }
    }

    public int getContextSize(MemorySegment context) {
        try { return (int) h.nCtx.invoke(context); }
        catch (Throwable e) { throw new RuntimeException("Failed to get context size", e); }
    }

    public int getVocabSize(MemorySegment model) {
        try { return (int) h.vocabNTokens.invoke(getVocab(model)); }
        catch (Throwable e) { throw new RuntimeException("Failed to get vocab size", e); }
    }

    public String getModelMetadata(MemorySegment model, String key) {
        try {
            h.require(h.modelMetaValStr, "llama_model_meta_val_str");
            MemorySegment keySeg = arena.allocateFrom(key);
            int size = (int) h.modelMetaValStr.invoke(model, keySeg, MemorySegment.NULL, 0L);
            if (size < 0) return null;
            MemorySegment buf = arena.allocate(size + 1);
            int result = (int) h.modelMetaValStr.invoke(model, keySeg, buf, (long) (size + 1));
            return result < 0 ? null : buf.getString(0L);
        } catch (Throwable e) {
            log.warnf("Failed to get metadata key %s: %s", key, e.getMessage());
            return null;
        }
    }

    public boolean isEndOfGeneration(MemorySegment model, int tokenId) {
        try {
            h.require(h.vocabIsEog, "llama_vocab_is_eog");
            return (int) h.vocabIsEog.invoke(getVocab(model), tokenId) != 0;
        } catch (Throwable e) { return tokenId == getEosToken(model); }
    }

    // ── Tokenization ─────────────────────────────────────────────────────────

    public int[] tokenize(MemorySegment model, String text, boolean addBos, boolean special) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment vocab = getVocab(model);
            MemorySegment textSeg = local.allocateFrom(text);
            int textLen = (int) textSeg.byteSize() - 1;
            int count = (int) h.tokenize.invoke(vocab, textSeg, textLen, MemorySegment.NULL, 0, addBos, special);
            int bufSize = count < 0 ? -count : count + 32;
            MemorySegment buf = local.allocate(ValueLayout.JAVA_INT, bufSize);
            int actual = (int) h.tokenize.invoke(vocab, textSeg, textLen, buf, bufSize, addBos, special);
            if (actual < 0) throw new RuntimeException("Tokenization failed: " + actual);
            int[] tokens = new int[actual];
            for (int i = 0; i < actual; i++) tokens[i] = buf.getAtIndex(ValueLayout.JAVA_INT, i);
            return tokens;
        } catch (Throwable e) { throw new RuntimeException("Failed to tokenize text", e); }
    }

    public String tokenToPiece(MemorySegment model, int token) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment vocab = getVocab(model);
            MemorySegment buf = local.allocate(ValueLayout.JAVA_BYTE, 256);
            int length;
            if (h.tokenToPiece != null) {
                length = (int) h.tokenToPiece.invoke(vocab, token, buf, 256, 0, false);
                if (length < 0 || length > 256) {
                    int needed = length > 256 ? length : Math.abs(length);
                    buf = local.allocate(ValueLayout.JAVA_BYTE, needed);
                    length = (int) h.tokenToPiece.invoke(vocab, token, buf, needed, 0, false);
                }
            } else {
                MemorySegment tokenSeg = local.allocate(ValueLayout.JAVA_INT, 1);
                tokenSeg.setAtIndex(ValueLayout.JAVA_INT, 0, token);
                length = (int) h.detokenize.invoke(vocab, tokenSeg, 1, buf, 256, false, true);
                if (length < 0 || length > 256) {
                    int needed = length > 256 ? length : Math.abs(length);
                    buf = local.allocate(ValueLayout.JAVA_BYTE, needed);
                    length = (int) h.detokenize.invoke(vocab, tokenSeg, 1, buf, needed, false, true);
                }
            }
            if (length <= 0) return "";
            byte[] bytes = new byte[length];
            MemorySegment.copy(buf, 0, MemorySegment.ofArray(bytes), 0, length);
            
            // Validate UTF-8 decoding and handle invalid sequences
            String decoded = decodeUtf8Safe(bytes);
            return decoded != null ? decoded : "";
        } catch (Throwable e) { 
            throw new RuntimeException("Failed to decode token: " + e.getMessage(), e); 
        }
    }

    /**
     * Safely decode UTF-8 bytes, replacing invalid sequences with replacement character.
     * This prevents garbled output from corrupted token streams.
     */
    private String decodeUtf8Safe(byte[] bytes) {
        try {
            String result = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
            
            // Additional validation: check for suspicious character patterns
            // that indicate decode corruption
            if (hasInvalidUtf8Patterns(result)) {
                return replaceInvalidCharacters(result);
            }
            return result;
        } catch (Exception e) {
            // Fallback: replace with question marks for invalid bytes
            return replaceInvalidCharacters(new String(bytes, java.nio.charset.StandardCharsets.ISO_8859_1));
        }
    }

    /**
     * Check if string has patterns indicating corrupt UTF-8 decode.
     */
    private boolean hasInvalidUtf8Patterns(String str) {
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            // Detect replacement character or unassigned codepoints
            if (c == '\uFFFD' || Character.getType(c) == Character.UNASSIGNED) {
                return true;
            }
            // Detect certain control character patterns that shouldn't appear
            if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') {
                return true;
            }
        }
        return false;
    }

    /**
     * Replace invalid or suspicious characters with safe alternatives.
     */
    private String replaceInvalidCharacters(String str) {
        StringBuilder safe = new StringBuilder();
        for (char c : str.toCharArray()) {
            if (Character.isDefined(c) && Character.getType(c) != Character.UNASSIGNED &&
                (c >= 0x20 || c == '\t' || c == '\n' || c == '\r')) {
                safe.append(c);
            } else if (c > 0x007F) {
                // Try to keep non-ASCII characters if they're valid
                if (!Character.isISOControl(c)) {
                    safe.append(c);
                }
            }
        }
        return safe.toString();
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    public int decode(MemorySegment context, MemorySegment batchSeg) {
        try { return (int) h.decode.invoke(context, batchSeg); }
        catch (Throwable e) { throw new RuntimeException("Failed to decode", e); }
    }

    public MemorySegment getLogits(MemorySegment context) {
        try { return (MemorySegment) h.getLogits.invoke(context); }
        catch (Throwable e) { throw new RuntimeException("Failed to get logits", e); }
    }

    public MemorySegment getLogitsIth(MemorySegment context, int index) {
        try { return (MemorySegment) h.getLogitsIth.invoke(context, index); }
        catch (Throwable e) { throw new RuntimeException("Failed to get logits at index", e); }
    }

    /** Clears the KV cache for the given context. */
    public void kvCacheClear(MemorySegment context) {
        try {
            h.require(h.getMemory, "llama_get_memory");
            h.require(h.memoryClear, "llama_memory_clear");
            MemorySegment memory = (MemorySegment) h.getMemory.invoke(context);
            h.memoryClear.invoke(memory, true);
        } catch (Throwable e) { throw new RuntimeException("Failed to clear KV cache", e); }
    }

    public boolean saveSession(MemorySegment context, Path sessionPath, int[] tokens, int count) {
        if (sessionPath == null || tokens == null || count <= 0) return false;
        try (Arena local = Arena.ofConfined()) {
            MemorySegment path = local.allocateFrom(sessionPath.toString());
            MemorySegment tokenSeg = local.allocate(ValueLayout.JAVA_INT, count);
            for (int i = 0; i < count; i++) tokenSeg.setAtIndex(ValueLayout.JAVA_INT, i, tokens[i]);
            return tech.kayys.gollek.llama.llama_h.llama_state_save_file(context, path, tokenSeg, count);
        } catch (Throwable e) { throw new RuntimeException("Failed to save session: " + sessionPath, e); }
    }

    public int[] loadSession(MemorySegment context, Path sessionPath, int maxTokens) {
        if (sessionPath == null || maxTokens <= 0) return new int[0];
        try (Arena local = Arena.ofConfined()) {
            MemorySegment path = local.allocateFrom(sessionPath.toString());
            MemorySegment tokenSeg = local.allocate(ValueLayout.JAVA_INT, maxTokens);
            MemorySegment countOut = local.allocate(ValueLayout.JAVA_LONG, 1);
            boolean ok = tech.kayys.gollek.llama.llama_h.llama_state_load_file(
                    context, path, tokenSeg, maxTokens, countOut);
            if (!ok) return new int[0];
            int size = (int) Math.min(countOut.get(ValueLayout.JAVA_LONG, 0), maxTokens);
            int[] result = new int[size];
            for (int i = 0; i < size; i++) result[i] = tokenSeg.getAtIndex(ValueLayout.JAVA_INT, i);
            return result;
        } catch (Throwable e) { throw new RuntimeException("Failed to load session: " + sessionPath, e); }
    }

    // ── Batch (delegated to LlamaBatchOps) ───────────────────────────────────

    public MemorySegment batchInit(int nTokens, int embd, int nSeqMax) { return batch.init(nTokens, embd, nSeqMax); }
    public void batchFree(MemorySegment b) { batch.free(b); }
    public void setBatchNTokens(MemorySegment b, int n) { batch.setNTokens(b, n); }
    public int getBatchNTokensCount(MemorySegment b) { return batch.getNTokens(b); }
    public void setBatchSize(MemorySegment b, int n) { batch.setNTokens(b, n); }
    public void setBatchToken(MemorySegment b, int i, int token, int pos, int seqId, boolean logits) { batch.setToken(b, i, token, pos, seqId, logits); }
    public void setBatchToken(MemorySegment b, int i, int token) { batch.setTokenId(b, i, token); }
    public void setBatchPos(MemorySegment b, int i, int pos) { batch.setPos(b, i, pos); }
    public void setBatchSeqId(MemorySegment b, int i, int seqId) { batch.setSeqId(b, i, seqId); }
    public void setBatchLogits(MemorySegment b, int i, boolean enable) { batch.setLogits(b, i, enable); }
    public void setBatchMultimodalEmbd(MemorySegment b, int i, MemorySegment embd, int pos, int seqId, boolean logits) { batch.setMultimodalEmbd(b, i, embd, pos, seqId, logits); }

    // ── Samplers (delegated to LlamaSamplerOps) ───────────────────────────────

    public MemorySegment createSamplerChain() { return sampler.createChain(); }
    public void addGreedySampler(MemorySegment chain) { sampler.addGreedy(chain); }
    public void addTopKSampler(MemorySegment chain, int k) { sampler.addTopK(chain, k); }
    public void addTopPSampler(MemorySegment chain, float p, long minKeep) { sampler.addTopP(chain, p, minKeep); }
    public void addMinPSampler(MemorySegment chain, float p, long minKeep) { sampler.addMinP(chain, p, minKeep); }
    public void addTempSampler(MemorySegment chain, float temp) { sampler.addTemp(chain, temp); }
    public void addDistSampler(MemorySegment chain, int seed) { sampler.addDist(chain, seed); }
    public void addPenaltiesSampler(MemorySegment chain, int lastN, float repeat, float freq, float presence) { sampler.addPenalties(chain, lastN, repeat, freq, presence); }
    public void addMirostatSampler(MemorySegment chain, int nVocab, int seed, float tau, float eta, int m) { sampler.addMirostat(chain, nVocab, seed, tau, eta, m); }
    public void addMirostatV2Sampler(MemorySegment chain, int seed, float tau, float eta) { sampler.addMirostatV2(chain, seed, tau, eta); }
    public void addGrammarSampler(MemorySegment chain, MemorySegment model, String grammarStr, String grammarRoot) { sampler.addGrammar(chain, getVocab(model), grammarStr, grammarRoot); }
    public void addTypicalSampler(MemorySegment chain, float p, long minKeep) { sampler.addTypical(chain, p, minKeep); }
    public int sample(MemorySegment chain, MemorySegment context, int index) { return sampler.sample(chain, context, index); }
    public void freeSampler(MemorySegment chain) { sampler.freeChain(chain); }

    // ── Embeddings ────────────────────────────────────────────────────────────

    public int nEmbd(MemorySegment model) throws Throwable { return (int) h.nEmbd.invoke(model); }
    public MemorySegment getEmbeddings(MemorySegment ctx) throws Throwable { return (MemorySegment) h.getEmbeddings.invoke(ctx); }
    public MemorySegment getEmbeddingsIth(MemorySegment ctx, int i) throws Throwable { return (MemorySegment) h.getEmbeddingsIth.invoke(ctx, i); }

    // ── LoRA adapters ─────────────────────────────────────────────────────────

    public MemorySegment loadLoraAdapter(MemorySegment model, String adapterPath) {
        try {
            h.require(h.adapterLoraInit, "llama_adapter_lora_init");
            MemorySegment path = arena.allocateFrom(adapterPath);
            MemorySegment adapter = (MemorySegment) h.adapterLoraInit.invoke(model, path);
            if (adapter == null || adapter.address() == 0)
                throw new RuntimeException("llama_adapter_lora_init returned null");
            return adapter;
        } catch (Throwable e) { throw new RuntimeException("Failed to load LoRA adapter: " + adapterPath, e); }
    }

    public void setLoraAdapter(MemorySegment context, MemorySegment adapter, float scale) {
        try {
            h.require(h.setAdapterLora, "llama_set_adapter_lora");
            int rc = (int) h.setAdapterLora.invoke(context, adapter, scale);
            if (rc != 0) throw new RuntimeException("llama_set_adapter_lora failed: " + rc);
        } catch (Throwable e) { throw new RuntimeException("Failed to set LoRA adapter", e); }
    }

    public void removeLoraAdapter(MemorySegment context, MemorySegment adapter) {
        try {
            h.require(h.rmAdapterLora, "llama_rm_adapter_lora");
            int rc = (int) h.rmAdapterLora.invoke(context, adapter);
            if (rc != 0) throw new RuntimeException("llama_rm_adapter_lora failed: " + rc);
        } catch (Throwable e) { throw new RuntimeException("Failed to remove LoRA adapter", e); }
    }

    public void clearLoraAdapters(MemorySegment context) {
        try {
            h.require(h.clearAdapterLora, "llama_clear_adapter_lora");
            h.clearAdapterLora.invoke(context);
        } catch (Throwable e) { throw new RuntimeException("Failed to clear LoRA adapters", e); }
    }

    public void freeLoraAdapter(MemorySegment adapter) {
        try {
            h.require(h.adapterLoraFree, "llama_adapter_lora_free");
            h.adapterLoraFree.invoke(adapter);
        } catch (Throwable e) { throw new RuntimeException("Failed to free LoRA adapter", e); }
    }

    // ── Struct param helpers ──────────────────────────────────────────────────

    public void setModelParam(MemorySegment params, String name, Object value) {
        setParam(LlamaStructLayouts.MODEL_PARAMS, params, name, value);
    }

    public void setContextParam(MemorySegment params, String name, Object value) {
        setParam(LlamaStructLayouts.CONTEXT_PARAMS, params, name, value);
    }

    private static void setParam(MemoryLayout layout, MemorySegment seg, String name, Object value) {
        long offset = layout.byteOffset(MemoryLayout.PathElement.groupElement(name));
        if (value instanceof Integer i)            seg.set(ValueLayout.JAVA_INT,   offset, i);
        else if (value instanceof Long l)          seg.set(ValueLayout.JAVA_LONG,  offset, l);
        else if (value instanceof Float f)         seg.set(ValueLayout.JAVA_FLOAT, offset, f);
        else if (value instanceof Boolean b)       seg.set(ValueLayout.JAVA_BYTE,  offset, (byte)(b ? 1 : 0));
        else if (value instanceof MemorySegment ms) seg.set(ValueLayout.ADDRESS,   offset, ms);
        else throw new IllegalArgumentException("Unsupported param type: " + value.getClass());
    }

    // ── Misc ──────────────────────────────────────────────────────────────────

    /** Returns the shared {@link Arena} used for long-lived native allocations. */
    public Arena getArena() { return arena; }

    /** Closes the backend and releases the shared arena. */
    public void close() {
        backendFree();
        arena.close();
    }
}
