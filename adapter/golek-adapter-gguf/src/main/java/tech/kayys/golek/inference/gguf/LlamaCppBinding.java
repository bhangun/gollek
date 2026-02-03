
package tech.kayys.golek.inference.gguf;

import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
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

    private final SymbolLookup symbolLookup;
    private final Arena arena;

    // Function handles
    private final MethodHandle llama_backend_init;
    private final MethodHandle llama_backend_free;
    private final MethodHandle llama_model_default_params;
    private final MethodHandle llama_context_default_params;
    private final MethodHandle llama_load_model_from_file;
    private final MethodHandle llama_new_context_with_model;
    private final MethodHandle llama_free_model;
    private final MethodHandle llama_free;
    private final MethodHandle llama_tokenize;
    private final MethodHandle llama_token_to_piece;
    private final MethodHandle llama_decode;
    private final MethodHandle llama_get_logits;
    private final MethodHandle llama_sample_token_greedy;
    private final MethodHandle llama_token_eos;
    private final MethodHandle llama_token_bos;
    private final MethodHandle llama_batch_init;
    private final MethodHandle llama_batch_free;
    private final MethodHandle llama_n_ctx;
    private final MethodHandle llama_n_vocab;

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
            ValueLayout.JAVA_BOOLEAN.withName("vocab_only"),
            ValueLayout.JAVA_BOOLEAN.withName("use_mmap"),
            ValueLayout.JAVA_BOOLEAN.withName("use_direct_io"),
            ValueLayout.JAVA_BOOLEAN.withName("use_mlock"),
            ValueLayout.JAVA_BOOLEAN.withName("check_tensors"),
            ValueLayout.JAVA_BOOLEAN.withName("use_extra_bufts"),
            ValueLayout.JAVA_BOOLEAN.withName("no_host"),
            ValueLayout.JAVA_BOOLEAN.withName("no_alloc")).withName("llama_model_params");

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
            ValueLayout.JAVA_BOOLEAN.withName("embeddings"),
            ValueLayout.JAVA_BOOLEAN.withName("offload_kqv"),
            ValueLayout.JAVA_BOOLEAN.withName("no_perf"),
            ValueLayout.JAVA_BOOLEAN.withName("op_offload"),
            ValueLayout.JAVA_BOOLEAN.withName("swa_full"),
            ValueLayout.JAVA_BOOLEAN.withName("kv_unified"),
            MemoryLayout.paddingLayout(2), // Align to 8 bytes for subsequent long/address
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

    private LlamaCppBinding(SymbolLookup symbolLookup) {
        this.symbolLookup = symbolLookup;
        this.arena = Arena.ofShared();

        // Link function handles
        Linker linker = Linker.nativeLinker();

        this.llama_backend_init = linkFunction(linker, "llama_backend_init",
                FunctionDescriptor.ofVoid());

        this.llama_backend_free = linkFunction(linker, "llama_backend_free",
                FunctionDescriptor.ofVoid());

        this.llama_model_default_params = linkFunction(linker, "llama_model_default_params",
                FunctionDescriptor.of(LLAMA_MODEL_PARAMS_LAYOUT));

        this.llama_context_default_params = linkFunction(linker, "llama_context_default_params",
                FunctionDescriptor.of(LLAMA_CONTEXT_PARAMS_LAYOUT));

        this.llama_load_model_from_file = linkFunction(linker, "llama_model_load_from_file",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LLAMA_MODEL_PARAMS_LAYOUT));
        this.llama_new_context_with_model = linkFunction(linker, "llama_init_from_model",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, LLAMA_CONTEXT_PARAMS_LAYOUT));
        this.llama_free_model = linkFunction(linker, "llama_model_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        this.llama_free = linkFunction(linker, "llama_free",
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        this.llama_tokenize = linkFunction(linker, "llama_tokenize",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN,
                        ValueLayout.JAVA_BOOLEAN));

        this.llama_token_to_piece = linkFunction(linker, "llama_token_to_piece",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        this.llama_decode = linkFunction(linker, "llama_decode",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, LLAMA_BATCH_LAYOUT));
        this.llama_get_logits = linkFunction(linker, "llama_get_logits",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        this.llama_sample_token_greedy = linkFunction(linker, "llama_sample_token_greedy",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        this.llama_token_eos = linkFunction(linker, "llama_token_eos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        this.llama_token_bos = linkFunction(linker, "llama_token_bos",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        this.llama_batch_init = linkFunction(linker, "llama_batch_init",
                FunctionDescriptor.of(LLAMA_BATCH_LAYOUT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT));

        this.llama_batch_free = linkFunction(linker, "llama_batch_free",
                FunctionDescriptor.ofVoid(LLAMA_BATCH_LAYOUT));

        this.llama_n_ctx = linkFunction(linker, "llama_n_ctx",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        this.llama_n_vocab = linkFunction(linker, "llama_n_vocab",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    }

    private MethodHandle linkFunction(Linker linker, String name, FunctionDescriptor descriptor) {
        return symbolLookup.find(name)
                .map(addr -> linker.downcallHandle(addr, descriptor))
                .orElseGet(() -> {
                    log.warnf("Native function not found: %s. Some features may be unavailable.", name);
                    return null;
                });
    }

    /**
     * Load the native library and create binding instance
     */
    public static LlamaCppBinding load() {
        try {
            // Extract and load native library
            Path nativeLib = extractNativeLibrary();
            System.load(nativeLib.toAbsolutePath().toString());

            // Create symbol lookup
            SymbolLookup symbolLookup = SymbolLookup.loaderLookup();

            LlamaCppBinding binding = new LlamaCppBinding(symbolLookup);

            // Initialize backend
            binding.backendInit();

            log.infof("Loaded llama.cpp native library: %s", nativeLib);
            return binding;

        } catch (Throwable e) {
            throw new RuntimeException("Failed to load llama.cpp library", e);
        }
    }

    /**
     * Extract native library from resources to temp directory
     */
    private static Path extractNativeLibrary() throws Exception {
        String osName = System.getProperty("os.name").toLowerCase();

        String libName;
        if (osName.contains("win")) {
            libName = "llama.dll";
        } else if (osName.contains("mac")) {
            libName = "libllama.dylib";
        } else {
            libName = "libllama.so";
        }

        // Check for CUDA support
        String cudaPath = System.getenv("CUDA_PATH");
        boolean hasCuda = cudaPath != null && !cudaPath.isEmpty();

        String resourcePath = hasCuda ? "/native-libs/cuda/" + libName : "/native-libs/cpu/" + libName;

        // Extract to temp directory
        Path tempDir = Files.createTempDirectory("llama-cpp");
        Path libPath = tempDir.resolve(libName);

        // Try multiple classloaders and paths
        InputStream is = getResourceAsStream(resourcePath);
        if (is == null) {
            String fallbackPath = "/native-libs/" + libName;
            is = getResourceAsStream(fallbackPath);
            if (is == null) {
                // Diagnostic logging of classpath if failed
                log.errorf("Native library not found in %s or %s", resourcePath, fallbackPath);
                throw new RuntimeException("Native library not found in " + resourcePath + " or " + fallbackPath);
            }
        }

        try (InputStream stream = is) {
            Files.copy(stream, libPath, StandardCopyOption.REPLACE_EXISTING);
        }

        // Also extract libggml if it exists (for newer llama.cpp versions)
        String ggmlName = libName.replace("llama", "ggml");
        String ggmlResource = "/native-libs/" + ggmlName;
        try (InputStream ggmlStream = getResourceAsStream(ggmlResource)) {
            if (ggmlStream != null) {
                Path ggmlPath = tempDir.resolve(ggmlName);
                Files.copy(ggmlStream, ggmlPath, StandardCopyOption.REPLACE_EXISTING);
                if (!osName.contains("win")) {
                    ggmlPath.toFile().setExecutable(true);
                }
            }
        }

        // Make executable on Unix
        if (!osName.contains("win")) {
            libPath.toFile().setExecutable(true);
        }

        return libPath;
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
        try {
            checkHandle(llama_backend_init, "llama_backend_init");
            llama_backend_init.invoke();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize llama backend", e);
        }
    }

    public void backendFree() {
        try {
            if (llama_backend_free != null) {
                llama_backend_free.invoke();
            }
        } catch (Throwable e) {
            log.error("Failed to free llama backend", e);
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
            checkHandle(llama_model_default_params, "llama_model_default_params");
            MemorySegment params = arena.allocate(LLAMA_MODEL_PARAMS_LAYOUT);
            llama_model_default_params.invoke(params);
            return params;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get default model params", e);
        }
    }

    public MemorySegment getDefaultContextParams() {
        try {
            checkHandle(llama_context_default_params, "llama_context_default_params");
            MemorySegment params = arena.allocate(LLAMA_CONTEXT_PARAMS_LAYOUT);
            llama_context_default_params.invoke(params);
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
        try {
            MemorySegment textSegment = arena.allocateFrom(text);

            // First call to get token count
            int tokenCount = (int) llama_tokenize.invoke(
                    model, textSegment, text.length(),
                    MemorySegment.NULL, 0, addBos, special);

            // Allocate buffer and tokenize
            MemorySegment tokensBuffer = arena.allocate(ValueLayout.JAVA_INT, tokenCount);

            int actualCount = (int) llama_tokenize.invoke(
                    model, textSegment, text.length(),
                    tokensBuffer, tokenCount, addBos, special);

            // Copy to Java array
            int[] tokens = new int[actualCount];
            for (int i = 0; i < actualCount; i++) {
                tokens[i] = tokensBuffer.getAtIndex(ValueLayout.JAVA_INT, i);
            }
            return tokens;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to tokenize text", e);
        }
    }

    public String tokenToPiece(MemorySegment model, int token) {
        try {
            MemorySegment buffer = arena.allocate(ValueLayout.JAVA_BYTE, 256);
            int length = (int) llama_token_to_piece.invoke(model, token, buffer, 256);
            if (length < 0) {
                return "";
            }

            byte[] bytes = new byte[length];
            for (int i = 0; i < length; i++) {
                bytes[i] = buffer.getAtIndex(ValueLayout.JAVA_BYTE, i);
            }
            return new String(bytes);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to convert token to piece", e);
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

    public int sampleTokenGreedy(MemorySegment context, MemorySegment logits) {
        try {
            return (int) llama_sample_token_greedy.invoke(context, logits);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to sample token", e);
        }
    }

    public int getEosToken(MemorySegment model) {
        try {
            return (int) llama_token_eos.invoke(model);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get EOS token", e);
        }
    }

    public int getBosToken(MemorySegment model) {
        try {
            return (int) llama_token_bos.invoke(model);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get BOS token", e);
        }
    }

    public MemorySegment batchInit(int nTokens, int embd, int nSeqMax) {
        try {
            MemorySegment batch = arena.allocate(LLAMA_BATCH_LAYOUT);
            llama_batch_init.invoke(batch, nTokens, embd, nSeqMax);
            return batch;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize batch", e);
        }
    }

    public void batchFree(MemorySegment batch) {
        try {
            llama_batch_free.invoke(batch);
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
            return (int) llama_n_vocab.invoke(model);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get vocab size", e);
        }
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
            segment.set(ValueLayout.JAVA_BOOLEAN, offset, b);
        } else if (value instanceof MemorySegment ms) {
            segment.set(ValueLayout.ADDRESS, offset, ms);
        } else {
            throw new IllegalArgumentException("Unsupported type: " + value.getClass());
        }
    }

    public void setBatchToken(MemorySegment batch, int index, int token) {
        MemorySegment tokenPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("token")));
        tokenPtr.setAtIndex(ValueLayout.JAVA_INT, index, token);
    }

    public void setBatchPos(MemorySegment batch, int index, int pos) {
        MemorySegment posPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("pos")));
        posPtr.setAtIndex(ValueLayout.JAVA_INT, index, pos);
    }

    public void setBatchSeqId(MemorySegment batch, int index, int seqId) {
        MemorySegment nSeqIdPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("n_seq_id")));
        nSeqIdPtr.setAtIndex(ValueLayout.JAVA_INT, index, 1);

        MemorySegment seqIdPtrPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("seq_id")));
        MemorySegment seqIdPtr = seqIdPtrPtr.getAtIndex(ValueLayout.ADDRESS, index);
        seqIdPtr.setAtIndex(ValueLayout.JAVA_INT, 0, seqId);
    }

    public void setBatchLogits(MemorySegment batch, int index, boolean enable) {
        MemorySegment logitsPtr = batch.get(ValueLayout.ADDRESS,
                LLAMA_BATCH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("logits")));
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

    public void close() {
        backendFree();
        arena.close();
    }
}
