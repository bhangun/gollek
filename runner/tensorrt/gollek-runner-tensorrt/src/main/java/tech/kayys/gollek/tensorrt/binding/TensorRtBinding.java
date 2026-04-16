package tech.kayys.gollek.tensorrt.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM binding to the TensorRT 10 C API ({@code nvinfer_c.h}).
 *
 * <p>Loads {@code libnvinfer_10.so} (Linux) at runtime and binds all flat
 * C-linkage functions from the {@code nvinfer_c.h} header. Mirrors the
 * singleton pattern of {@link tech.kayys.gollek.kernel.fa3.FlashAttention3Binding}.
 *
 * <h2>What TensorRT does</h2>
 * <p>TensorRT compiles an ONNX or custom model into a platform-specific
 * {@code .engine} file optimised for a target GPU. At inference time the
 * engine is deserialised and executed with zero-copy CUDA device buffers.
 * Compared to raw ONNX Runtime CUDA EP, TensorRT typically delivers:
 * <ul>
 *   <li>2–4× lower latency for INT8 batch=1 (fused kernels, kernel auto-tuning)</li>
 *   <li>Up to 8× throughput for large batch FP16 (tensor parallelism fused ops)</li>
 * </ul>
 *
 * <h2>TRT 10 C API</h2>
 * <p>TensorRT 10 introduced a proper flat C API in {@code nvinfer_c.h} with
 * symbols prefixed {@code nvinfer}. Prior versions required C++ vtable dispatch.
 * All bound symbols below have {@code extern "C"} linkage in the TRT 10 headers.
 *
 * <h2>C functions bound (from {@code nvinfer_c.h}, TRT 10.x)</h2>
 * <pre>
 * void*    nvinferCreateRuntime(NvInferILogger*)
 * void     nvinferDestroyRuntime(void*)
 * void*    nvinferRuntimeDeserializeCudaEngine(void* runtime, void* blob, size_t)
 * void     nvinferDestroyEngine(void*)
 * void*    nvinferEngineCreateExecutionContext(void*)
 * void     nvinferDestroyExecutionContext(void*)
 * int32_t  nvinferEngineGetNbIOTensors(void*)
 * char*    nvinferEngineGetIOTensorName(void*, int32_t)
 * int32_t  nvinferEngineGetTensorIOMode(void*, char*)
 * int64_t  nvinferEngineGetTensorBytesPerComponent(void*, char*, int32_t)
 * bool     nvinferExecutionContextSetTensorAddress(void*, char*, void*)
 * void*    nvinferExecutionContextGetTensorAddress(void*, char*)
 * bool     nvinferExecutionContextEnqueueV3(void*, cudaStream_t)
 * int32_t  nvinferGetNbTensorDimensions(void*, char*)
 * bool     nvinferGetTensorDimensions(void*, char*, int64_t*, int32_t)
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * <pre>{@code
 * TensorRtBinding.initialize(Path.of("/usr/lib/x86_64-linux-gnu/libnvinfer_10.so"));
 * TensorRtBinding trt = TensorRtBinding.getInstance();
 *
 * // Load a compiled .engine file:
 * byte[] engineBlob = Files.readAllBytes(Path.of("model.engine"));
 * MemorySegment runtime = trt.createRuntime();
 * MemorySegment engine  = trt.deserializeEngine(runtime, engineBlob);
 * MemorySegment ctx     = trt.createExecutionContext(engine);
 *
 * // Bind IO tensors (CUDA device pointers):
 * trt.setTensorAddress(ctx, "input_ids",    deviceInputPtr);
 * trt.setTensorAddress(ctx, "logits",       deviceOutputPtr);
 *
 * // Execute on a CUDA stream:
 * trt.enqueueV3(ctx, cudaStream);
 *
 * // Cleanup:
 * trt.destroyExecutionContext(ctx);
 * trt.destroyEngine(engine);
 * trt.destroyRuntime(runtime);
 * }</pre>
 *
 * <h2>Build / install TensorRT</h2>
 * <pre>
 * # Via pip (includes C library):
 * pip install tensorrt==10.x.x
 * find / -name "libnvinfer_10.so" 2>/dev/null
 *
 * # Or download the tar.gz from:
 * # https://developer.nvidia.com/tensorrt/download
 * # and extract to /usr/local/tensorrt/
 * </pre>
 *
 * <h2>Compile a TRT engine from ONNX</h2>
 * <pre>
 * # Using trtexec (ships with TensorRT):
 * trtexec --onnx=model.onnx \
 *         --saveEngine=model.engine \
 *         --fp16 \
 *         --minShapes=input_ids:1x1 \
 *         --optShapes=input_ids:1x512 \
 *         --maxShapes=input_ids:1x2048
 *
 * # INT8 with calibration:
 * trtexec --onnx=model.onnx \
 *         --saveEngine=model_int8.engine \
 *         --int8 --calib=calib_data.json
 * </pre>
 */
public class TensorRtBinding {

    private static final Logger LOG = Logger.getLogger(TensorRtBinding.class);
    private static volatile TensorRtBinding instance;

    // ── C function names from nvinfer_c.h (TRT 10) ───────────────────────────
    private static final String FN_CREATE_RUNTIME          = "nvinferCreateRuntime";
    private static final String FN_DESTROY_RUNTIME         = "nvinferDestroyRuntime";
    private static final String FN_DESERIALIZE_ENGINE      = "nvinferRuntimeDeserializeCudaEngine";
    private static final String FN_DESTROY_ENGINE          = "nvinferDestroyEngine";
    private static final String FN_CREATE_EXEC_CTX         = "nvinferEngineCreateExecutionContext";
    private static final String FN_DESTROY_EXEC_CTX        = "nvinferDestroyExecutionContext";
    private static final String FN_GET_NB_IO_TENSORS       = "nvinferEngineGetNbIOTensors";
    private static final String FN_GET_IO_TENSOR_NAME      = "nvinferEngineGetIOTensorName";
    private static final String FN_GET_TENSOR_IO_MODE      = "nvinferEngineGetTensorIOMode";
    private static final String FN_GET_BYTES_PER_COMPONENT = "nvinferEngineGetTensorBytesPerComponent";
    private static final String FN_SET_TENSOR_ADDRESS      = "nvinferExecutionContextSetTensorAddress";
    private static final String FN_GET_TENSOR_ADDRESS      = "nvinferExecutionContextGetTensorAddress";
    private static final String FN_ENQUEUE_V3              = "nvinferExecutionContextEnqueueV3";
    private static final String FN_GET_NB_DIMS             = "nvinferGetNbTensorDimensions";
    private static final String FN_GET_DIMS                = "nvinferGetTensorDimensions";

    // nvinferEngineGetTensorIOMode return values
    public static final int TRT_IO_MODE_INPUT  = 0;
    public static final int TRT_IO_MODE_OUTPUT = 1;

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    // Null logger shim — TRT expects an ILogger* pointer.
    // We allocate a minimal struct: { vtable*, log_fn* }
    private static MemorySegment nullLoggerPtr = MemorySegment.NULL;

    private TensorRtBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new TensorRtBinding(lk);
            LOG.infof("TensorRtBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load TensorRT from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new TensorRtBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new TensorRtBinding(null);
        LOG.info("TensorRtBinding: CPU fallback mode");
    }

    public static TensorRtBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("TensorRtBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Runtime ───────────────────────────────────────────────────────────────

    /**
     * Create a TensorRT runtime.
     *
     * <p>The runtime is required to deserialise engine files. Pass a null
     * logger pointer — TRT 10 accepts NULL for the logger and falls back
     * to its default stderr logger.
     *
     * @return opaque {@code IRuntime*} pointer
     */
    public MemorySegment createRuntime() {
        if (!nativeAvailable) throw unsupported("createRuntime");
        try {
            return (MemorySegment) invoke(FN_CREATE_RUNTIME, MemorySegment.NULL);
        } catch (Throwable t) { throw new RuntimeException("TRT createRuntime failed", t); }
    }

    public void destroyRuntime(MemorySegment runtime) {
        if (!nativeAvailable || isNull(runtime)) return;
        try { invoke(FN_DESTROY_RUNTIME, runtime); }
        catch (Throwable t) { LOG.warnf("TRT destroyRuntime failed: %s", t.getMessage()); }
    }

    // ── Engine ────────────────────────────────────────────────────────────────

    /**
     * Deserialise a compiled TRT engine from a byte array.
     *
     * <p>The engine blob is typically read from a {@code .engine} or {@code .trt}
     * file on disk. TRT validates the engine against the current GPU's SM version.
     *
     * @param runtime   IRuntime* from {@link #createRuntime}
     * @param engineBlob raw bytes of the serialised engine
     * @return opaque {@code ICudaEngine*} pointer
     */
    public MemorySegment deserializeEngine(MemorySegment runtime, byte[] engineBlob) {
        if (!nativeAvailable) throw unsupported("deserializeEngine");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment blobSeg = a.allocate(engineBlob.length, 1);
            for (int i = 0; i < engineBlob.length; i++)
                blobSeg.setAtIndex(ValueLayout.JAVA_BYTE, i, engineBlob[i]);
            return (MemorySegment) invoke(FN_DESERIALIZE_ENGINE,
                    runtime, blobSeg, (long) engineBlob.length);
        } catch (Throwable t) { throw new RuntimeException("TRT deserializeEngine failed", t); }
    }

    /**
     * Deserialise a TRT engine from an off-heap {@link MemorySegment}.
     * Zero-copy path: the segment is passed directly to TRT.
     */
    public MemorySegment deserializeEngineSegment(MemorySegment runtime,
                                                   MemorySegment blobSeg) {
        if (!nativeAvailable) throw unsupported("deserializeEngineSegment");
        try {
            return (MemorySegment) invoke(FN_DESERIALIZE_ENGINE,
                    runtime, blobSeg, blobSeg.byteSize());
        } catch (Throwable t) {
            throw new RuntimeException("TRT deserializeEngineSegment failed", t);
        }
    }

    public void destroyEngine(MemorySegment engine) {
        if (!nativeAvailable || isNull(engine)) return;
        try { invoke(FN_DESTROY_ENGINE, engine); }
        catch (Throwable t) { LOG.warnf("TRT destroyEngine failed: %s", t.getMessage()); }
    }

    // ── Execution Context ─────────────────────────────────────────────────────

    /**
     * Create an execution context from a loaded engine.
     *
     * <p>Each context holds per-inference state (CUDA device buffers for
     * activations, scratchpad memory). Multiple contexts can be created from
     * the same engine for concurrent batch execution.
     *
     * @param engine ICudaEngine* from {@link #deserializeEngine}
     * @return opaque {@code IExecutionContext*}
     */
    public MemorySegment createExecutionContext(MemorySegment engine) {
        if (!nativeAvailable) throw unsupported("createExecutionContext");
        try {
            return (MemorySegment) invoke(FN_CREATE_EXEC_CTX, engine);
        } catch (Throwable t) {
            throw new RuntimeException("TRT createExecutionContext failed", t);
        }
    }

    public void destroyExecutionContext(MemorySegment ctx) {
        if (!nativeAvailable || isNull(ctx)) return;
        try { invoke(FN_DESTROY_EXEC_CTX, ctx); }
        catch (Throwable t) { LOG.warnf("TRT destroyExecutionContext failed: %s", t.getMessage()); }
    }

    // ── Engine introspection ──────────────────────────────────────────────────

    /**
     * Number of IO tensors (inputs + outputs) in this engine.
     */
    public int getNbIOTensors(MemorySegment engine) {
        if (!nativeAvailable) return 0;
        try { return (int) invoke(FN_GET_NB_IO_TENSORS, engine); }
        catch (Throwable t) { return 0; }
    }

    /**
     * Name of IO tensor at position {@code index}.
     *
     * @return tensor name string (owned by TRT — do not free)
     */
    public String getIOTensorName(MemorySegment engine, int index) {
        if (!nativeAvailable) return "tensor_" + index;
        try {
            MemorySegment namePtr = (MemorySegment) invoke(FN_GET_IO_TENSOR_NAME, engine, index);
            return isNull(namePtr) ? "tensor_" + index
                    : namePtr.reinterpret(256).getString(0);
        } catch (Throwable t) { return "tensor_" + index; }
    }

    /**
     * IO mode of tensor: {@link #TRT_IO_MODE_INPUT} or {@link #TRT_IO_MODE_OUTPUT}.
     */
    public int getTensorIOMode(MemorySegment engine, String tensorName) {
        if (!nativeAvailable) return TRT_IO_MODE_INPUT;
        try (Arena a = Arena.ofConfined()) {
            return (int) invoke(FN_GET_TENSOR_IO_MODE, engine, a.allocateFrom(tensorName));
        } catch (Throwable t) { return TRT_IO_MODE_INPUT; }
    }

    /**
     * Bytes per element component for a tensor.
     * For FP16 returns 2, FP32 returns 4, INT8 returns 1.
     */
    public long getBytesPerComponent(MemorySegment engine, String tensorName) {
        if (!nativeAvailable) return 4L;
        try (Arena a = Arena.ofConfined()) {
            return (long) invoke(FN_GET_BYTES_PER_COMPONENT,
                    engine, a.allocateFrom(tensorName), 0);
        } catch (Throwable t) { return 4L; }
    }

    /**
     * Number of dimensions for a named tensor.
     */
    public int getNbTensorDimensions(MemorySegment engine, String tensorName) {
        if (!nativeAvailable) return 0;
        try (Arena a = Arena.ofConfined()) {
            return (int) invoke(FN_GET_NB_DIMS, engine, a.allocateFrom(tensorName));
        } catch (Throwable t) { return 0; }
    }

    /**
     * Shape dimensions for a named tensor, returned as a long[].
     */
    public long[] getTensorDimensions(MemorySegment engine, String tensorName) {
        if (!nativeAvailable) return new long[0];
        int nbDims = getNbTensorDimensions(engine, tensorName);
        if (nbDims <= 0) return new long[0];
        try (Arena a = Arena.ofConfined()) {
            MemorySegment dimsSeg = a.allocate((long) nbDims * 8L, 8);
            invoke(FN_GET_DIMS, engine, a.allocateFrom(tensorName), dimsSeg, nbDims);
            long[] dims = new long[nbDims];
            for (int i = 0; i < nbDims; i++)
                dims[i] = dimsSeg.getAtIndex(ValueLayout.JAVA_LONG, i);
            return dims;
        } catch (Throwable t) { return new long[0]; }
    }

    // ── Execution ─────────────────────────────────────────────────────────────

    /**
     * Bind a CUDA device pointer to a named tensor in the execution context.
     *
     * <p>Must be called for every input and output tensor before
     * {@link #enqueueV3}. The device pointer must have been allocated with
     * {@code cudaMalloc} and remain valid until the CUDA stream completes.
     *
     * @param ctx        IExecutionContext*
     * @param tensorName tensor name (from {@link #getIOTensorName})
     * @param devicePtr  CUDA device pointer
     * @return true on success
     */
    public boolean setTensorAddress(MemorySegment ctx,
                                    String tensorName,
                                    MemorySegment devicePtr) {
        if (!nativeAvailable) return false;
        try (Arena a = Arena.ofConfined()) {
            return (boolean) invoke(FN_SET_TENSOR_ADDRESS,
                    ctx, a.allocateFrom(tensorName), devicePtr);
        } catch (Throwable t) {
            throw new RuntimeException("TRT setTensorAddress(" + tensorName + ") failed", t);
        }
    }

    /**
     * Get the CUDA device pointer currently bound to a tensor.
     */
    public MemorySegment getTensorAddress(MemorySegment ctx, String tensorName) {
        if (!nativeAvailable) return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            return (MemorySegment) invoke(FN_GET_TENSOR_ADDRESS,
                    ctx, a.allocateFrom(tensorName));
        } catch (Throwable t) { return MemorySegment.NULL; }
    }

    /**
     * Asynchronously enqueue inference on a CUDA stream.
     *
     * <p>This is the TRT 10 API. All IO tensor addresses must be set via
     * {@link #setTensorAddress} before calling. Returns immediately after
     * enqueueing — the caller must synchronise the stream to ensure completion.
     *
     * @param ctx    IExecutionContext*
     * @param stream CUDA stream handle (opaque pointer, use 0 for default)
     * @return true on success
     */
    public boolean enqueueV3(MemorySegment ctx, MemorySegment stream) {
        if (!nativeAvailable) return false;
        try {
            return (boolean) invoke(FN_ENQUEUE_V3, ctx, stream);
        } catch (Throwable t) {
            throw new RuntimeException("TRT enqueueV3 failed", t);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // void* nvinferCreateRuntime(void* logger) -> pointer
        bind(FN_CREATE_RUNTIME, FunctionDescriptor.of(
                ValueLayout.ADDRESS,   // return: IRuntime*
                ValueLayout.ADDRESS    // logger (NULL accepted)
        ));

        // void nvinferDestroyRuntime(void*)
        bind(FN_DESTROY_RUNTIME, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void* nvinferRuntimeDeserializeCudaEngine(void*, void*, size_t) -> pointer
        bind(FN_DESERIALIZE_ENGINE, FunctionDescriptor.of(
                ValueLayout.ADDRESS,   // return: ICudaEngine*
                ValueLayout.ADDRESS,   // runtime
                ValueLayout.ADDRESS,   // blob
                ValueLayout.JAVA_LONG  // size
        ));

        // void nvinferDestroyEngine(void*)
        bind(FN_DESTROY_ENGINE, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void* nvinferEngineCreateExecutionContext(void*) -> pointer
        bind(FN_CREATE_EXEC_CTX, FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS    // engine
        ));

        // void nvinferDestroyExecutionContext(void*)
        bind(FN_DESTROY_EXEC_CTX, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // int32_t nvinferEngineGetNbIOTensors(void*) -> int
        bind(FN_GET_NB_IO_TENSORS, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS
        ));

        // const char* nvinferEngineGetIOTensorName(void*, int32_t) -> pointer
        bind(FN_GET_IO_TENSOR_NAME, FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT
        ));

        // int32_t nvinferEngineGetTensorIOMode(void*, const char*) -> int
        bind(FN_GET_TENSOR_IO_MODE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // engine
                ValueLayout.ADDRESS    // tensor name
        ));

        // int64_t nvinferEngineGetTensorBytesPerComponent(void*, const char*, int32_t) -> long
        bind(FN_GET_BYTES_PER_COMPONENT, FunctionDescriptor.of(
                ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS,   // engine
                ValueLayout.ADDRESS,   // tensor name
                ValueLayout.JAVA_INT   // component index (0 for scalars)
        ));

        // bool nvinferExecutionContextSetTensorAddress(void*, const char*, void*) -> byte (bool)
        bind(FN_SET_TENSOR_ADDRESS, FunctionDescriptor.of(
                ValueLayout.JAVA_BOOLEAN,
                ValueLayout.ADDRESS,   // ctx
                ValueLayout.ADDRESS,   // tensor name
                ValueLayout.ADDRESS    // device pointer
        ));

        // void* nvinferExecutionContextGetTensorAddress(void*, const char*) -> pointer
        bind(FN_GET_TENSOR_ADDRESS, FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,   // ctx
                ValueLayout.ADDRESS    // tensor name
        ));

        // bool nvinferExecutionContextEnqueueV3(void*, cudaStream_t) -> byte (bool)
        bind(FN_ENQUEUE_V3, FunctionDescriptor.of(
                ValueLayout.JAVA_BOOLEAN,
                ValueLayout.ADDRESS,   // ctx
                ValueLayout.ADDRESS    // stream
        ));

        // int32_t nvinferGetNbTensorDimensions(void*, const char*) -> int
        bind(FN_GET_NB_DIMS, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,   // engine
                ValueLayout.ADDRESS    // tensor name
        ));

        // bool nvinferGetTensorDimensions(void*, const char*, int64_t*, int32_t) -> bool
        bind(FN_GET_DIMS, FunctionDescriptor.of(
                ValueLayout.JAVA_BOOLEAN,
                ValueLayout.ADDRESS,   // engine
                ValueLayout.ADDRESS,   // tensor name
                ValueLayout.ADDRESS,   // dims int64_t* out
                ValueLayout.JAVA_INT   // maxDims
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("TensorRtBinding: bound %s", name);
        } else {
            LOG.warnf("TensorRtBinding: symbol not found — %s", name);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Object invoke(String fn, Object... args) {
        MethodHandle mh = methodHandles.get(fn);
        if (mh == null) throw new IllegalStateException("TRT symbol not bound: " + fn);
        try { return mh.invokeWithArguments(args); }
        catch (Throwable t) { throw new RuntimeException("TRT " + fn + " failed", t); }
    }

    private static boolean isNull(MemorySegment s) {
        return s == null || s.equals(MemorySegment.NULL) || s.address() == 0;
    }

    private static IllegalStateException unsupported(String fn) {
        return new IllegalStateException(
                "TensorRtBinding." + fn + ": native library not available");
    }

    public static void reset() { instance = null; }
}
