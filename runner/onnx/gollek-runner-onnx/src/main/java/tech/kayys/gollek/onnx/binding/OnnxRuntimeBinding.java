package tech.kayys.gollek.onnx.binding;

import jakarta.enterprise.context.ApplicationScoped;
import io.quarkus.arc.Unremovable;
import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM binding to the ONNX Runtime C API ({@code onnxruntime_c_api.h}).
 *
 * <p>
 * Loads {@code libonnxruntime.so} (Linux) or {@code libonnxruntime.dylib}
 * (macOS) at runtime. Mirrors the singleton pattern of
 * {@link tech.kayys.gollek.kernel.fa3.FlashAttention3Binding} exactly.
 *
 * <h2>ONNX Runtime architecture</h2>
 * <p>
 * All ORT operations go through a versioned vtable obtained by calling
 * {@code OrtGetApiBase()->GetApi(ORT_API_VERSION)}. The vtable is a C struct
 * of function pointers. This binding calls each function pointer through
 * a dedicated FFM {@link MethodHandle}.
 *
 * <h2>Execution Providers (EPs)</h2>
 * <p>
 * ONNX Runtime routes operators to accelerator-specific kernels via EPs:
 * <ul>
 * <li><b>CPUExecutionProvider</b> — always available; MLAS for SIMD
 * matmuls.</li>
 * <li><b>CUDAExecutionProvider</b> — NVIDIA CUDA (all sm_* ≥ sm_37).</li>
 * <li><b>ROCMExecutionProvider</b> — AMD ROCm/HIP (MI200/MI300 family).</li>
 * <li><b>TensorrtExecutionProvider</b> — wraps TensorRT for optimal NVIDIA
 * inference.</li>
 * <li><b>CoreMLExecutionProvider</b> — Apple Neural Engine + Metal on
 * macOS/iOS.</li>
 * <li><b>DirectMLExecutionProvider</b> — DirectX 12 ML on Windows.</li>
 * <li><b>OpenVINOExecutionProvider</b> — Intel CPUs/iGPUs/VPUs.</li>
 * </ul>
 *
 * <h2>C functions bound</h2>
 * 
 * <pre>
 * const OrtApiBase* OrtGetApiBase();
 *
 * // Via OrtApi vtable:
 * OrtStatus* CreateEnv(OrtLoggingLevel, char* logid, OrtEnv**)
 * OrtStatus* CreateSessionOptions(OrtSessionOptions**)
 * OrtStatus* SetIntraOpNumThreads(OrtSessionOptions*, int)
 * OrtStatus* SetInterOpNumThreads(OrtSessionOptions*, int)
 * OrtStatus* SetSessionGraphOptimizationLevel(OrtSessionOptions*, GraphOptimizationLevel)
 * OrtStatus* SessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions*, OrtCUDAProviderOptions*)
 * OrtStatus* SessionOptionsAppendExecutionProvider_ROCM(OrtSessionOptions*, OrtROCMProviderOptions*)
 * OrtStatus* CreateSession(OrtEnv*, char* path, OrtSessionOptions*, OrtSession**)
 * OrtStatus* CreateRunOptions(OrtRunOptions**)
 * size_t     SessionGetInputCount(OrtSession*, size_t*)
 * size_t     SessionGetOutputCount(OrtSession*, size_t*)
 * OrtStatus* SessionGetInputName(OrtSession*, size_t, OrtAllocator*, char**)
 * OrtStatus* SessionGetOutputName(OrtSession*, size_t, OrtAllocator*, char**)
 * OrtStatus* CreateCpuMemoryInfo(OrtAllocatorType, OrtMemType, OrtMemoryInfo**)
 * OrtStatus* CreateTensorWithDataAsOrtValue(OrtMemoryInfo*, void*, size_t,
 *                int64_t* shape, size_t shape_len, ONNXTensorElementDataType, OrtValue**)
 * OrtStatus* Run(OrtSession*, OrtRunOptions*, char**, OrtValue**, size_t,
 *                char**, size_t, OrtValue**)
 * OrtStatus* GetTensorMutableData(OrtValue*, void**)
 * OrtStatus* GetTensorTypeAndShape(OrtValue*, OrtTensorTypeAndShapeInfo**)
 * OrtStatus* GetDimensionsCount(OrtTensorTypeAndShapeInfo*, size_t*)
 * OrtStatus* GetDimensions(OrtTensorTypeAndShapeInfo*, int64_t*, size_t)
 * void       ReleaseSession(OrtSession*)
 * void       ReleaseSessionOptions(OrtSessionOptions*)
 * void       ReleaseEnv(OrtEnv*)
 * void       ReleaseValue(OrtValue*)
 * void       ReleaseRunOptions(OrtRunOptions*)
 * void       ReleaseMemoryInfo(OrtMemoryInfo*)
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * 
 * <pre>{@code
 * OnnxRuntimeBinding.initialize(
 *         Path.of("/usr/lib/libonnxruntime.so"));
 * OnnxRuntimeBinding ort = OnnxRuntimeBinding.getInstance();
 *
 * MemorySegment env = ort.createEnv("gollek");
 * MemorySegment opts = ort.createSessionOptions();
 * ort.setIntraOpNumThreads(opts, 4);
 * ort.appendCudaProvider(opts, 0); // GPU device 0
 *
 * MemorySegment session = ort.createSession(env, "/models/model.onnx", opts);
 *
 * // Run inference:
 * MemorySegment inputVal = ort.createTensorFloat(memInfo, inputData, shape);
 * MemorySegment outputVal = ort.run(session, inputVal, inputName, outputName);
 * float[] logits = ort.getTensorDataFloat(outputVal, vocabSize);
 *
 * ort.releaseValue(outputVal);
 * ort.releaseSession(session);
 * ort.releaseEnv(env);
 * }</pre>
 *
 * <h2>Build / install ONNX Runtime</h2>
 * 
 * <pre>
 * # Gollek installer (downloads a prebuilt archive + installs to ~/.gollek/libs):
 * make -C inference-gollek/extension/runner/onnx/gollek-runner-onnx/src/main/cpp/onnxruntime install
 *
 * # Linux (from pip wheel, fastest):
 * pip install onnxruntime-gpu
 * find /usr -name "libonnxruntime.so" 2>/dev/null
 *
 * # Or from release:
 * wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/
 *      onnxruntime-linux-x64-gpu-1.19.2.tgz
 * tar xzf onnxruntime-linux-x64-gpu-1.19.2.tgz
 * cp lib/libonnxruntime.so /opt/gollek/lib/
 *
 * # macOS:
 * brew install onnxruntime
 * </pre>
 */
@ApplicationScoped
@Unremovable
public class OnnxRuntimeBinding {

    private static final Logger LOG = Logger.getLogger(OnnxRuntimeBinding.class);
    private static volatile OnnxRuntimeBinding instance;

    // ── ORT function names (resolved through the vtable) ─────────────────────
    private static final String FN_GET_API_BASE = "OrtGetApiBase";

    // Stored vtable slot indices (OrtApi struct, ORT 1.19.2)
    // Counted from struct OrtApi { in onnxruntime_c_api.h
    // Each ORT_API2_STATUS / ORT_CLASS_RELEASE entry is one slot (0-indexed)
    private static final int SLOT_CREATE_STATUS = 0;
    private static final int SLOT_GET_ERROR_CODE = 1;
    private static final int SLOT_GET_ERROR_MESSAGE = 2;
    private static final int SLOT_RELEASE_STATUS = 93;
    private static final int SLOT_CREATE_ENV = 3;
    private static final int SLOT_CREATE_SESSION_OPTIONS = 10;
    private static final int SLOT_SET_INTRA_OP_THREADS = 24;
    private static final int SLOT_SET_INTER_OP_THREADS = 25;
    private static final int SLOT_SET_GRAPH_OPT_LEVEL = 23;
    private static final int SLOT_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS = 78;
    private static final int SLOT_APPEND_CUDA_EP = 152;
    private static final int SLOT_APPEND_ROCM_EP = 153;
    private static final int SLOT_CREATE_SESSION = 7;
    private static final int SLOT_CREATE_RUN_OPTIONS = 39;
    private static final int SLOT_SESSION_INPUT_COUNT = 30;
    private static final int SLOT_SESSION_OUTPUT_COUNT = 31;
    private static final int SLOT_SESSION_GET_INPUT_NAME = 36;
    private static final int SLOT_SESSION_GET_OUTPUT_NAME = 37;
    private static final int SLOT_CREATE_CPU_MEMORY_INFO = 69;
    private static final int SLOT_CREATE_TENSOR_WITH_DATA = 49;
    private static final int SLOT_RUN = 9;
    private static final int SLOT_GET_TENSOR_MUTABLE_DATA = 51;
    private static final int SLOT_GET_TENSOR_TYPE_AND_SHAPE = 65;
    private static final int SLOT_GET_DIMENSIONS_COUNT = 61;
    private static final int SLOT_GET_DIMENSIONS = 62;
    private static final int SLOT_RELEASE_SESSION = 95;
    private static final int SLOT_RELEASE_SESSION_OPTIONS = 100;
    private static final int SLOT_RELEASE_ENV = 92;
    private static final int SLOT_RELEASE_VALUE = 96;
    private static final int SLOT_RELEASE_RUN_OPTIONS = 97;
    private static final int SLOT_RELEASE_MEMORY_INFO = 94;


    // ORT enum constants
    public static final int ORT_LOGGING_LEVEL_WARNING = 2;
    public static final int ORT_LOGGING_LEVEL_ERROR = 3;
    public static final int GRAPH_OPT_LEVEL_ENABLE_ALL = 99;
    public static final int ONNX_TENSOR_FLOAT = 1; // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    public static final int ONNX_TENSOR_INT32 = 6;
    public static final int ONNX_TENSOR_INT64 = 7;
    public static final int ORT_DEVICE_ALLOCATOR = 0;
    public static final int ORT_MEM_TYPE_CPU_INPUT = -2;

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    // vtable pointer — populated from OrtGetApiBase()->GetApi()
    private MemorySegment ortApiVtable = MemorySegment.NULL;
    // vtable slot method handles (resolved lazily from the vtable pointer)
    private final Map<Integer, MethodHandle> vtableHandles = new ConcurrentHashMap<>();

    protected OnnxRuntimeBinding() {
        this.lookup = null;
        this.nativeAvailable = false;
    }

    private OnnxRuntimeBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) {
            bindGetApiBase();
            resolveVtable();
        }
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null)
            return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new OnnxRuntimeBinding(lk);
            LOG.infof("OnnxRuntimeBinding loaded from %s (native=%s)",
                    libraryPath, instance.nativeAvailable);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load ONNX Runtime from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new OnnxRuntimeBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new OnnxRuntimeBinding(null);
        LOG.info("OnnxRuntimeBinding: CPU fallback mode");
    }

    public static OnnxRuntimeBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("OnnxRuntimeBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Environment ───────────────────────────────────────────────────────────

    /**
     * Create an {@code OrtEnv} — the root ORT object.
     *
     * @param logId label used in ORT log messages
     * @return opaque OrtEnv* pointer (must be released with {@link #releaseEnv})
     */
    public MemorySegment createEnv(String logId) {
        if (!nativeAvailable)
            throw unsupported("createEnv");
        try (Arena a = Arena.ofConfined()) {
            // CRITICAL: logId must remain valid for the entire lifetime of the OrtEnv.
            // Using Arena.global() for the string ensures it doesn't dangle.
            MemorySegment logIdSeg = Arena.global().allocateFrom(logId);
            MemorySegment envPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_CREATE_ENV,
                    FunctionDescriptor.of(
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, // logging level
                            ValueLayout.ADDRESS, // log_id
                            ValueLayout.ADDRESS // OrtEnv** out
                    )).invokeExact(ORT_LOGGING_LEVEL_WARNING, logIdSeg, envPtr);
            checkStatusPtr(status, "CreateEnv");
            return envPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT CreateEnv failed", t);
        }
    }

    public void releaseEnv(MemorySegment env) {
        if (!nativeAvailable || isNull(env))
            return;
        try {
            vtable(SLOT_RELEASE_ENV,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(env);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseEnv failed: %s", t.getMessage());
        }
    }

    // ── Session Options ───────────────────────────────────────────────────────

    /**
     * Create {@code OrtSessionOptions}.
     */
    public MemorySegment createSessionOptions() {
        if (!nativeAvailable)
            throw unsupported("createSessionOptions");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment optsPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_CREATE_SESSION_OPTIONS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(optsPtr);
            checkStatusPtr(status, "CreateSessionOptions");
            return optsPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT CreateSessionOptions failed", t);
        }
    }

    public void setIntraOpNumThreads(MemorySegment opts, int threads) {
        if (!nativeAvailable)
            return;
        try {
            MemorySegment status = (MemorySegment) vtable(SLOT_SET_INTRA_OP_THREADS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT))
                    .invokeExact(opts, threads);
            checkStatusPtr(status, "SetIntraOpNumThreads");
        } catch (Throwable t) {
            throw new RuntimeException("ORT SetIntraOpNumThreads failed", t);
        }
    }

    public void setInterOpNumThreads(MemorySegment opts, int threads) {
        if (!nativeAvailable)
            return;
        try {
            MemorySegment status = (MemorySegment) vtable(SLOT_SET_INTER_OP_THREADS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT))
                    .invokeExact(opts, threads);
            checkStatusPtr(status, "SetInterOpNumThreads");
        } catch (Throwable t) {
            throw new RuntimeException("ORT SetInterOpNumThreads failed", t);
        }
    }

    public void setGraphOptimizationLevel(MemorySegment opts, int level) {
        if (!nativeAvailable)
            return;
        try {
            MemorySegment status = (MemorySegment) vtable(SLOT_SET_GRAPH_OPT_LEVEL,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT))
                    .invokeExact(opts, level);
            checkStatusPtr(status, "SetGraphOptimizationLevel");
        } catch (Throwable t) {
            throw new RuntimeException("ORT SetGraphOptimizationLevel failed", t);
        }
    }

    /**
     * Append the CUDA Execution Provider to a session's options.
     *
     * @param opts     OrtSessionOptions*
     * @param deviceId CUDA device index (0 = first GPU)
     */
    public void appendCudaProvider(MemorySegment opts, int deviceId) {
        if (!nativeAvailable)
            return;
        try (Arena a = Arena.ofConfined()) {
            // OrtCUDAProviderOptions: first field is device_id (int32), rest default to 0
            MemorySegment cudaOpts = a.allocate(256L, 8);
            cudaOpts.setAtIndex(ValueLayout.JAVA_INT, 0, deviceId);
            vtable(SLOT_APPEND_CUDA_EP,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(opts, cudaOpts);
        } catch (Throwable t) {
            LOG.warnf("ORT CUDA EP not available: %s", t.getMessage());
        }
    }

    /**
     * Append the ROCm Execution Provider to a session's options.
     *
     * @param opts     OrtSessionOptions*
     * @param deviceId ROCm device index
     */
    public void appendRocmProvider(MemorySegment opts, int deviceId) {
        if (!nativeAvailable)
            return;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment rocmOpts = a.allocate(256L, 8);
            rocmOpts.setAtIndex(ValueLayout.JAVA_INT, 0, deviceId);
            vtable(SLOT_APPEND_ROCM_EP,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(opts, rocmOpts);
        } catch (Throwable t) {
            LOG.warnf("ORT ROCm EP not available: %s", t.getMessage());
        }
    }

    /**
     * Append the CoreML Execution Provider to a session's options.
     *
     * <p>Uses the deprecated C API entrypoint when available:
     * {@code OrtSessionOptionsAppendExecutionProvider_CoreML(OrtSessionOptions*, uint32_t)}.
     *
     * @param opts  OrtSessionOptions*
     * @param flags CoreML flags bitmask (0 = defaults)
     * @return true when the CoreML EP was successfully appended
     */
    public boolean appendCoreMlProvider(MemorySegment opts, int flags) {
        if (!nativeAvailable)
            return false;
        try {
            Optional<MethodHandle> handle = bindDirect(
                    "OrtSessionOptionsAppendExecutionProvider_CoreML",
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT));
            if (handle.isEmpty()) {
                return false;
            }
            MemorySegment status = (MemorySegment) handle.get().invokeExact(opts, flags);
            if (status != null && !status.equals(MemorySegment.NULL)) {
                LOG.warn("ORT CoreML EP append returned a non-null status");
                return false;
            }
            return true;
        } catch (Throwable t) {
            LOG.warnf("ORT CoreML EP not available: %s", t.getMessage());
            return false;
        }
    }

    public boolean supportsCoreMl() {
        if (!nativeAvailable)
            return false;
        return lookup.find("OrtSessionOptionsAppendExecutionProvider_CoreML").isPresent();
    }

    public void releaseSessionOptions(MemorySegment opts) {
        if (!nativeAvailable || isNull(opts))
            return;
        try {
            vtable(SLOT_RELEASE_SESSION_OPTIONS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(opts);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseSessionOptions failed: %s", t.getMessage());
        }
    }

    // ── Session ───────────────────────────────────────────────────────────────

    /**
     * Load an ONNX model and create an inference session.
     *
     * @param env       OrtEnv* from {@link #createEnv}
     * @param modelPath path to the {@code .onnx} file
     * @param opts      OrtSessionOptions*
     * @return OrtSession* (must be released with {@link #releaseSession})
     */
    public MemorySegment createSession(MemorySegment env, String modelPath, MemorySegment opts) {
        if (!nativeAvailable)
            throw unsupported("createSession");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pathSeg = a.allocateFrom(modelPath);
            MemorySegment sessionPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_CREATE_SESSION,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, // env
                            ValueLayout.ADDRESS, // model_path
                            ValueLayout.ADDRESS, // options
                            ValueLayout.ADDRESS // OrtSession** out
                    )).invokeExact(env, pathSeg, opts, sessionPtr);
            checkStatusPtr(status, "CreateSession for " + modelPath);
            return sessionPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT CreateSession failed", t);
        }
    }

    public long getInputCount(MemorySegment session) {
        if (!nativeAvailable)
            return 0;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment countPtr = a.allocate(ValueLayout.JAVA_LONG);
            MemorySegment status = (MemorySegment) vtable(SLOT_SESSION_INPUT_COUNT,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(session, countPtr);
            checkStatusPtr(status, "SessionGetInputCount");
            return countPtr.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            return 0;
        }
    }

    public long getOutputCount(MemorySegment session) {
        if (!nativeAvailable)
            return 0;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment countPtr = a.allocate(ValueLayout.JAVA_LONG);
            MemorySegment status = (MemorySegment) vtable(SLOT_SESSION_OUTPUT_COUNT,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(session, countPtr);
            checkStatusPtr(status, "SessionGetOutputCount");
            return countPtr.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            return 0;
        }
    }

    public MemorySegment getAllocatorWithDefaultOptions() {
        if (!nativeAvailable)
            throw unsupported("getAllocatorWithDefaultOptions");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment allocPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(allocPtr);
            checkStatusPtr(status, "GetAllocatorWithDefaultOptions");
            return allocPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT GetAllocatorWithDefaultOptions failed", t);
        }
    }

    /**
     * Get the name of input tensor at {@code index}.
     * The returned string is allocated by ORT's default allocator and
     * must be freed via {@code OrtAllocator::Free} — callers should copy it.
     */
    public String getInputName(MemorySegment session, long index) {
        if (!nativeAvailable)
            return "input_" + index;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment namePtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment allocator = getAllocatorWithDefaultOptions();
            MemorySegment status = (MemorySegment) vtable(SLOT_SESSION_GET_INPUT_NAME,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(session, index, allocator, namePtr);
            // We check status ptr inside this try-catch to avoid native failures
            if (!isNull(status)) {
                // If it fails, return a synthetic name
                return "input_" + index;
            }
            MemorySegment str = namePtr.get(ValueLayout.ADDRESS, 0);
            return isNull(str) ? "input_" + index : str.reinterpret(256).getString(0);
        } catch (Throwable t) {
            return "input_" + index;
        }
    }

    public String getOutputName(MemorySegment session, long index) {
        if (!nativeAvailable)
            return "output_" + index;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment namePtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment allocator = getAllocatorWithDefaultOptions();
            MemorySegment status = (MemorySegment) vtable(SLOT_SESSION_GET_OUTPUT_NAME,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(session, index, allocator, namePtr);
            if (!isNull(status)) {
                return "output_" + index;
            }
            MemorySegment str = namePtr.get(ValueLayout.ADDRESS, 0);
            return isNull(str) ? "output_" + index : str.reinterpret(256).getString(0);
        } catch (Throwable t) {
            return "output_" + index;
        }
    }

    public long getDimensionsCount(MemorySegment info) {
        if (!nativeAvailable)
            return 0;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(ValueLayout.JAVA_LONG);
            MemorySegment status = (MemorySegment) vtable(SLOT_GET_DIMENSIONS_COUNT,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(info, out);
            checkStatusPtr(status, "GetDimensionsCount");
            return out.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            return 0;
        }
    }

    public long[] getDimensions(MemorySegment info) {
        if (!nativeAvailable)
            return new long[0];
        long count = getDimensionsCount(info);
        if (count <= 0)
            return new long[0];
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(count * 8L, 8);
            MemorySegment status = (MemorySegment) vtable(SLOT_GET_DIMENSIONS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG))
                    .invokeExact(info, out, count);
            checkStatusPtr(status, "GetDimensions");
            long[] dims = new long[(int) count];
            for (int i = 0; i < (int) count; i++)
                dims[i] = out.getAtIndex(ValueLayout.JAVA_LONG, i);
            return dims;
        } catch (Throwable t) {
            return new long[0];
        }
    }

    public void releaseSession(MemorySegment session) {
        if (!nativeAvailable || isNull(session))
            return;
        try {
            vtable(SLOT_RELEASE_SESSION,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(session);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseSession failed: %s", t.getMessage());
        }
    }

    // ── Tensor creation ───────────────────────────────────────────────────────

    /**
     * Create a CPU memory info object (required for tensor allocation).
     *
     * @return OrtMemoryInfo* (must be released with {@link #releaseMemoryInfo})
     */
    public MemorySegment createCpuMemoryInfo() {
        if (!nativeAvailable)
            throw unsupported("createCpuMemoryInfo");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment infoPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_CREATE_CPU_MEMORY_INFO,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, // allocator type: OrtDeviceAllocator=0
                            ValueLayout.JAVA_INT, // mem type: OrtMemTypeDefault=0
                            ValueLayout.ADDRESS // OrtMemoryInfo** out
                    )).invokeExact(ORT_DEVICE_ALLOCATOR, 0, infoPtr);
            checkStatusPtr(status, "CreateCpuMemoryInfo");
            return infoPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT CreateCpuMemoryInfo failed", t);
        }
    }

    /**
     * Wrap an existing {@link MemorySegment} as an OrtValue tensor (zero copy).
     *
     * <p>
     * The segment must remain valid for the lifetime of the returned OrtValue.
     * On unified-memory systems (Apple Silicon, CUDA UVM) the same segment
     * can be passed directly as GPU input.
     *
     * @param memInfo  OrtMemoryInfo* from {@link #createCpuMemoryInfo}
     * @param data     raw data buffer (float32 or int64)
     * @param shape    tensor shape
     * @param dataType {@link #ONNX_TENSOR_FLOAT} or {@link #ONNX_TENSOR_INT64}
     * @return OrtValue* (must be released with {@link #releaseValue})
     */
    public MemorySegment createTensorWithData(MemorySegment memInfo,
            MemorySegment data,
            long[] shape,
            int dataType) {
        if (!nativeAvailable)
            throw unsupported("createTensorWithData");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment shapeSeg = a.allocate((long) shape.length * 8L, 8);
            for (int i = 0; i < shape.length; i++)
                shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);

            MemorySegment valPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_CREATE_TENSOR_WITH_DATA,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, // OrtMemoryInfo*
                            ValueLayout.ADDRESS, // data pointer
                            ValueLayout.JAVA_LONG, // data length bytes
                            ValueLayout.ADDRESS, // shape int64*
                            ValueLayout.JAVA_LONG, // shape length
                            ValueLayout.JAVA_INT, // element type
                            ValueLayout.ADDRESS // OrtValue** out
                    )).invokeExact(
                            memInfo, data, data.byteSize(),
                            shapeSeg, (long) shape.length,
                            dataType, valPtr);
            checkStatusPtr(status, "CreateTensorWithDataAsOrtValue");
            return valPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT CreateTensorWithData failed", t);
        }
    }

    public void releaseValue(MemorySegment value) {
        if (!nativeAvailable || isNull(value))
            return;
        try {
            vtable(SLOT_RELEASE_VALUE,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(value);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseValue failed: %s", t.getMessage());
        }
    }

    public void releaseMemoryInfo(MemorySegment info) {
        if (!nativeAvailable || isNull(info))
            return;
        try {
            vtable(SLOT_RELEASE_MEMORY_INFO,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(info);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseMemoryInfo failed: %s", t.getMessage());
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Run inference.
     *
     * @param session     OrtSession*
     * @param runOpts     OrtRunOptions* (may be NULL for defaults)
     * @param inputNames  input tensor names (must match model graph)
     * @param inputValues OrtValue* array, one per input
     * @param outputNames output tensor names
     * @return OrtValue* array, one per output (must be released)
     */
    public MemorySegment[] run(MemorySegment session,
            MemorySegment runOpts,
            String[] inputNames,
            MemorySegment[] inputValues,
            String[] outputNames) {
        if (!nativeAvailable)
            throw unsupported("run");
        try (Arena a = Arena.ofConfined()) {
            // Pack input name pointers
            MemorySegment inNames = packStrings(a, inputNames);
            MemorySegment inVals = packPtrs(a, inputValues);
            MemorySegment outNames = packStrings(a, outputNames);
            MemorySegment outVals = a.allocate((long) outputNames.length * 8L, 8);

            MemorySegment status = (MemorySegment) vtable(SLOT_RUN,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, // session
                            ValueLayout.ADDRESS, // run options
                            ValueLayout.ADDRESS, // input names**
                            ValueLayout.ADDRESS, // input values**
                            ValueLayout.JAVA_LONG, // input count
                            ValueLayout.ADDRESS, // output names**
                            ValueLayout.JAVA_LONG, // output count
                            ValueLayout.ADDRESS // output values** (out)
                    )).invokeExact(
                            session, runOpts,
                            inNames, inVals, (long) inputValues.length,
                            outNames, (long) outputNames.length,
                            outVals);
            checkStatusPtr(status, "Run");

            MemorySegment[] result = new MemorySegment[outputNames.length];
            for (int i = 0; i < result.length; i++)
                result[i] = outVals.getAtIndex(ValueLayout.ADDRESS, i);
            return result;
        } catch (Throwable t) {
            throw new RuntimeException("ORT Run failed: " + t.getMessage(), t);
        }
    }

    /**
     * Get a raw pointer to the tensor's data buffer.
     * For CPU tensors this is a CPU pointer; for GPU tensors it is a device
     * pointer.
     *
     * @param value OrtValue*
     * @return raw data pointer
     */
    public MemorySegment getTensorMutableData(MemorySegment value) {
        if (!nativeAvailable)
            throw unsupported("getTensorMutableData");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment dataPtr = a.allocate(ValueLayout.ADDRESS);
            MemorySegment status = (MemorySegment) vtable(SLOT_GET_TENSOR_MUTABLE_DATA,
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(value, dataPtr);
            checkStatusPtr(status, "GetTensorMutableData");
            return dataPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            throw new RuntimeException("ORT GetTensorMutableData failed", t);
        }
    }

    /**
     * Read float32 tensor data into a Java array.
     *
     * @param value     OrtValue* holding float32 tensor
     * @param numFloats number of float elements to read
     * @return float array with tensor data
     */
    public float[] getTensorDataFloat(MemorySegment value, int numFloats) {
        if (!nativeAvailable)
            return new float[numFloats];
        MemorySegment dataPtr = getTensorMutableData(value);
        MemorySegment data = dataPtr.reinterpret((long) numFloats * 4L);
        float[] result = new float[numFloats];
        for (int i = 0; i < numFloats; i++)
            result[i] = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        return result;
    }

    // ── Run options ───────────────────────────────────────────────────────────

    public MemorySegment createRunOptions() {
        if (!nativeAvailable)
            return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment optsPtr = a.allocate(ValueLayout.ADDRESS);
            vtable(SLOT_CREATE_RUN_OPTIONS,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(optsPtr);
            return optsPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            return MemorySegment.NULL;
        }
    }

    public void releaseRunOptions(MemorySegment opts) {
        if (!nativeAvailable || isNull(opts))
            return;
        try {
            vtable(SLOT_RELEASE_RUN_OPTIONS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)).invokeExact(opts);
        } catch (Throwable t) {
            LOG.warnf("ORT ReleaseRunOptions failed: %s", t.getMessage());
        }
    }

    // ── Vtable resolution ─────────────────────────────────────────────────────

    /**
     * Call {@code OrtGetApiBase()->GetApi(ORT_API_VERSION)} to obtain the
     * versioned OrtApi vtable pointer. All subsequent ORT calls use slots
     * from this vtable.
     */
    private void resolveVtable() {
        try {
            MethodHandle getApiBase = methodHandles.get(FN_GET_API_BASE);
            if (getApiBase == null) {
                LOG.warn("OrtGetApiBase not bound");
                return;
            }

            // OrtGetApiBase() → OrtApiBase* (struct: GetApi fn ptr, GetVersionString fn
            // ptr)
            MemorySegment apiBase = (MemorySegment) getApiBase.invokeExact();
            if (isNull(apiBase)) {
                LOG.warn("OrtGetApiBase returned NULL");
                return;
            }
            apiBase = apiBase.reinterpret(16); // two function pointers

            // GetApi(uint32_t version) → OrtApi*
            // Version 17 = ORT 1.19.x
            MemorySegment getApiFnPtr = apiBase.get(ValueLayout.ADDRESS, 0);
            MethodHandle getApi = Linker.nativeLinker().downcallHandle(
                    getApiFnPtr,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            MemorySegment ortApi = (MemorySegment) getApi.invokeExact(17);
            if (isNull(ortApi)) {
                LOG.warn("OrtGetApiBase->GetApi(17) returned NULL");
                return;
            }

            // The vtable has ~150 function pointer slots (each 8 bytes on 64-bit)
            ortApiVtable = ortApi.reinterpret(200 * 8L);
            LOG.info("OrtApi vtable resolved (ORT API version 17)");
        } catch (Throwable t) {
            LOG.warnf("OrtApi vtable resolution failed: %s", t.getMessage());
        }
    }

    /**
     * Get or create a {@link MethodHandle} for a vtable slot.
     * The vtable is a flat array of function pointers; slot {@code n} is at
     * byte offset {@code n * 8} (pointer size on 64-bit).
     */
    private MethodHandle vtable(int slot, FunctionDescriptor descriptor) {
        return vtableHandles.computeIfAbsent(slot * 1000 + descriptor.hashCode(), k -> {
            if (isNull(ortApiVtable))
                throw new IllegalStateException("OrtApi vtable not resolved");
            MemorySegment fnPtr = ortApiVtable.getAtIndex(ValueLayout.ADDRESS, slot);
            if (isNull(fnPtr))
                throw new IllegalStateException("OrtApi vtable slot " + slot + " is NULL");
            return Linker.nativeLinker().downcallHandle(fnPtr, descriptor);
        });
    }

    private Optional<MethodHandle> bindDirect(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            return Optional.of(Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
        }
        return Optional.empty();
    }

    // ── FFM wiring (static exports) ───────────────────────────────────────────

    private void bindGetApiBase() {
        bind(FN_GET_API_BASE,
                FunctionDescriptor.of(ValueLayout.ADDRESS)); // () → OrtApiBase*
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("OnnxRuntimeBinding: bound %s", name);
        } else {
            LOG.warnf("OnnxRuntimeBinding: symbol not found — %s", name);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private MemorySegment packStrings(Arena arena, String[] names) {
        MemorySegment ptrs = arena.allocate((long) names.length * 8L, 8);
        for (int i = 0; i < names.length; i++)
            ptrs.setAtIndex(ValueLayout.ADDRESS, i,
                    arena.allocateFrom(names[i]));
        return ptrs;
    }

    private MemorySegment packPtrs(Arena arena, MemorySegment[] segs) {
        MemorySegment ptrs = arena.allocate((long) segs.length * 8L, 8);
        for (int i = 0; i < segs.length; i++)
            ptrs.setAtIndex(ValueLayout.ADDRESS, i, segs[i]);
        return ptrs;
    }

    private void checkStatus(int status, String op) {
        if (status != 0)
            throw new RuntimeException("ORT " + op + " returned non-zero status: " + status);
    }

    /** Check OrtStatus* pointer: NULL means success, non-NULL is an error. */
    private void checkStatusPtr(MemorySegment status, String op) {
        if (isNull(status))
            return;

        String msg = "Unknown error";
        try {
            // GetErrorMessage(OrtStatus*) -> const char*
            MemorySegment msgPtr = (MemorySegment) vtable(SLOT_GET_ERROR_MESSAGE,
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                    .invokeExact(status);
            if (!isNull(msgPtr)) {
                msg = msgPtr.reinterpret(1024).getString(0);
            }
        } catch (Throwable t) {
            msg = "Failed to extract error: " + t.getMessage();
        } finally {
            try {
                // ReleaseStatus(OrtStatus*)
                vtable(SLOT_RELEASE_STATUS,
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS))
                        .invokeExact(status);
            } catch (Throwable t) {
                LOG.warn("Failed to release ORT status", t);
            }
        }
        System.err.println("!!! ORT ERROR: " + op + " failed: " + msg);
        throw new RuntimeException("ORT " + op + " failed: " + msg);
    }

    private static boolean isNull(MemorySegment seg) {
        return seg == null || seg.equals(MemorySegment.NULL) || seg.address() == 0;
    }

    private static IllegalStateException unsupported(String fn) {
        return new IllegalStateException(
                "OnnxRuntimeBinding." + fn + ": native library not available");
    }

    public static void reset() {
        instance = null;
    }
}
