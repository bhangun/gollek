package tech.kayys.gollek.diffuser;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;

/**
 * FFM bridge to the ONNX Runtime C API (v18+).
 *
 * Uses JDK 25 Foreign Function & Memory API exclusively —
 * no JNI, no JNA, no Jna.
 *
 * ORT C API access pattern:
 * OrtApiBase* base = OrtGetApiBase();
 * OrtApi* api = base->GetApi(ORT_API_VERSION);
 * // Then call api->CreateEnv, api->CreateSession, etc. as function pointers
 */
public final class OnnxRuntimeBridge implements AutoCloseable {

    // ── ORT constants ────────────────────────────────────────────────────────
    private static final int ORT_API_VERSION = 18;
    private static final int ORT_LOGGING_LEVEL_WARNING = 2;

    // ── FFM plumbing ─────────────────────────────────────────────────────────
    private final Arena arena;
    private final Linker linker;
    private final SymbolLookup lookup;

    // ORT handle pointers (off-heap)
    MemorySegment pEnv; // OrtEnv*
    MemorySegment pSessionOptions;

    // Resolved OrtApi function-pointer method handles
    MethodHandle mhCreateEnv;
    MethodHandle mhCreateSessionOptions;
    MethodHandle mhCreateSession;
    MethodHandle mhCreateTensorWithDataAsOrtValue;
    MethodHandle mhRun;
    MethodHandle mhGetTensorMutableData;
    MethodHandle mhGetTensorTypeAndShape;
    MethodHandle mhGetDimensionsCount;
    MethodHandle mhGetDimensions;
    MethodHandle mhReleaseValue;
    MethodHandle mhReleaseSession;

    // Raw pointer to OrtApi struct (needed to resolve vtable slots)
    private MemorySegment pApi;

    // ── Construction ──────────────────────────────────────────────────────────

    public OnnxRuntimeBridge(Arena arena) throws Throwable {
        this.arena = arena;
        this.linker = Linker.nativeLinker();
        this.lookup = SymbolLookup.loaderLookup();

        initApi();
        createEnv();
        createSessionOptions();
    }

    // ── API initialisation ────────────────────────────────────────────────────

    /**
     * Call OrtGetApiBase(), then GetApi(ORT_API_VERSION) to get the
     * OrtApi vtable pointer. All subsequent ORT calls go through this vtable.
     */
    private void initApi() throws Throwable {
        // OrtApiBase* OrtGetApiBase(void)
        MethodHandle getApiBase = linker.downcallHandle(
                lookup.find("OrtGetApiBase").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS));

        MemorySegment apiBase = (MemorySegment) getApiBase.invoke();
        apiBase = apiBase.reinterpret(Long.MAX_VALUE); // unbounded for struct access

        // OrtApiBase.GetApi is the SECOND function pointer in the struct.
        // Slot 0: GetApi(uint32_t version) → OrtApi*
        // Slot 1: GetVersionString()
        // Each function pointer is ADDRESS_SIZE bytes wide.
        long ptrSize = ValueLayout.ADDRESS.byteSize(); // 8 on 64-bit

        MemorySegment fnGetApi = apiBase.get(ValueLayout.ADDRESS, 0);
        fnGetApi = fnGetApi.reinterpret(Long.MAX_VALUE);

        MethodHandle mhGetApi = linker.downcallHandle(
                fnGetApi,
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        pApi = (MemorySegment) mhGetApi.invoke(ORT_API_VERSION);
        pApi = pApi.reinterpret(Long.MAX_VALUE);

        // Resolve commonly used vtable slots.
        // OrtApi vtable layout (offset = slot_index × pointer_size):
        // 0: CreateStatus 1: GetErrorCode 2: GetErrorMessage
        // 3: CreateEnv 4: CreateEnvWithCustomLogger
        // 5: EnableTelemetryEvents ... (see onnxruntime_c_api.h)
        // We bind the slots we need. Offsets must match ORT 1.18 header exactly.
        mhCreateEnv = resolveVtableSlot(3,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, // logging level
                        ValueLayout.ADDRESS, // log id (const char*)
                        ValueLayout.ADDRESS)); // OrtEnv** out

        mhCreateSessionOptions = resolveVtableSlot(11,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS)); // OrtSessionOptions** out

        mhCreateSession = resolveVtableSlot(12,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, // OrtEnv*
                        ValueLayout.ADDRESS, // model path (wchar_t* on Win, char* on Unix)
                        ValueLayout.ADDRESS, // OrtSessionOptions*
                        ValueLayout.ADDRESS)); // OrtSession** out

        mhCreateTensorWithDataAsOrtValue = resolveVtableSlot(52,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, // OrtMemoryInfo*
                        ValueLayout.ADDRESS, // void* data
                        ValueLayout.JAVA_LONG, // size_t data_length
                        ValueLayout.ADDRESS, // int64_t* shape
                        ValueLayout.JAVA_LONG, // size_t shape_len
                        ValueLayout.JAVA_INT, // ONNXTensorElementDataType
                        ValueLayout.ADDRESS)); // OrtValue** out

        mhRun = resolveVtableSlot(13,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, // OrtSession*
                        ValueLayout.ADDRESS, // OrtRunOptions* (nullable)
                        ValueLayout.ADDRESS, // input names char**
                        ValueLayout.ADDRESS, // input OrtValue**
                        ValueLayout.JAVA_LONG, // input count
                        ValueLayout.ADDRESS, // output names char**
                        ValueLayout.JAVA_LONG, // output count
                        ValueLayout.ADDRESS)); // output OrtValue**

        mhGetTensorMutableData = resolveVtableSlot(55,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, // OrtValue*
                        ValueLayout.ADDRESS)); // void** out

        mhReleaseValue = resolveVtableSlot(62,
                FunctionDescriptor.ofVoid(
                        ValueLayout.ADDRESS)); // OrtValue*
    }

    /**
     * Bind a function-pointer slot from the OrtApi vtable.
     * slotIndex: zero-based index in the vtable (each slot = one pointer).
     */
    private MethodHandle resolveVtableSlot(int slotIndex, FunctionDescriptor fd) {
        long offset = (long) slotIndex * ValueLayout.ADDRESS.byteSize();
        MemorySegment fnPtr = pApi.get(ValueLayout.ADDRESS, offset);
        fnPtr = fnPtr.reinterpret(Long.MAX_VALUE);
        return linker.downcallHandle(fnPtr, fd);
    }

    // ── Session lifecycle helpers ─────────────────────────────────────────────

    private void createEnv() throws Throwable {
        MemorySegment logId = arena.allocateFrom("sd-java");
        // OrtEnv** double-pointer
        MemorySegment ppEnv = arena.allocate(ValueLayout.ADDRESS);

        int status = (int) mhCreateEnv.invoke(
                ORT_LOGGING_LEVEL_WARNING, logId, ppEnv);
        checkStatus(status, "CreateEnv");

        pEnv = ppEnv.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
    }

    private void createSessionOptions() throws Throwable {
        MemorySegment ppOpts = arena.allocate(ValueLayout.ADDRESS);
        int status = (int) mhCreateSessionOptions.invoke(ppOpts);
        checkStatus(status, "CreateSessionOptions");
        pSessionOptions = ppOpts.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
    }

    /**
     * Load an ONNX model and return an opaque session pointer.
     */
    public MemorySegment createSession(java.nio.file.Path modelPath) throws Throwable {
        MemorySegment pathSeg = arena.allocateFrom(modelPath.toString());
        MemorySegment ppSession = arena.allocate(ValueLayout.ADDRESS);

        int status = (int) mhCreateSession.invoke(
                pEnv, pathSeg, pSessionOptions, ppSession);
        checkStatus(status, "CreateSession: " + modelPath);

        return ppSession.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
    }

    // ── Tensor creation/destruction ───────────────────────────────────────────

    /**
     * Wrap an off-heap float array (MemorySegment) as an OrtValue tensor.
     * shape: e.g., {1, 77, 768}
     */
    public MemorySegment createFloatTensor(MemorySegment data, long[] shape) throws Throwable {
        // allocate shape array off-heap
        MemorySegment shapeSeg = arena.allocate(shape.length * Long.BYTES, 8);
        for (int i = 0; i < shape.length; i++) {
            shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
        }

        // CPU memory info (index 0 = OrtArenaAllocator, OrtMemTypeDefault)
        MemorySegment cpuMemInfo = getCpuMemoryInfo();

        MemorySegment ppValue = arena.allocate(ValueLayout.ADDRESS);
        long dataBytes = data.byteSize();
        int ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1;

        int status = (int) mhCreateTensorWithDataAsOrtValue.invoke(
                cpuMemInfo, data, dataBytes,
                shapeSeg, (long) shape.length,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ppValue);
        checkStatus(status, "CreateTensorWithData");

        return ppValue.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
    }

    public MemorySegment createInt64Tensor(MemorySegment data, long[] shape) throws Throwable {
        MemorySegment shapeSeg = arena.allocate(shape.length * Long.BYTES, 8);
        for (int i = 0; i < shape.length; i++) {
            shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
        }

        MemorySegment cpuMemInfo = getCpuMemoryInfo();
        MemorySegment ppValue = arena.allocate(ValueLayout.ADDRESS);
        long dataBytes = data.byteSize();
        int ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7;

        int status = (int) mhCreateTensorWithDataAsOrtValue.invoke(
                cpuMemInfo, data, dataBytes,
                shapeSeg, (long) shape.length,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                ppValue);
        checkStatus(status, "CreateTensorWithData");

        return ppValue.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
    }

    /**
     * Run an OrtSession. inputNames/outputNames are null-terminated C string
     * arrays.
     */
    public MemorySegment[] run(
            MemorySegment session,
            String[] inputNames,
            MemorySegment[] inputs,
            String[] outputNames) throws Throwable {

        int nIn = inputNames.length;
        int nOut = outputNames.length;

        // Build char** arrays for names
        MemorySegment pInNames = buildNameArray(inputNames);
        MemorySegment pOutNames = buildNameArray(outputNames);

        // OrtValue* input array
        MemorySegment pInputs = arena.allocate((long) nIn * ValueLayout.ADDRESS.byteSize(), 8);
        for (int i = 0; i < nIn; i++) {
            pInputs.setAtIndex(ValueLayout.ADDRESS, i, inputs[i]);
        }

        // OrtValue* output array (filled by ORT)
        MemorySegment pOutputs = arena.allocate((long) nOut * ValueLayout.ADDRESS.byteSize(), 8);

        int status = (int) mhRun.invoke(
                session, MemorySegment.NULL,
                pInNames, pInputs, (long) nIn,
                pOutNames, (long) nOut, pOutputs);
        checkStatus(status, "Run");

        MemorySegment[] out = new MemorySegment[nOut];
        for (int i = 0; i < nOut; i++) {
            out[i] = pOutputs.getAtIndex(ValueLayout.ADDRESS, i).reinterpret(Long.MAX_VALUE);
        }
        return out;
    }

    /** Extract raw float data pointer from an OrtValue. */
    public MemorySegment getTensorFloatData(MemorySegment ortValue, long expectedFloats) throws Throwable {
        MemorySegment ppData = arena.allocate(ValueLayout.ADDRESS);
        int status = (int) mhGetTensorMutableData.invoke(ortValue, ppData);
        checkStatus(status, "GetTensorMutableData");
        MemorySegment raw = ppData.get(ValueLayout.ADDRESS, 0);
        return raw.reinterpret(expectedFloats * Float.BYTES);
    }

    public void releaseValue(MemorySegment ortValue) throws Throwable {
        mhReleaseValue.invoke(ortValue);
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private MemorySegment cpuMemInfo = null;

    private MemorySegment getCpuMemoryInfo() throws Throwable {
        if (cpuMemInfo != null)
            return cpuMemInfo;

        // OrtApi->CreateCpuMemoryInfo(OrtArenaAllocator=0, OrtMemTypeDefault=0,
        // OrtMemoryInfo**)
        MethodHandle mhCreate = resolveVtableSlot(49,
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
        checkStatus((int) mhCreate.invoke(0, 0, pp), "CreateCpuMemoryInfo");
        cpuMemInfo = pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        return cpuMemInfo;
    }

    private MemorySegment buildNameArray(String[] names) {
        MemorySegment arr = arena.allocate((long) names.length * ValueLayout.ADDRESS.byteSize(), 8);
        for (int i = 0; i < names.length; i++) {
            arr.setAtIndex(ValueLayout.ADDRESS, i, arena.allocateFrom(names[i]));
        }
        return arr;
    }

    private void checkStatus(int statusCode, String op) {
        if (statusCode != 0) {
            throw new RuntimeException("ORT error " + statusCode + " in: " + op);
        }
    }

    @Override
    public void close() {
        // OrtEnv and session options are owned by the arena; it handles cleanup.
    }
}
