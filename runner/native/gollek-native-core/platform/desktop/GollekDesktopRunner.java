/**
 * GollekDesktopRunner.java
 * Desktop / Server JVM runner — replaces LiteRTCpuRunner.java + LiteRTNativeBindings.java
 *
 * Loads libgollek_core.so (Linux), libgollek_core.dylib (macOS), or
 * gollek_core.dll (Windows) via JNI.  No more Java 21 FFM / Project Panama
 * boilerplate — the C++ core handles all TFLite API calls internally.
 *
 * Drop-in replacement: implement the same RunnerPlugin SPI used by
 * LiteRTRunnerPlugin so the rest of the Gollek server code is unchanged.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

package tech.kayys.gollek.runner.litert;

import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.*;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Native binding constants (mirror gollek_engine.h enums)
 * ═══════════════════════════════════════════════════════════════════════════ */

final class GollekStatus {
    static final int OK = 0;
    static final int ERROR = 1;
    static final int ERROR_MODEL_LOAD = 2;
    static final int ERROR_ALLOC_TENSORS = 3;
    static final int ERROR_INVOKE = 4;
    static final int ERROR_INVALID_ARG = 5;
    static final int ERROR_NOT_INITIALIZED = 6;
    static final int ERROR_DELEGATE_FAILED = 7;

    static String name(int code) {
        return switch (code) {
            case OK -> "OK";
            case ERROR -> "Generic error";
            case ERROR_MODEL_LOAD -> "Model load failed";
            case ERROR_ALLOC_TENSORS -> "Tensor allocation failed";
            case ERROR_INVOKE -> "Invoke failed";
            case ERROR_INVALID_ARG -> "Invalid argument";
            case ERROR_NOT_INITIALIZED -> "Not initialized";
            case ERROR_DELEGATE_FAILED -> "Delegate failed";
            default -> "Unknown(" + code + ")";
        };
    }
}

public enum GollekDelegate {
    NONE(0), CPU(1), GPU(2), NNAPI(3), HEXAGON(4), COREML(5), AUTO(6);

    final int value;

    GollekDelegate(int v) {
        this.value = v;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * 2. Low-level JNI bridge (internal)
 * ═══════════════════════════════════════════════════════════════════════════
 */

final class GollekNativeLib {

    private static volatile boolean loaded = false;

    static void ensureLoaded(Path libPath) {
        if (loaded)
            return;
        synchronized (GollekNativeLib.class) {
            if (loaded)
                return;
            if (libPath != null) {
                System.load(libPath.toAbsolutePath().toString());
            } else {
                System.loadLibrary("gollek_core");
            }
            loaded = true;
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────
    static native long nativeCreate(int numThreads, int delegate,
            boolean enableXnnpack, boolean useMemoryPool);

    static native void nativeDestroy(long handle);

    // ── Model loading ──────────────────────────────────────────────────────
    static native int nativeLoadModelFromFile(long handle, String path);

    static native int nativeLoadModelFromBuffer(long handle, ByteBuffer buffer);

    // ── Introspection ──────────────────────────────────────────────────────
    static native int nativeGetInputCount(long handle);

    static native int nativeGetOutputCount(long handle);

    /** Returns long[] = { type, numDims, dim0..dimN, byteSize } */
    static native long[] nativeGetInputInfo(long handle, int index);

    static native long[] nativeGetOutputInfo(long handle, int index);

    // ── Inference ──────────────────────────────────────────────────────────
    static native int nativeSetInput(long handle, int index,
            ByteBuffer buf, int bytes);

    static native int nativeInvoke(long handle);

    static native int nativeGetOutput(long handle, int index,
            ByteBuffer buf, int bytes);

    // ── Diagnostics ────────────────────────────────────────────────────────
    static native String nativeLastError(long handle);

    static native String nativeVersion();
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * 3. Tensor descriptor (mirrors GollekTensorInfo)
 * ═══════════════════════════════════════════════════════════════════════════
 */

public record TensorMeta(int type, int[] dims, long byteSize) {

    /** Number of elements (product of all dims). */
    public long elementCount() {
        long n = 1;
        for (int d : dims)
            n *= d;
        return n;
    }

    static TensorMeta fromRaw(long[] raw) {
        if (raw == null || raw.length < 3)
            return null;
        int type = (int) raw[0];
        int numDims = (int) raw[1];
        int[] dims = new int[numDims];
        for (int i = 0; i < numDims; i++)
            dims[i] = (int) raw[2 + i];
        long byteSize = raw[2 + numDims];
        return new TensorMeta(type, dims, byteSize);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * 4. Public GollekEngine — resource-safe handle
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * High-level wrapper around the native Gollek C++ core.
 * Implements AutoCloseable — use in try-with-resources.
 *
 * <pre>{@code
 * try (GollekEngine engine = new GollekEngine()) {
 *     engine.loadModelFromFile("/models/mobilenet_v2.litertlm");
 *     ByteBuffer output = ByteBuffer.allocateDirect(1001);
 *     engine.infer(inputBuffer, output);
 * }
 * }</pre>
 */
public class GollekEngine implements AutoCloseable {

    private final long handle;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public GollekEngine() {
        this(4, GollekDelegate.AUTO, true, true, null);
    }

    public GollekEngine(int numThreads, GollekDelegate delegate,
            boolean enableXnnpack, boolean useMemoryPool,
            Path libPath) {
        GollekNativeLib.ensureLoaded(libPath);
        this.handle = GollekNativeLib.nativeCreate(
                numThreads, delegate.value, enableXnnpack, useMemoryPool);
        if (this.handle == 0)
            throw new IllegalStateException("gollek_engine_create returned null");
    }

    /* ── Model loading ──────────────────────────────────────────────── */

    public void loadModelFromFile(String path) {
        checkOpen();
        int s = GollekNativeLib.nativeLoadModelFromFile(handle, path);
        checkStatus(s, "loadModelFromFile");
    }

    public void loadModelFromBuffer(ByteBuffer buffer) {
        checkOpen();
        if (!buffer.isDirect())
            throw new IllegalArgumentException("Buffer must be direct");
        int s = GollekNativeLib.nativeLoadModelFromBuffer(handle, buffer);
        checkStatus(s, "loadModelFromBuffer");
    }

    /* ── Introspection ──────────────────────────────────────────────── */

    public int inputCount() {
        checkOpen();
        return GollekNativeLib.nativeGetInputCount(handle);
    }

    public int outputCount() {
        checkOpen();
        return GollekNativeLib.nativeGetOutputCount(handle);
    }

    public TensorMeta inputMeta(int index) {
        checkOpen();
        return TensorMeta.fromRaw(GollekNativeLib.nativeGetInputInfo(handle, index));
    }

    public TensorMeta outputMeta(int index) {
        checkOpen();
        return TensorMeta.fromRaw(GollekNativeLib.nativeGetOutputInfo(handle, index));
    }

    /* ── Inference ──────────────────────────────────────────────────── */

    public void setInput(int index, ByteBuffer buf) {
        checkOpen();
        if (!buf.isDirect())
            throw new IllegalArgumentException("Buffer must be direct");
        int s = GollekNativeLib.nativeSetInput(handle, index, buf, buf.remaining());
        checkStatus(s, "setInput[" + index + "]");
    }

    public void invoke() {
        checkOpen();
        checkStatus(GollekNativeLib.nativeInvoke(handle), "invoke");
    }

    public void getOutput(int index, ByteBuffer dst) {
        checkOpen();
        if (!dst.isDirect())
            throw new IllegalArgumentException("Buffer must be direct");
        int s = GollekNativeLib.nativeGetOutput(handle, index, dst, dst.remaining());
        checkStatus(s, "getOutput[" + index + "]");
    }

    /**
     * Single-shot convenience: set input 0 → invoke → fill output 0.
     * Both buffers must be direct ByteBuffers.
     */
    public void infer(ByteBuffer input, ByteBuffer output) {
        setInput(0, input);
        invoke();
        getOutput(0, output);
    }

    /* ── Diagnostics ────────────────────────────────────────────────── */

    public String lastError() {
        return GollekNativeLib.nativeLastError(handle);
    }

    public String version() {
        return GollekNativeLib.nativeVersion();
    }

    /* ── Lifecycle ──────────────────────────────────────────────────── */

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            GollekNativeLib.nativeDestroy(handle);
        }
    }

    /* ── Helpers ────────────────────────────────────────────────────── */

    private void checkOpen() {
        if (closed.get())
            throw new IllegalStateException("GollekEngine already closed");
    }

    private void checkStatus(int s, String op) {
        if (s != GollekStatus.OK) {
            throw new GollekException(s, op + " failed: " +
                    GollekStatus.name(s) + " — " + lastError());
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * 5. Exception type
 * ═══════════════════════════════════════════════════════════════════════════
 */

class GollekException extends RuntimeException {
    final int statusCode;

    GollekException(int code, String message) {
        super(message);
        this.statusCode = code;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * 6. RunnerPlugin implementation — drop-in for LiteRTRunnerPlugin
 * Wires GollekEngine into the existing Gollek server plugin SPI.
 * ═══════════════════════════════════════════════════════════════════════════
 */

public class GollekLiteRTRunner implements RunnerPlugin {

    private static final Logger LOG = Logger.getLogger(GollekLiteRTRunner.class);

    public static final String ID = "gollek-litert-runner";

    private GollekEngine engine;
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "Gollek LiteRT Runner (C++ Core)";
    }

    @Override
    public String version() {
        return "2.0.0-cpp";
    }

    @Override
    public String description() {
        return "TFLite inference via Gollek C++ native core";
    }

    @Override
    public String format() {
        return "litert";
    }

    @Override
    public void initialize(RunnerContext context) throws RunnerException {
        try {
            int threads = context.getConfig().getIntOrDefault("num-threads", 4);
            String delStr = context.getConfig().getStringOrDefault("delegate", "AUTO");
            GollekDelegate del = GollekDelegate.valueOf(delStr.toUpperCase());

            engine = new GollekEngine(threads, del, true, true, null);
            initialized = true;
            LOG.infof("GollekLiteRTRunner initialized — engine version: %s", engine.version());
        } catch (Exception e) {
            throw new RunnerInitializationException(ID, "Failed to create native engine: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean isAvailable() {
        try {
            GollekNativeLib.ensureLoaded(null);
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    @Override
    public RunnerHealth health() {
        if (!initialized || engine == null)
            return RunnerHealth.unhealthy("Not initialized");
        return RunnerHealth.healthy(Map.of("version", engine.version()));
    }

    @Override
    public Set<String> supportedFormats() {
        return Set.of(".litertlm", ".tfl");
    }

    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("mobilenet", "efficientnet", "bert", "yolo");
    }

    @Override
    public Set<RequestType> supportedRequestTypes() {
        return Set.of(RequestType.INFER, RequestType.CLASSIFY);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> RunnerResult<T> execute(RunnerRequest request, RunnerContext context)
            throws RunnerException {
        if (!initialized)
            throw new RunnerExecutionException(ID, request.getType().name(), "Not initialized");
        try {
            // 1. Load model if not yet loaded (model path from request)
            String modelPath = request.getModelPath();
            engine.loadModelFromFile(modelPath);

            // 2. Build input buffer from request TensorData
            TensorMeta inMeta = engine.inputMeta(0);
            ByteBuffer input = buildInputBuffer(request, inMeta);

            // 3. Allocate output buffer
            TensorMeta outMeta = engine.outputMeta(0);
            ByteBuffer output = ByteBuffer.allocateDirect((int) outMeta.byteSize());

            // 4. Run
            engine.infer(input, output);

            // 5. Wrap result
            return (RunnerResult<T>) RunnerResult.success(Map.of(
                    "status", "success",
                    "outputBytes", outMeta.byteSize(),
                    "engine", engine.version()));
        } catch (GollekException e) {
            throw new RunnerExecutionException(ID, request.getType().name(), e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        initialized = false;
        if (engine != null) {
            engine.close();
            engine = null;
        }
        LOG.info("GollekLiteRTRunner shutdown");
    }

    @Override
    public RunnerValidationResult validate() {
        return isAvailable()
                ? RunnerValidationResult.builder().valid(true).build()
                : RunnerValidationResult.invalid(List.of("libgollek_core not found on library path"));
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
                "format", "litert",
                "backend", "Gollek C++ Core",
                "version", engine != null ? engine.version() : "not loaded");
    }

    // ── Helper ──────────────────────────────────────────────────────────────

    private static ByteBuffer buildInputBuffer(RunnerRequest request, TensorMeta meta) {
        ByteBuffer buf = ByteBuffer.allocateDirect((int) meta.byteSize());
        // TODO: copy from request.getInputs() TensorData using LiteRTTensorUtils logic
        // For now, zeroed buffer (smoke-test placeholder)
        return buf;
    }
}
