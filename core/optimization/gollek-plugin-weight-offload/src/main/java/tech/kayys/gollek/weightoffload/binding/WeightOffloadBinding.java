package tech.kayys.gollek.weightoffload.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the weight offloading CUDA helper library.
 *
 * <p>Loads {@code libgollek_offload.so} at runtime.
 *
 * <h2>C ABI exposed by libgollek_offload.so</h2>
 * <pre>
 * // Async host→device copy using a CUDA stream
 * int gollek_offload_h2d_async(
 *     void*        dst_device,   // device pointer
 *     const void*  src_host,     // pinned host pointer
 *     size_t       bytes,
 *     int          stream_id     // CUDA stream index (0 = default)
 * );
 *
 * // Synchronise a stream (block until h2d_async completes)
 * int gollek_offload_stream_sync(int stream_id);
 *
 * // Advise CUDA driver on memory usage (cuMemAdvise)
 * // advice: 0=prefetch_device, 1=prefetch_host, 2=read_mostly, 3=preferred_location
 * int gollek_offload_mem_advise(
 *     const void* ptr,
 *     size_t      bytes,
 *     int         advice,
 *     int         device_id
 * );
 *
 * // Query GPU VRAM utilisation [0.0, 1.0]
 * float gollek_offload_vram_util(int device_id);
 *
 * // Count of pending h2d stalls (GPU waited > 1ms for transfer)
 * long gollek_offload_stall_count(int stream_id);
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/offload   # requires CUDA 12.x
 * </pre>
 */
public class WeightOffloadBinding {

    private static final Logger LOG = Logger.getLogger(WeightOffloadBinding.class);
    private static volatile WeightOffloadBinding instance;

    private static final String FN_H2D_ASYNC    = "gollek_offload_h2d_async";
    private static final String FN_STREAM_SYNC  = "gollek_offload_stream_sync";
    private static final String FN_MEM_ADVISE   = "gollek_offload_mem_advise";
    private static final String FN_VRAM_UTIL    = "gollek_offload_vram_util";
    private static final String FN_STALL_COUNT  = "gollek_offload_stall_count";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private WeightOffloadBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new WeightOffloadBinding(lk);
            LOG.infof("WeightOffload native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load WeightOffload native library from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new WeightOffloadBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new WeightOffloadBinding(null);
        LOG.info("WeightOffload initialized in CPU fallback mode");
    }

    public static WeightOffloadBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("WeightOffloadBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Operations ────────────────────────────────────────────────────────────

    /**
     * Asynchronously copy {@code bytes} from pinned host memory to device memory.
     *
     * <p>On Apple Silicon (unified memory) this is a no-op because host and
     * device share the same physical DRAM. The Metal path uses
     * {@link tech.kayys.gollek.extension.metal.binding.MetalBinding} instead.
     *
     * @param dstDevice  device pointer (MemorySegment backed by GPU memory)
     * @param srcHost    pinned host pointer (Arena.ofShared() segment)
     * @param bytes      number of bytes to transfer
     * @param streamId   CUDA stream index (0 = default stream)
     * @return 0 on success, CUDA error code on failure
     */
    public int h2dAsync(MemorySegment dstDevice, MemorySegment srcHost,
                        long bytes, int streamId) {
        if (!nativeAvailable) {
            return WeightOffloadCpuFallback.h2dAsync(dstDevice, srcHost, bytes, streamId);
        }
        MethodHandle mh = methodHandles.get(FN_H2D_ASYNC);
        if (mh == null) throw new IllegalStateException(FN_H2D_ASYNC + " not bound");
        try {
            return (int) mh.invokeExact(dstDevice, srcHost, bytes, streamId);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_H2D_ASYNC, e);
        }
    }

    /**
     * Block the calling thread until all operations on {@code streamId} complete.
     */
    public int streamSync(int streamId) {
        if (!nativeAvailable) return WeightOffloadCpuFallback.streamSync(streamId);
        MethodHandle mh = methodHandles.get(FN_STREAM_SYNC);
        if (mh == null) return 0;
        try { return (int) mh.invokeExact(streamId); }
        catch (Throwable e) { throw new RuntimeException("Failed to invoke " + FN_STREAM_SYNC, e); }
    }

    /**
     * Advise the CUDA driver on memory locality (cuMemAdvise).
     *
     * @param ptr      host-side pointer to the memory region
     * @param bytes    region size
     * @param advice   0=prefetch_device, 1=prefetch_host, 2=read_mostly, 3=preferred_location
     * @param deviceId target GPU device index
     */
    public int memAdvise(MemorySegment ptr, long bytes, int advice, int deviceId) {
        if (!nativeAvailable) return WeightOffloadCpuFallback.memAdvise(ptr, bytes, advice, deviceId);
        MethodHandle mh = methodHandles.get(FN_MEM_ADVISE);
        if (mh == null) return 0;
        try { return (int) mh.invokeExact(ptr, bytes, advice, deviceId); }
        catch (Throwable e) { throw new RuntimeException("Failed to invoke " + FN_MEM_ADVISE, e); }
    }

    /**
     * Query GPU VRAM utilisation in [0.0, 1.0].
     */
    public float vramUtil(int deviceId) {
        if (!nativeAvailable) return WeightOffloadCpuFallback.vramUtil(deviceId);
        MethodHandle mh = methodHandles.get(FN_VRAM_UTIL);
        if (mh == null) return 0.0f;
        try { return (float) mh.invokeExact(deviceId); }
        catch (Throwable e) { return 0.0f; }
    }

    /**
     * Number of times the GPU stalled waiting for a host→device transfer.
     * Used by the adaptive depth tuner.
     */
    public long stallCount(int streamId) {
        if (!nativeAvailable) return WeightOffloadCpuFallback.stallCount(streamId);
        MethodHandle mh = methodHandles.get(FN_STALL_COUNT);
        if (mh == null) return 0L;
        try { return (long) mh.invokeExact(streamId); }
        catch (Throwable e) { return 0L; }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_offload_h2d_async(void*, void*, size_t, int) -> int
        bind(FN_H2D_ASYNC, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // dst_device
                ValueLayout.ADDRESS,    // src_host
                ValueLayout.JAVA_LONG,  // bytes
                ValueLayout.JAVA_INT    // stream_id
        ));

        // int gollek_offload_stream_sync(int) -> int
        bind(FN_STREAM_SYNC, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT    // stream_id
        ));

        // int gollek_offload_mem_advise(void*, size_t, int, int) -> int
        bind(FN_MEM_ADVISE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // ptr
                ValueLayout.JAVA_LONG,  // bytes
                ValueLayout.JAVA_INT,   // advice
                ValueLayout.JAVA_INT    // device_id
        ));

        // float gollek_offload_vram_util(int) -> float
        bind(FN_VRAM_UTIL, FunctionDescriptor.of(
                ValueLayout.JAVA_FLOAT,
                ValueLayout.JAVA_INT    // device_id
        ));

        // long gollek_offload_stall_count(int) -> long
        bind(FN_STALL_COUNT, FunctionDescriptor.of(
                ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_INT    // stream_id
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("Bound native symbol: %s", name);
        } else {
            LOG.warnf("Native symbol not found: %s (some features may be unavailable)", name);
        }
    }

    static void reset() { instance = null; }
}
