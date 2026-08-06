package tech.kayys.gollek.weightoffload.binding;

import java.lang.foreign.MemorySegment;

/**
 * Pure-Java CPU fallback for {@link WeightOffloadBinding}.
 *
 * <p>Used when {@code libgollek_offload.so} is not present (CI, non-CUDA machines).
 * {@code h2dAsync} becomes a synchronous {@link MemorySegment#copyFrom};
 * all GPU-specific operations (vramUtil, stallCount, memAdvise) return benign defaults.
 */
public final class WeightOffloadCpuFallback {

    private WeightOffloadCpuFallback() {}

    public static int h2dAsync(MemorySegment dst, MemorySegment src, long bytes, int streamId) {
        long copyLen = Math.min(Math.min(src.byteSize(), dst.byteSize()), bytes);
        MemorySegment.copy(src, 0L, dst, 0L, copyLen);
        return 0;
    }

    public static int streamSync(int streamId) { return 0; }

    public static int memAdvise(MemorySegment ptr, long bytes, int advice, int deviceId) {
        return 0; // no-op — OS handles paging
    }

    public static float vramUtil(int deviceId) { return 0.0f; }

    public static long stallCount(int streamId) { return 0L; }
}
