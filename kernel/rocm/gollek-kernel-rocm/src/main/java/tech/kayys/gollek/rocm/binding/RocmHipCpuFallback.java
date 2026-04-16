package tech.kayys.gollek.rocm.binding;

import java.lang.foreign.MemorySegment;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for {@link RocmHipBinding}.
 *
 * <p>Memory operations fall back to {@link MemorySegment} copies.
 * Kernel launches are no-ops. Used for CI and development without ROCm.
 */
public final class RocmHipCpuFallback {

    private static final Logger LOG = Logger.getLogger(RocmHipCpuFallback.class);

    private RocmHipCpuFallback() {}

    public static int memcpy(MemorySegment dst, MemorySegment src, long bytes, int kind) {
        long len = Math.min(Math.min(dst.byteSize(), src.byteSize()), bytes);
        MemorySegment.copy(src, 0L, dst, 0L, len);
        return 0;
    }

    public static float[] runFallback(int numOutputFloats) {
        LOG.debug("ROCm CPU fallback (native unavailable) — returning zeros");
        return new float[numOutputFloats];
    }
}
