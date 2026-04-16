package tech.kayys.gollek.onnx.binding;

import java.lang.foreign.MemorySegment;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for {@link OnnxRuntimeBinding}.
 *
 * <p>
 * All inference calls return a zeroed output tensor. Used for CI and
 * development on machines without ONNX Runtime installed.
 */
public final class OnnxRuntimeCpuFallback {

    private static final Logger LOG = Logger.getLogger(OnnxRuntimeCpuFallback.class);

    private OnnxRuntimeCpuFallback() {
    }

    /** Returns a zeroed float array of {@code numFloats} elements. */
    public static float[] run(MemorySegment inputData, int numOutputFloats) {
        LOG.debug("OnnxRuntime CPU fallback (native unavailable) — returning zeros");
        return new float[numOutputFloats];
    }
}
