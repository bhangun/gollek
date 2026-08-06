package tech.kayys.gollek.tensorrt.binding;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for {@link TensorRtBinding}.
 *
 * <p>Used when {@code libnvinfer_10.so} is absent (CI, non-NVIDIA machines).
 * All inference calls return zeroed output tensors.
 */
public final class TensorRtCpuFallback {

    private static final Logger LOG = Logger.getLogger(TensorRtCpuFallback.class);

    private TensorRtCpuFallback() {}

    /** Returns a zeroed float array of {@code numFloats} elements. */
    public static float[] run(int numOutputFloats) {
        LOG.debug("TensorRT CPU fallback (native unavailable) — returning zeros");
        return new float[numOutputFloats];
    }
}
