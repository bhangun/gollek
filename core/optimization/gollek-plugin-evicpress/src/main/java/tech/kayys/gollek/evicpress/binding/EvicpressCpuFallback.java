package tech.kayys.gollek.evicpress.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for EVICPRESS kernels.
 *
 * <p>Scoring uses recency only (linear ramp). Compression is a no-op
 * (cannot reduce memory in pure Java without native quantization kernel).
 * Route protection marks the last 20 % of blocks as protected.
 */
public final class EvicpressCpuFallback {

    private static final Logger LOG = Logger.getLogger(EvicpressCpuFallback.class);

    private EvicpressCpuFallback() {}

    /** Recency-based importance: block i scores i/numBlocks (later = more important). */
    public static int score(
            MemorySegment scores,
            MemorySegment attnWeights,
            int numBlocks, int blockSize, int numLayers, int gvoteSamples) {

        LOG.debug("EVICPRESS CPU fallback score (native unavailable) — recency heuristic");
        for (int i = 0; i < numBlocks; i++) {
            float recency = (float) i / numBlocks;
            // Also incorporate raw attention sum if available
            float attnSum = 0f;
            for (int t = 0; t < blockSize && (long)(i * blockSize + t) < attnWeights.byteSize() / 4; t++) {
                attnSum += attnWeights.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(i * blockSize + t));
            }
            scores.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    0.5f * recency + 0.5f * Math.min(attnSum / blockSize, 1.0f));
        }
        return 0;
    }

    /** No-op: cannot reduce VRAM from Java — logs the intent only. */
    public static int compress(MemorySegment blockData, long bytes, String dtype) {
        LOG.debugf("EVICPRESS CPU fallback compress (no-op) — would quantize to %s", dtype);
        return 0;
    }

    /** Protect last 20% of blocks (most recent context). */
    public static int routeProtect(
            MemorySegment protectedMask,
            MemorySegment attnWeights,
            int numBlocks, int blockSize, int numLayers) {

        int protectFrom = (int)(numBlocks * 0.8);
        long maskWords  = (numBlocks + 63) / 64;
        for (long w = 0; w < maskWords; w++) {
            protectedMask.setAtIndex(ValueLayout.JAVA_LONG, w, 0L);
        }
        for (int b = protectFrom; b < numBlocks; b++) {
            long word = b / 64;
            long bit  = b % 64;
            long cur  = protectedMask.getAtIndex(ValueLayout.JAVA_LONG, word);
            protectedMask.setAtIndex(ValueLayout.JAVA_LONG, word, cur | (1L << bit));
        }
        return 0;
    }
}
