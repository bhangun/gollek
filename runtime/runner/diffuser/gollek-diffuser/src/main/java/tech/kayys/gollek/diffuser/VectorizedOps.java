package tech.kayys.gollek.diffuser;
import java.nio.ByteOrder;

import jdk.incubator.vector.*;

import java.lang.foreign.*;

/**
 * SIMD-accelerated operations using the JDK Vector API (incubator).
 *
 * Key operations:
 * - classifierFreeGuidance : fused multiply-add across latent tensors
 * - postProcess : normalize float[-1,1] → clamp float[0,1] pixel output
 *
 * The Vector API auto-selects the widest available SIMD lane count at runtime
 * (SSE2 → AVX2 → AVX-512 → SVE on ARM), so this code is portable yet optimal.
 */
public final class VectorizedOps {

    // Preferred species: widest float vector the hardware supports
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private VectorizedOps() {
    }

    // ── Classifier-Free Guidance ──────────────────────────────────────────────

    /**
     * Computes element-wise:
     * guided[i] = uncond[i] + scale * (cond[i] - uncond[i])
     *
     * Runs entirely on off-heap MemorySegments via the Vector API's
     * fromMemorySegment / intoMemorySegment methods — zero heap copies.
     *
     * @param uncond ORT output tensor (float[], off-heap)
     * @param cond   ORT output tensor (float[], off-heap)
     * @param scale  guidance scale (e.g. 7.5)
     * @param arena  arena for the output allocation
     */
    public static MemorySegment classifierFreeGuidance(
            MemorySegment uncond,
            MemorySegment cond,
            float scale,
            Arena arena) {

        long byteLen = uncond.byteSize();
        int nFloats = (int) (byteLen / Float.BYTES);

        MemorySegment output = arena.allocate(byteLen, 8);

        FloatVector vScale = FloatVector.broadcast(SPECIES, scale);

        int i = 0;
        int bound = SPECIES.loopBound(nFloats);

        // Vectorized loop: processes SPECIES.length() floats per iteration
        for (; i < bound; i += SPECIES.length()) {
            FloatVector vu = FloatVector.fromMemorySegment(SPECIES, uncond, (long) i * Float.BYTES,
                    ByteOrder.nativeOrder());
            FloatVector vc = FloatVector.fromMemorySegment(SPECIES, cond, (long) i * Float.BYTES,
                    ByteOrder.nativeOrder());

            // guided = uncond + scale * (cond - uncond) → FMA: cond*scale +
            // uncond*(1-scale)
            FloatVector diff = vc.sub(vu);
            FloatVector guided = vu.add(diff.mul(vScale)); // fused on AVX-512

            guided.intoMemorySegment(output, (long) i * Float.BYTES, ByteOrder.nativeOrder());
        }

        // Scalar tail for lengths not divisible by SPECIES.length()
        for (; i < nFloats; i++) {
            float u = uncond.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float c = cond.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            output.setAtIndex(ValueLayout.JAVA_FLOAT, i, u + scale * (c - u));
        }

        return output;
    }

    // ── Post-Processing ───────────────────────────────────────────────────────

    /**
     * Converts VAE decoder output (float[-1,1]) to normalized pixel values
     * (float[0,1]).
     *
     * Formula: pixel = clamp((x + 1) / 2, 0, 1)
     *
     * Output layout: interleaved RGBA, length = width × height × 4.
     * (Alpha channel is set to 1.0 by this method.)
     *
     * @param decoded raw VAE output MemorySegment, shape [1, 3, H, W]
     * @param w       image width
     * @param h       image height
     * @return heap float[] for easy downstream use (PNG encoding etc.)
     */
    public static float[] postProcess(MemorySegment decoded, int w, int h) {
        int pixelCount = w * h;
        int channelSize = pixelCount; // per channel element count
        int nIn = 3 * channelSize; // RGB channels from VAE

        float[] rgba = new float[pixelCount * 4];

        FloatVector vHalf = FloatVector.broadcast(SPECIES, 0.5f);
        FloatVector vOne = FloatVector.broadcast(SPECIES, 1.0f);
        FloatVector vZero = FloatVector.broadcast(SPECIES, 0.0f);

        // Process R, G, B channels, output interleaved RGBA
        for (int ch = 0; ch < 3; ch++) {
            int inOffset = ch * channelSize;
            int bound = SPECIES.loopBound(channelSize);
            int p = 0;

            for (; p < bound; p += SPECIES.length()) {
                FloatVector v = FloatVector.fromMemorySegment(SPECIES,
                        decoded, (long) (inOffset + p) * Float.BYTES, ByteOrder.nativeOrder());

                // normalize: (x + 1) / 2
                FloatVector norm = v.add(vOne).mul(vHalf);

                // clamp [0, 1]
                FloatVector clamped = norm.max(vZero).min(vOne);

                // Scatter into RGBA heap array (stride = 4 per pixel)
                float[] lane = clamped.toArray();
                for (int k = 0; k < lane.length; k++) {
                    rgba[(p + k) * 4 + ch] = lane[k];
                }
            }

            // Scalar tail
            for (; p < channelSize; p++) {
                float x = decoded.getAtIndex(ValueLayout.JAVA_FLOAT, inOffset + p);
                rgba[p * 4 + ch] = Math.clamp((x + 1f) / 2f, 0f, 1f);
            }
        }

        // Fill alpha channel = 1.0
        for (int p = 0; p < pixelCount; p++) {
            rgba[p * 4 + 3] = 1.0f;
        }

        return rgba;
    }

    // ── Scheduler helpers ─────────────────────────────────────────────────────

    /**
     * Element-wise addition of two off-heap float segments.
     * Used by the DDIM scheduler step.
     */
    public static MemorySegment add(MemorySegment a, MemorySegment b, Arena arena) {
        int n = (int) (a.byteSize() / Float.BYTES);
        MemorySegment out = arena.allocate(a.byteSize(), 8);

        int bound = SPECIES.loopBound(n);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a, (long) i * Float.BYTES, ByteOrder.nativeOrder());
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b, (long) i * Float.BYTES, ByteOrder.nativeOrder());
            va.add(vb).intoMemorySegment(out, (long) i * Float.BYTES, ByteOrder.nativeOrder());
        }
        for (; i < n; i++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    a.getAtIndex(ValueLayout.JAVA_FLOAT, i) +
                            b.getAtIndex(ValueLayout.JAVA_FLOAT, i));
        }
        return out;
    }

    /**
     * Element-wise scalar multiply of an off-heap float segment.
     */
    public static MemorySegment scale(MemorySegment src, float scalar, Arena arena) {
        int n = (int) (src.byteSize() / Float.BYTES);
        MemorySegment out = arena.allocate(src.byteSize(), 8);
        FloatVector vs = FloatVector.broadcast(SPECIES, scalar);

        int bound = SPECIES.loopBound(n);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, src, (long) i * Float.BYTES,
                    ByteOrder.nativeOrder());
            v.mul(vs).intoMemorySegment(out, (long) i * Float.BYTES, ByteOrder.nativeOrder());
        }
        for (; i < n; i++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i,
                    src.getAtIndex(ValueLayout.JAVA_FLOAT, i) * scalar);
        }
        return out;
    }
}
