package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Precomputed RoPE (Rotary Positional Embedding) table.
 * Caches sine and cosine values to avoid expensive trigonometric functions during the forward pass.
 */
public final class RoPECache {

    public final MemorySegment cos;
    public final MemorySegment sin;

    public final int maxSeq;
    public final int dim;

    private RoPECache(MemorySegment cos, MemorySegment sin, int maxSeq, int dim) {
        this.cos = cos;
        this.sin = sin;
        this.maxSeq = maxSeq;
        this.dim = dim;
    }

    public static RoPECache build(Arena arena, int maxSeq, int dim, float base) {
        System.out.print("Building RoPE cache (" + maxSeq + "x" + dim + ")... ");
        System.out.flush();
        int half = dim / 2;
        MemorySegment sin = arena.allocate((long) maxSeq * half * Float.BYTES, 64);
        MemorySegment cos = arena.allocate((long) maxSeq * half * Float.BYTES, 64);
        
        float[] invFreq = new float[half];
        for (int i = 0; i < half; i++) {
            invFreq[i] = (float) (1.0 / Math.pow(base, (2.0 * i) / (double) dim));
        }

        for (int pos = 0; pos < maxSeq; pos++) {
            for (int i = 0; i < half; i++) {
                float theta = pos * invFreq[i];
                sin.set(ValueLayout.JAVA_FLOAT, ((long) pos * half + i) * 4, (float) Math.sin(theta));
                cos.set(ValueLayout.JAVA_FLOAT, ((long) pos * half + i) * 4, (float) Math.cos(theta));
            }
        }
        System.out.println("Done.");
        return new RoPECache(cos, sin, maxSeq, dim);
    }
}
