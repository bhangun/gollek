package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Simple QKV prepacker: converts a weight matrix [hidden, 3*hidden]
 * into packed layout [3][numHeads][headDim][hidden] for efficient per-head dot products.
 * <p>
 * This is a prototype helper to be used by the native runner to prepack model weights
 * after GGUF / SafeTensor loading. It operates on raw MemorySegment handles (FFM).
 */
public final class QKVPrepacker {

    private QKVPrepacker() {}

    public static MemorySegment prepack(
            MemorySegment wqkv, // [hidden, 3*hidden] row-major contiguous
            int hidden,
            int numHeads,
            int headDim
    ) {
        int totalOut = numHeads * headDim;
        long packedElements = (long) 3 * numHeads * headDim * hidden;
        long bytes = packedElements * Float.BYTES;
        MemorySegment packed = MemorySegment.allocateNative(bytes);

        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < totalOut; j++) {
                int head = j / headDim;
                int dim = j % headDim;

                long baseSrc = ((long)i * 3 * totalOut + j) * Float.BYTES;

                float wq = wqkv.get(ValueLayout.JAVA_FLOAT, baseSrc);
                float wk = wqkv.get(ValueLayout.JAVA_FLOAT, baseSrc + (long) totalOut * Float.BYTES);
                float wv = wqkv.get(ValueLayout.JAVA_FLOAT, baseSrc + 2L * (long) totalOut * Float.BYTES);

                long dstQ = index(0, head, dim, i, numHeads, headDim, hidden);
                long dstK = index(1, head, dim, i, numHeads, headDim, hidden);
                long dstV = index(2, head, dim, i, numHeads, headDim, hidden);

                packed.set(ValueLayout.JAVA_FLOAT, dstQ * Float.BYTES, wq);
                packed.set(ValueLayout.JAVA_FLOAT, dstK * Float.BYTES, wk);
                packed.set(ValueLayout.JAVA_FLOAT, dstV * Float.BYTES, wv);
            }
        }

        return packed;
    }

    private static long index(int qkv, int head, int dim, int i, int numHeads, int headDim, int hidden) {
        return (((long) qkv * numHeads + head) * headDim + dim) * (long) hidden + i;
    }
}
