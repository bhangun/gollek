package tech.kayys.gollek.gguf.loader;

import java.lang.foreign.MemorySegment;

/**
 * Structural wrapper pointing to the zero-copy, read-only 
 * FFM MemorySegment instances containing prepacked layer weights.
 */
public final class TransformerLayerWeights {

    public final MemorySegment rmsWeight;        // [hidden]

    public final MemorySegment wqkvPacked;       // [3][heads][headDim][hidden]
    public final MemorySegment bqkv;             // [heads * headDim + 2 * headKv * headDim]

    public final TensorData wo;               // [hidden, hidden]
    public final MemorySegment bo;               // [hidden] (Bias for output projection)

    public final MemorySegment ffnNormWeight;    // [hidden]

    public final TensorData wG;               // [ffnDim, hidden]
    public final MemorySegment bg;               // [ffnDim] (Bias for gate projection)
    public final TensorData wU;               // [ffnDim, hidden]
    public final MemorySegment bu;               // [ffnDim] (Bias for up projection)
    public final TensorData wD;               // [hidden, ffnDim]
    public final MemorySegment bd;               // [hidden] (Bias for down projection)

    public TransformerLayerWeights(
            MemorySegment rmsWeight,
            MemorySegment wqkvPacked,
            MemorySegment bqkv,
            TensorData wo,
            MemorySegment bo,
            MemorySegment ffnNormWeight,
            TensorData wG,
            MemorySegment bg,
            TensorData wU,
            MemorySegment bu,
            TensorData wD,
            MemorySegment bd) {
        this.rmsWeight = rmsWeight;
        this.wqkvPacked = wqkvPacked;
        this.bqkv = bqkv;
        this.wo = wo;
        this.bo = bo;
        this.ffnNormWeight = ffnNormWeight;
        this.wG = wG;
        this.bg = bg;
        this.wU = wU;
        this.bu = bu;
        this.wD = wD;
        this.bd = bd;
    }
}
