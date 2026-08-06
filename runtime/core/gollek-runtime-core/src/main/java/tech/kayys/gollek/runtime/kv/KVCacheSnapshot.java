package tech.kayys.gollek.runtime.kv;

public final class KVCacheSnapshot {
    public final byte[] kData;
    public final byte[] vData;
    public final int position;
    public final int heads;
    public final int headDim;
    public final int maxSeq;

    public KVCacheSnapshot(
            byte[] kData,
            byte[] vData,
            int position,
            int heads,
            int headDim,
            int maxSeq) {
        this.kData = kData;
        this.vData = vData;
        this.position = position;
        this.heads = heads;
        this.headDim = headDim;
        this.maxSeq = maxSeq;
    }
}