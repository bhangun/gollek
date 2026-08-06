package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;

public final class KVCacheSerializer {
    private KVCacheSerializer() {
    }

    public static KVCacheSnapshot snapshot(KVCache cache) {
        int elems = cache.position() * cache.heads() * cache.headDim();
        int bytes = elems * cache.codec().bytesPerElement();
        byte[] k = new byte[bytes];
        byte[] v = new byte[bytes];
        MemorySegment.copy(
                cache.rawK(), 0,
                MemorySegment.ofArray(k), 0,
                bytes);
        MemorySegment.copy(
                cache.rawV(), 0,
                MemorySegment.ofArray(v), 0,
                bytes);
        return new KVCacheSnapshot(
                k,
                v,
                cache.position(),
                cache.heads(),
                cache.headDim(),
                cache.capacity());
    }

    public static KVCache restore(
            KVCacheSnapshot snap,
            KVCodec codec) {
        KVCache cache = new KVCache(
                snap.maxSeq,
                snap.heads,
                snap.headDim,
                codec);
        int bytes = snap.kData.length;
        MemorySegment.copy(
                MemorySegment.ofArray(snap.kData), 0,
                cache.rawK(), 0,
                bytes);
        MemorySegment.copy(MemorySegment.ofArray(snap.vData), 0,
                cache.rawV(), 0,
                bytes);
        cache.setPosition(snap.position);
        return cache;
    }
}