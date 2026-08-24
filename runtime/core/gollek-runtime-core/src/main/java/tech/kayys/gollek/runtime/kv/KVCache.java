package tech.kayys.gollek.runtime.kv;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.memory.CpuBuffer;
import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class KVCache {
    private final CpuBuffer kBuf;
    private final CpuBuffer vBuf;
    private final KVCodec codec;
    private final int maxSeq;
    private final int heads;
    private final int headDim;
    private final int elemsPerToken;
    private final AtomicInteger position = new AtomicInteger(0);
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public KVCache(int maxSeq, int heads, int headDim, KVCodec codec) {
        this.codec = codec;
        this.maxSeq = maxSeq;
        this.heads = heads;
        this.headDim = headDim;
        this.elemsPerToken = heads * headDim;
        long totalElems = (long) maxSeq * elemsPerToken;
        long totalBytes = totalElems * codec.bytesPerElement();
        this.kBuf = new CpuBuffer(totalBytes);
        this.vBuf = new CpuBuffer(totalBytes);
    }

    public int position() {
        return position.get();
    }

    public void setPosition(int pos) {
        position.set(pos);
    }

    public int capacity() {
        return maxSeq;
    }

    public boolean canAppend() {
        return position.get() < maxSeq;
    }

    /**
     * Appends single-token K and V tensors to the cache.
     */
    public void append(Tensor k, Tensor v) {
        if (!(k instanceof DefaultTensor dk) || !(v instanceof DefaultTensor dv)) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "Only DefaultTensor supported for KVCache append");
        }
        
        long seqLen = k.shape().dim(0);
        if (seqLen == 1) {
            append(dk.buffer().segment(), dv.buffer().segment());
        } else {
            // For multi-token, we need to append each token's K/V.
            // This is a bit inefficient without a bulk append, but correct.
            List<Tensor> kTokens = dk.backend().split(k, 0, (int) seqLen);
            List<Tensor> vTokens = dv.backend().split(v, 0, (int) seqLen);
            for (int i = 0; i < seqLen; i++) {
                append(((DefaultTensor)kTokens.get(i)).buffer().segment(), 
                       ((DefaultTensor)vTokens.get(i)).buffer().segment());
            }
        }
    }

    public void append(MemorySegment kNew, MemorySegment vNew) {
        lock.writeLock().lock();
        try {
            int pos = position.get();
            if (pos >= maxSeq) {
                throw new IllegalStateException(
                        String.format("KVCache overflow: position=%d, capacity=%d", pos, maxSeq));
            }
            long offsetBytes = (long) pos * elemsPerToken * codec.bytesPerElement();
            MemorySegment kDst = kBuf.segment().asSlice(offsetBytes);
            MemorySegment vDst = vBuf.segment().asSlice(offsetBytes);
            
            // For now assume elements are compatible or codec handles it
            codec.encode(kNew, kDst, elemsPerToken);
            codec.encode(vNew, vDst, elemsPerToken);
            
            position.incrementAndGet();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Returns a Tensor representing the full K cache up to current position.
     */
    public Tensor getFullK(ComputeBackend backend) {
        int pos = position.get();
        if (pos == 0) return null;
        
        // This is a simplified view. In a real engine, we might return a view 
        // or a tensor that handles the dequantization/decoding lazily.
        // For now, we'll assume the backend can handle the raw buffer or we dequantize.
        
        // We'll return a tensor of shape [pos, heads, headDim]
        Shape shape = new Shape(pos, heads, headDim);
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, kBuf, backend);
    }

    public Tensor getFullV(ComputeBackend backend) {
        int pos = position.get();
        if (pos == 0) return null;
        Shape shape = new Shape(pos, heads, headDim);
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, vBuf, backend);
    }

    public MemorySegment rawK() { return kBuf.segment(); }
    public MemorySegment rawV() { return vBuf.segment(); }
    public int heads() { return heads; }
    public int headDim() { return headDim; }
    public KVCodec codec() { return codec; }

    public void clear() {
        lock.writeLock().lock();
        try {
            position.set(0);
        } finally {
            lock.writeLock().unlock();
        }
    }
}
