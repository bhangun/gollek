/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Default implementation of UnifiedKVCache using the GKV binary format.
 * Supports off-heap segments and zero-copy forking.
 */
public class DefaultGKVUnifiedKVCache implements UnifiedKVCache {

    private final GKVHeader header;
    private final List<GKVLayerEntry> layers;
    private final MemorySegment segment;
    private boolean isShared;
    private boolean closed = false;

    public DefaultGKVUnifiedKVCache(GKVHeader header, List<GKVLayerEntry> layers, MemorySegment segment) {
        this.header = header;
        this.layers = List.copyOf(layers);
        this.segment = segment;
        this.isShared = false;
    }

    @Override
    public String modelId() {
        return "GKV-" + header.magic();
    }

    @Override
    public int seqLength() {
        return header.seqLen();
    }

    @Override
    public int numLayers() {
        return header.numLayers();
    }

    @Override
    public KVBlock getLayer(int layerId) {
        GKVLayerEntry entry = layers.get(layerId);
        GKVDataType dtype = GKVDataType.fromId(header.dtypeId());
        
        long halfSize = entry.sizeBytes() / 2;
        return new KVBlock(
            layerId,
            header.numHeads(),
            header.headDim(),
            header.seqLen(),
            segment.asSlice(entry.keyOffset(), halfSize),
            segment.asSlice(entry.valueOffset(), halfSize),
            KVBlock.DataType.valueOf(dtype.name())
        );
    }

    @Override
    public Collection<KVBlock> getAllLayers() {
        return IntStream.range(0, numLayers())
                .mapToObj(this::getLayer)
                .collect(Collectors.toList());
    }

    @Override
    public GKVHeader getHeader() {
        return header;
    }

    @Override
    public MemorySegment asBinarySegment() {
        return segment;
    }

    @Override
    public UnifiedKVCache fork() {
        this.isShared = true;
        // Shallow copy: same memory segment, marked as shared
        return new DefaultGKVUnifiedKVCache(header, layers, segment);
    }

    @Override
    public UnifiedKVCache forkForSpeculation(int lookahead) {
        this.isShared = true;
        // For default GKV, speculation fork acts as a standard shared fork. 
        // More advanced runners (like vLLM) use lookahead to allocate extra pages.
        return new DefaultGKVUnifiedKVCache(header, layers, segment);
    }

    @Override
    public ByteBuffer serialize() {
        return segment.asByteBuffer();
    }

    @Override
    public void close() {
        if (!closed) {
            // Note: In a production pool, we'd return the segment to KVPool
            // but for simplicity here we assume the Arena handles cleanup.
            closed = true;
        }
    }

    public boolean isShared() {
        return isShared;
    }
}
