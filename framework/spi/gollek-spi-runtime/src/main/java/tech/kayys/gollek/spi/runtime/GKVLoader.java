/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * mmap-based loader for GKV (Gollek KV) segments.
 * Provides zero-copy access to token memory using JDK 25 Foreign Memory API.
 */
public class GKVLoader {

    /**
     * Loads a GKV file from disk into a memory-mapped segment.
     */
    public static LoadedGKV load(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
            
            // Read Header
            int magic = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 0);
            if (magic != GKVHeader.MAGIC) {
                throw new IOException("Invalid GKV magic: " + Integer.toHexString(magic));
            }

            int version = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 4);
            int numLayers = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 8);
            int numHeads = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 12);
            int headDim = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 16);
            int seqLen = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 20);
            int dtypeId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 24);
            int quantizationId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, 28);
            long layerDirOffset = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, 32);
            long kvDataOffset = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, 40);
            long totalSize = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, 48);

            GKVHeader header = new GKVHeader(
                magic, version, numLayers, numHeads, headDim, seqLen, dtypeId, quantizationId,
                layerDirOffset, kvDataOffset, totalSize, ByteOrder.nativeOrder()
            );

            // Read Layer Directory
            List<GKVLayerEntry> layers = new ArrayList<>(numLayers);
            for (int i = 0; i < numLayers; i++) {
                long entryPos = layerDirOffset + (long) i * GKVLayerEntry.ENTRY_SIZE;
                int layerId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, entryPos);
                long kOffset = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, entryPos + 8);
                long vOffset = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, entryPos + 16);
                long sizeBytes = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, entryPos + 24);
                
                layers.add(new GKVLayerEntry(layerId, kOffset, vOffset, sizeBytes));
            }

            return new LoadedGKV(header, layers, segment);
        }
    }

    /**
     * Record containing the mapped GKV data.
     */
    public record LoadedGKV(
        GKVHeader header,
        List<GKVLayerEntry> layers,
        MemorySegment segment
    ) {
        public KVBlock getLayerBlock(int index) {
            GKVLayerEntry entry = layers.get(index);
            GKVDataType dtype = GKVDataType.fromId(header.dtypeId());
            
            return new KVBlock(
                entry.layerId(),
                header.numHeads(),
                header.headDim(),
                header.seqLen(),
                segment.asSlice(entry.keyOffset(), entry.sizeBytes() / 2),
                segment.asSlice(entry.valueOffset(), entry.sizeBytes() / 2),
                KVBlock.DataType.valueOf(dtype.name()) // Mapping for SPI compat
            );
        }
    }
}
