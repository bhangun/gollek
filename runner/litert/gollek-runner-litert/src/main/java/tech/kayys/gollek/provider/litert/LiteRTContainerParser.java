package tech.kayys.gollek.provider.litert;

import lombok.extern.slf4j.Slf4j;

import org.jboss.logging.Logger;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Parser for LITERTLM container format and TFLite FlatBuffer metadata.
 *
 * <p>The LITERTLM container is a multi-model format:
 * <pre>
 * [8 bytes]  Magic: "LITERTLM"
 * [4 bytes]  Version (uint32 LE)
 * [4 bytes]  Entry count (uint32 LE)
 * [8 bytes]  Reserved
 * [variable] FlatBuffer TOC + embedded TFLite models
 * </pre>
 *
 * <p>Also parses standalone .task files (MediaPipe Task Bundles) to extract
 * embedded SPM tokenizer and detect LLM model type from TFLite metadata.
 */
@Slf4j
public class LiteRTContainerParser {

    private static final byte[] LITERTLM_MAGIC = "LITERTLM".getBytes();
    private static final byte[] TFL3_MAGIC = { 'T', 'F', 'L', '3' };
    private static final byte[] ZIP_MAGIC = { 'P', 'K', 0x03, 0x04 };
    private static final int HEADER_SIZE = 24; // 8 + 4 + 4 + 8

    // ===== Container parsing result =====

    public record ContainerInfo(
            ContainerFormat format,
            int version,
            List<SubModelEntry> subModels,
            long tfliteOffset,    // offset of the primary TFLite model (for .task files)
            long tfliteSize,      // size of the primary TFLite model
            Map<String, BufferRef> metadataBuffers // name -> buffer ref from TFLite metadata
    ) {
        /** Get the prefill-decode sub-model (main LLM model). */
        public Optional<SubModelEntry> prefillDecodeModel() {
            var named = subModels.stream()
                    .filter(e -> e.modelType().contains("prefill_decode"))
                    .findFirst();
            if (named.isPresent()) return named;
            
            // Fallback to the largest segment if nothing named prefill_decode exists
            return subModels.stream()
                    .max((a, b) -> Long.compare(a.size(), b.size()));
        }

        /** Get the MTP drafter sub-model for speculative decoding. */
        public Optional<SubModelEntry> drafterModel() {
            return subModels.stream()
                    .filter(e -> e.modelType().contains("mtp_drafter"))
                    .findFirst();
        }

        /** Get the embedder sub-model. */
        public Optional<SubModelEntry> embedderModel() {
            return subModels.stream()
                    .filter(e -> e.modelType().contains("embedder") && !e.modelType().contains("per_layer"))
                    .findFirst();
        }

        /** Check if this is an LLM model. */
        public boolean isLlmModel() {
            if (format == ContainerFormat.LITERTLM) {
                return prefillDecodeModel().isPresent();
            }
            
            // For .task files, check metadata
            boolean hasLlmMeta = metadataBuffers.containsKey("spm_vocab_model")
                    || metadataBuffers.containsKey("odml.infra.proto.LlmParameters");
            
            // Fallback: If metadata parsing failed (brittle), use name hint for known LLM task files
            if (!hasLlmMeta && format == ContainerFormat.TFLITE) {
                // We'll be conservative but allow the gemma .task files
                // This will be caught later if it's truly not an LLM
                return true; 
            }
            
            return hasLlmMeta;
        }
    }

    public record SubModelEntry(
            String modelType,
            long offset,   // offset within the file
            long size       // size in bytes
    ) {}

    public record BufferRef(
            int bufferIndex,
            long dataOffset,
            long dataSize
    ) {}

    public record WeightEntry(
            String name,
            long offset,
            long size,
            int type // kTfLiteType
    ) {}

    public enum ContainerFormat {
        LITERTLM,   // Multi-model container (.litertlm)
        TFLITE,      // Standard TFLite FlatBuffer (.tflite, .task)
        UNKNOWN
    }

    // ===== Public API =====

    /**
     * Parse a model file and determine its format and structure.
     */
    public static ContainerInfo parse(Path modelPath) throws IOException {
        long fileSize = Files.size(modelPath);
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate((int) Math.min(32L, fileSize));
            header.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header, 0);
            header.flip();

            String fileName = modelPath.getFileName().toString().toLowerCase();
            boolean isTaskFile = fileName.endsWith(".task") || fileName.endsWith(".bin");

            if (fileSize >= 8 && matchesMagic(header, LITERTLM_MAGIC)) {
                return parseLitertlmContainer(channel, fileSize);
            }

            if (isTaskFile || (fileSize >= 4 && matchesMagic(header, ZIP_MAGIC))) {
                 return scanForSegments(channel, fileSize);
            }

            if (fileSize >= 8 && matchesTfl3(header)) {
                return parseTfliteFile(channel, fileSize);
            }

            return new ContainerInfo(ContainerFormat.UNKNOWN, 0,
                    List.of(), 0, fileSize, Map.of());
        }
    }

    /** Extract the primary TFLite model from any container format. */
    public static byte[] extractTfliteModel(Path modelPath) throws IOException {
        ContainerInfo info = parse(modelPath);
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate((int) info.tfliteSize());
            channel.read(buf, info.tfliteOffset());
            buf.flip();
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            return data;
        }
    }

    /** Extract all weight tensors with their names and file offsets. */
    public static Map<String, WeightEntry> extractWeightMap(Path modelPath) throws IOException {
        ContainerInfo info = parse(modelPath);
        long segmentOffset = info.tfliteOffset();
        long segmentSize = info.tfliteSize();
        
        // For large segments (>1GB), we can't load the entire thing into byte[].
        // Strategy: read a generous metadata chunk (64MB) for structural parsing,
        // then use a separate pass for buffer data resolution if needed.
        // FlatBuffer metadata (vtables, tensor descriptors, strings) is at the
        // beginning; weight DATA buffers are at large offsets (we only record positions).
        int metadataChunkSize = (int) Math.min(segmentSize, 64 * 1024 * 1024L);
        
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(metadataChunkSize);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(buf, segmentOffset);
            buf.flip();
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            
            return extractWeightMapFromBytes(data, segmentOffset);
        }
    }
    
    /** Core weight extraction from an already-loaded TFLite FlatBuffer byte array. */
    static Map<String, WeightEntry> extractWeightMapFromBytes(byte[] data, long fileBaseOffset) {
        Map<String, WeightEntry> weights = new LinkedHashMap<>();
        try {
            int rootOffset = readUint32(data, 0);
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int vtableSize = readUint16(data, vtablePos);
            int numFields = (vtableSize - 4) / 2;

            if (numFields <= 2) return weights;

            // Field 2 = subgraphs vector
            int sgOff = readUint16(data, vtablePos + 4 + 2 * 2);
            if (sgOff == 0) return weights;

            int sgVecPos = rootOffset + sgOff + readUint32(data, rootOffset + sgOff);
            int sgCount = readUint32(data, sgVecPos);

            for (int si = 0; si < sgCount; si++) {
                int entryPos = sgVecPos + 4 + si * 4 + readUint32(data, sgVecPos + 4 + si * 4);
                int evtPos = entryPos - readInt32(data, entryPos);
                int tensorsOff = readUint16(data, evtPos + 4);
                if (tensorsOff == 0) continue;

                int tVecPos = entryPos + tensorsOff + readUint32(data, entryPos + tensorsOff);
                int tCount = readUint32(data, tVecPos);

                for (int ti = 0; ti < tCount; ti++) {
                    try {
                        int tPos = tVecPos + 4 + ti * 4 + readUint32(data, tVecPos + 4 + ti * 4);
                        int tvPos = tPos - readInt32(data, tPos);
                        int tFields = (readUint16(data, tvPos) - 4) / 2;

                        // Field 3 = name (string)
                        String name = null;
                        if (tFields > 3) {
                            int nOff = readUint16(data, tvPos + 4 + 3 * 2);
                            if (nOff > 0) {
                                int namePos = tPos + nOff + readUint32(data, tPos + nOff);
                                int nameLen = readUint32(data, namePos);
                                name = new String(data, namePos + 4, Math.min(nameLen, 256));
                            }
                        }
                        if (name == null || name.isEmpty()) continue;

                        // Field 1 = type (TensorType enum, stored as byte)
                        int typeOff = tFields > 1 ? readUint16(data, tvPos + 4 + 1 * 2) : 0;
                        int type = typeOff > 0 ? data[tPos + typeOff] : 0;

                        // Field 2 = buffer index
                        int bufIdxOff = tFields > 2 ? readUint16(data, tvPos + 4 + 2 * 2) : 0;
                        int bufIdx = bufIdxOff > 0 ? readUint32(data, tPos + bufIdxOff) : 0;

                        BufferRef br = resolveBufferRef(data, rootOffset, bufIdx);
                        if (br != null && br.dataSize() > 0) {
                            weights.put(name, new WeightEntry(name, 
                                    fileBaseOffset + br.dataOffset(), br.dataSize(), type));
                        }
                    } catch (ArrayIndexOutOfBoundsException e) {
                        // Skip malformed tensor entries
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Failed to extract weight map: {}", e.getMessage());
        }
        return weights;
    }

    /** Extract a specific sub-model segment from a .litertlm container. */
    public static Optional<byte[]> extractSubModelData(Path modelPath, String modelType) throws IOException {
        ContainerInfo info = parse(modelPath);
        if (info.format() != ContainerFormat.LITERTLM) return Optional.empty();
        
        return info.subModels().stream()
                .filter(m -> m.modelType().equalsIgnoreCase(modelType))
                .findFirst()
                .map(m -> {
                    try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
                        ByteBuffer buf = ByteBuffer.allocate((int) m.size());
                        channel.read(buf, m.offset());
                        buf.flip();
                        byte[] data = new byte[buf.remaining()];
                        buf.get(data);
                        return data;
                    } catch (IOException e) {
                        return null;
                    }
                });
    }

    private static ContainerInfo scanForSegments(FileChannel channel, long fileSize) throws IOException {
        List<Long> offsets = findTfl3Segments(channel, fileSize);
        List<SubModelEntry> subModels = new ArrayList<>();
        
        for (int i = 0; i < offsets.size(); i++) {
            long offset = offsets.get(i);
            long size = (i + 1 < offsets.size()) ? offsets.get(i + 1) - offset : fileSize - offset;
            subModels.add(new SubModelEntry("segment_" + i, offset, size));
        }

        if (subModels.isEmpty()) {
            return new ContainerInfo(ContainerFormat.UNKNOWN, 0, List.of(), 0, fileSize, Map.of());
        }

        // Heuristic: the largest segment is usually the main LLM model
        SubModelEntry largest = subModels.stream()
                .max((a, b) -> Long.compare(a.size(), b.size()))
                .orElse(subModels.get(0));

        log.info("Scanned container: found {} segments, primary segment at 0x{} ({} bytes)", 
                subModels.size(), Long.toHexString(largest.offset()), largest.size());

        // Also try to parse metadata from the FIRST segment (often contains the task metadata)
        Map<String, BufferRef> metadata = new LinkedHashMap<>();
        if (!subModels.isEmpty()) {
            ContainerInfo firstSegmentMeta = parseTfliteFileAtOffset(channel, subModels.get(0).offset(), subModels.get(0).size());
            metadata.putAll(firstSegmentMeta.metadataBuffers());
        }

        return new ContainerInfo(ContainerFormat.TFLITE, 1, subModels, 
                largest.offset(), largest.size(), metadata);
    }

    private static ContainerInfo parseTfliteFileAtOffset(FileChannel channel, long offset, long size) throws IOException {
        // Implementation similar to parseTfliteFile but with offset
        int readSize = (int) Math.min(256 * 1024L, size); 
        ByteBuffer buf = ByteBuffer.allocate(readSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        channel.read(buf, offset);
        buf.flip();
        byte[] data = new byte[buf.remaining()];
        buf.get(data);

        Map<String, BufferRef> metadataBuffers = new LinkedHashMap<>();
        try {
            int rootOffset = readUint32(data, 0);
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int vtableSize = readUint16(data, vtablePos);
            int numFields = (vtableSize - 4) / 2;

            if (numFields > 6) {
                int field6Offset = readUint16(data, vtablePos + 4 + 6 * 2);
                if (field6Offset > 0) {
                    int abs6 = rootOffset + field6Offset;
                    int metaVecRel = readUint32(data, abs6);
                    int metaVecPos = abs6 + metaVecRel;
                    int metaCount = readUint32(data, metaVecPos);

                    for (int mi = 0; mi < metaCount && mi < 20; mi++) {
                        try {
                            int entryRel = readUint32(data, metaVecPos + 4 + mi * 4);
                            int entryPos = metaVecPos + 4 + mi * 4 + entryRel;
                            int evRel = readInt32(data, entryPos);
                            int evPos = entryPos - evRel;
                            int evSize = readUint16(data, evPos);
                            int eFields = (evSize - 4) / 2;
                            if (eFields < 2) continue;
                            int nameOff = readUint16(data, evPos + 4);
                            int bufOff = readUint16(data, evPos + 6);

                            if (nameOff > 0 && bufOff > 0) {
                                int nameAbs = entryPos + nameOff;
                                int nameRel = readUint32(data, nameAbs);
                                int namePos = nameAbs + nameRel;
                                int nameLen = readUint32(data, namePos);
                                String name = new String(data, namePos + 4, Math.min(nameLen, 200));
                                int bufIdx = readUint32(data, entryPos + bufOff);
                                BufferRef bufRef = resolveBufferRef(data, rootOffset, bufIdx);
                                if (bufRef != null) {
                                    // Adjust buffer offset to be relative to the start of the file
                                    metadataBuffers.put(name, new BufferRef(bufIdx, offset + bufRef.dataOffset(), bufRef.dataSize()));
                                }
                            }
                        } catch (Exception e) {}
                    }
                }
            }
        } catch (Exception e) {}
        
        return new ContainerInfo(ContainerFormat.TFLITE, 3, List.of(), offset, size, metadataBuffers);
    }

    /**
     * Extract the SPM vocabulary model bytes from a .task file.
     */
    public static Optional<byte[]> extractSpmVocab(Path modelPath) throws IOException {
        ContainerInfo info = parse(modelPath);
        BufferRef spmRef = info.metadataBuffers().get("spm_vocab_model");
        if (spmRef == null || spmRef.dataOffset() <= 0 || spmRef.dataSize() <= 0) {
            return Optional.empty();
        }

        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate((int) spmRef.dataSize());
            channel.read(buf, spmRef.dataOffset());
            buf.flip();
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            return Optional.of(data);
        }
    }

    /**
     * Find the best model file in a directory for LLM inference.
     * Priority: .litertlm > .task
     */
    public static Optional<Path> findBestModelFile(Path modelDir) throws IOException {
        if (!Files.isDirectory(modelDir)) {
            if (Files.isRegularFile(modelDir)) {
                String name = modelDir.getFileName().toString();
                // If it's not the optimal format, check if optimal .litertlm exists in the same directory
                if (name.endsWith(".task") || name.endsWith(".tflite") || name.endsWith(".bin")) {
                    try (var stream = Files.list(modelDir.getParent())) {
                        Optional<Path> litertlm = stream
                                .filter(p -> p.getFileName().toString().endsWith(".litertlm"))
                                .findFirst();
                        if (litertlm.isPresent()) {
                            log.info("Auto-upgrading model path from {} to optimal format {}", name, litertlm.get().getFileName());
                            return litertlm;
                        }
                    } catch (Exception e) {
                        // ignore and fall back
                    }
                }
                return Optional.of(modelDir);
            }
            return Optional.empty();
        }

        // Priority 1: .litertlm files
        try (var stream = Files.list(modelDir)) {
            Optional<Path> litertlm = stream
                    .filter(p -> p.getFileName().toString().endsWith(".litertlm"))
                    .findFirst();
            if (litertlm.isPresent()) {
                return litertlm;
            }
        }

        // Priority 2: .task files
        try (var stream = Files.list(modelDir)) {
            Optional<Path> task = stream
                    .filter(p -> p.getFileName().toString().endsWith(".task"))
                    .findFirst();
            if (task.isPresent()) {
                return task;
            }
        }

        // Priority 3: .tflite files
        try (var stream = Files.list(modelDir)) {
            return stream
                    .filter(p -> {
                        String name = p.getFileName().toString();
                        return name.endsWith(".tflite") || name.endsWith(".tfl");
                    })
                    .findFirst();
        }
    }

    // ===== Internal: LITERTLM container parsing =====

    private static ContainerInfo parseLitertlmContainer(FileChannel channel, long fileSize)
            throws IOException {
        ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE);
        header.order(ByteOrder.LITTLE_ENDIAN);
        channel.read(header, 0);
        header.flip();

        header.position(8); // skip magic
        int version = header.getInt();
        int entryCount = header.getInt();

        log.info("LITERTLM container v{}, {} entries, file size: {} bytes", version, entryCount, fileSize);

        // Read the TOC area (after the 24-byte header)
        int tocSize = (int) Math.min(8192L, fileSize - HEADER_SIZE);
        ByteBuffer tocBuf = ByteBuffer.allocate(tocSize);
        tocBuf.order(ByteOrder.LITTLE_ENDIAN);
        channel.read(tocBuf, HEADER_SIZE);
        tocBuf.flip();

        byte[] tocBytes = new byte[tocBuf.remaining()];
        tocBuf.get(tocBytes);

        List<SubModelEntry> subModels = new ArrayList<>();
        List<ModelTypeRef> modelTypeRefs = findModelTypeEntries(tocBytes);
        List<Long> tfl3Offsets = findTfl3Segments(channel, fileSize);

        // Map names from TOC to TFLite segments by count/index
        for (int i = 0; i < tfl3Offsets.size(); i++) {
            long offset = tfl3Offsets.get(i);
            long size = (i + 1 < tfl3Offsets.size()) ? tfl3Offsets.get(i + 1) - offset : fileSize - offset;
            String name = (i < modelTypeRefs.size()) ? modelTypeRefs.get(i).modelType : "segment_" + i;
            subModels.add(new SubModelEntry(name, offset, size));
            log.debug("  Found model segment: {} at offset 0x{} ({} bytes)", name, Long.toHexString(offset), size);
        }

        // Heuristic: the largest segment is usually the main LLM model
        SubModelEntry largest = subModels.stream()
                .max((a, b) -> Long.compare(a.size(), b.size()))
                .orElse(null);

        long primaryOffset = largest == null ? 0 : largest.offset();
        long primarySize = largest == null ? fileSize : largest.size();

        return new ContainerInfo(ContainerFormat.LITERTLM, version,
                Collections.unmodifiableList(subModels),
                primaryOffset, primarySize, Map.of());
    }

    // ===== Internal: TFLite file parsing =====

    private static ContainerInfo parseTfliteFile(FileChannel channel, long fileSize)
            throws IOException {
        // Read enough to parse metadata
        int readSize = (int) Math.min(256 * 1024L, fileSize); // 256KB for metadata
        ByteBuffer buf = ByteBuffer.allocate(readSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        channel.read(buf, 0);
        buf.flip();
        byte[] data = new byte[buf.remaining()];
        buf.get(data);

        Map<String, BufferRef> metadataBuffers = new LinkedHashMap<>();

        // Parse TFLite FlatBuffer to extract metadata entries
        try {
            int rootOffset = readUint32(data, 0);

            // Find metadata field (field 6)
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int vtableSize = readUint16(data, vtablePos);
            int numFields = (vtableSize - 4) / 2;

            if (numFields > 6) {
                int field6Offset = readUint16(data, vtablePos + 4 + 6 * 2);
                if (field6Offset > 0) {
                    int abs6 = rootOffset + field6Offset;
                    int metaVecRel = readUint32(data, abs6);
                    int metaVecPos = abs6 + metaVecRel;
                    int metaCount = readUint32(data, metaVecPos);

                    for (int mi = 0; mi < metaCount && mi < 20; mi++) {
                        try {
                            int entryRel = readUint32(data, metaVecPos + 4 + mi * 4);
                            int entryPos = metaVecPos + 4 + mi * 4 + entryRel;

                            int evRel = readInt32(data, entryPos);
                            int evPos = entryPos - evRel;
                            int evSize = readUint16(data, evPos);
                            int eFields = (evSize - 4) / 2;

                            if (eFields < 2) continue;

                            int nameOff = readUint16(data, evPos + 4);
                            int bufOff = readUint16(data, evPos + 6);

                            if (nameOff > 0 && bufOff > 0) {
                                int nameAbs = entryPos + nameOff;
                                int nameRel = readUint32(data, nameAbs);
                                int namePos = nameAbs + nameRel;
                                int nameLen = readUint32(data, namePos);
                                String name = new String(data, namePos + 4,
                                        Math.min(nameLen, 200));
                                int bufIdx = readUint32(data, entryPos + bufOff);

                                // Resolve buffer offset from the buffers vector
                                BufferRef bufRef = resolveBufferRef(data, rootOffset, bufIdx);
                                if (bufRef != null) {
                                    metadataBuffers.put(name, bufRef);
                                } else {
                                    metadataBuffers.put(name,
                                            new BufferRef(bufIdx, -1, -1));
                                }

                                log.debug("  TFLite metadata[{}]: name=\"{}\", buffer_index={}", mi, name, bufIdx);
                            }
                        } catch (Exception e) {
                            log.debug("Failed to parse metadata entry {}: {}", mi, e.getMessage());
                        }
                    }
                }
            }

            // Parse subgraph name
            String subgraphName = parseSubgraphName(data, rootOffset, numFields);
            log.info("TFLite model, subgraph: \"{}\", metadata entries: {}", subgraphName, metadataBuffers.size());

        } catch (Exception e) {
            log.warn("Failed to parse TFLite metadata: {}", e.getMessage());
        }

        return new ContainerInfo(ContainerFormat.TFLITE, 3,
                List.of(), 0, fileSize, metadataBuffers);
    }

    // ===== Internal helpers =====

    private static long readUint64(byte[] data, int offset) {
        long low = readUint32(data, offset) & 0xFFFFFFFFL;
        long high = readUint32(data, offset + 4) & 0xFFFFFFFFL;
        return low | (high << 32);
    }

    private static BufferRef resolveBufferRef(byte[] data, int rootOffset, int bufferIndex) {
        try {
            // Field 4 = buffers vector
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int numFields = (readUint16(data, vtablePos) - 4) / 2;

            if (numFields <= 4) return null;
            int field4Offset = readUint16(data, vtablePos + 4 + 4 * 2);
            if (field4Offset == 0) return null;

            int abs4 = rootOffset + field4Offset;
            int bufVecRel = readUint32(data, abs4);
            int bufVecPos = abs4 + bufVecRel;
            int bufCount = readUint32(data, bufVecPos);

            if (bufferIndex >= bufCount) return null;

            int bufEntryRel = readUint32(data, bufVecPos + 4 + bufferIndex * 4);
            int bufEntryPos = bufVecPos + 4 + bufferIndex * 4 + bufEntryRel;

            int bvRel = readInt32(data, bufEntryPos);
            int bvPos = bufEntryPos - bvRel;
            int bvSize = readUint16(data, bvPos);
            int bFields = (bvSize - 4) / 2;

            int dataOff = bFields > 0 ? readUint16(data, bvPos + 4) : 0;
            if (dataOff > 0) {
                int dataAbs = bufEntryPos + dataOff;
                int dataRel = readUint32(data, dataAbs);
                int dataPos = dataAbs + dataRel;
                int dataLen = readUint32(data, dataPos);
                return new BufferRef(bufferIndex, dataPos + 4, dataLen);
            }

            // Check for external buffer offset/size (fields 1 and 2)
            if (bFields > 2) {
                int offField = readUint16(data, bvPos + 4 + 1 * 2);
                int szField = readUint16(data, bvPos + 4 + 2 * 2);
                if (offField > 0 && szField > 0) {
                    long extOffset = readUint64(data, bufEntryPos + offField);
                    long extSize = readUint64(data, bufEntryPos + szField);
                    if (extSize > 0) {
                        return new BufferRef(bufferIndex, extOffset, extSize);
                    }
                }
            }

            return new BufferRef(bufferIndex, -1, -1);
        } catch (Exception e) {
            return null;
        }
    }

    private static String parseSubgraphName(byte[] data, int rootOffset, int numFields) {
        try {
            if (numFields <= 2) return "<unknown>";
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int field2Offset = readUint16(data, vtablePos + 4 + 2 * 2);
            if (field2Offset == 0) return "<no subgraphs>";

            int abs2 = rootOffset + field2Offset;
            int subVecRel = readUint32(data, abs2);
            int subVecPos = abs2 + subVecRel;
            int subCount = readUint32(data, subVecPos);

            if (subCount == 0) return "<empty>";

            // Parse first subgraph
            int sub0Rel = readUint32(data, subVecPos + 4);
            int sub0Pos = subVecPos + 4 + sub0Rel;

            int svRel = readInt32(data, sub0Pos);
            int svPos = sub0Pos - svRel;
            int svSize = readUint16(data, svPos);
            int sFields = (svSize - 4) / 2;

            // Field 4 = name
            if (sFields > 4) {
                int nameOff = readUint16(data, svPos + 4 + 4 * 2);
                if (nameOff > 0) {
                    int nameAbs = sub0Pos + nameOff;
                    int nameRel = readUint32(data, nameAbs);
                    int namePos = nameAbs + nameRel;
                    int nameLen = readUint32(data, namePos);
                    return new String(data, namePos + 4, Math.min(nameLen, 100));
                }
            }
        } catch (Exception e) {
            // ignore
        }
        return "<unknown>";
    }

    /**
     * Check if a TFLite model at the given offset has INT4 tensors.
     */
    public static boolean hasInt4Tensors(Path modelPath, long offset, long size) {
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            int readSize = (int) Math.min(1024 * 1024L, size); 
            ByteBuffer buf = ByteBuffer.allocate(readSize);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(buf, offset);
            buf.flip();
            byte[] data = new byte[buf.remaining()];
            buf.get(data);

            int rootOffset = readUint32(data, 0);
            int vtableRel = readInt32(data, rootOffset);
            int vtablePos = rootOffset - vtableRel;
            int vtableSize = readUint16(data, vtablePos);
            int numFields = (vtableSize - 4) / 2;

            // Subgraphs are field 3
            if (numFields > 3) {
                int sgOff = readUint16(data, vtablePos + 4 + 3 * 2);
                if (sgOff > 0) {
                    int absSg = rootOffset + sgOff;
                    int sgVecRel = readUint32(data, absSg);
                    int sgVecPos = absSg + sgVecRel;
                    int sgCount = readUint32(data, sgVecPos);

                    for (int si = 0; si < sgCount; si++) {
                        int entryRel = readUint32(data, sgVecPos + 4 + si * 4);
                        int entryPos = sgVecPos + 4 + si * 4 + entryRel;
                        int evRel = readInt32(data, entryPos);
                        int evPos = entryPos - evRel;
                        
                        // Tensors are field 0 of subgraph
                        int tensorsOff = readUint16(data, evPos + 4);
                        if (tensorsOff > 0) {
                            int absT = entryPos + tensorsOff;
                            int tVecRel = readUint32(data, absT);
                            int tVecPos = absT + tVecRel;
                            int tCount = readUint32(data, tVecPos);

                            for (int ti = 0; ti < tCount; ti++) {
                                int tRel = readUint32(data, tVecPos + 4 + ti * 4);
                                int tPos = tVecPos + 4 + ti * 4 + tRel;
                                int tvRel = readInt32(data, tPos);
                                int tvPos = tPos - tvRel;
                                int tvSize = readUint16(data, tvPos);
                                int tFields = (tvSize - 4) / 2;
                                
                                // Type is field 2 of tensor
                                if (tFields > 2) {
                                    int typeOff = readUint16(data, tvPos + 4 + 2 * 2);
                                    if (typeOff > 0) {
                                        int type = readUint32(data, tPos + typeOff);
                                        if (type == 19) return true; // kTfLiteInt4
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {}
        return false;
    }

    private record ModelTypeRef(String modelType, int tocOffset) {}

    private static List<ModelTypeRef> findModelTypeEntries(byte[] tocBytes) {
        List<ModelTypeRef> refs = new ArrayList<>();
        byte[] prefix = "tf_lite_".getBytes();

        for (int i = 0; i < tocBytes.length - prefix.length; i++) {
            boolean match = true;
            for (int j = 0; j < prefix.length; j++) {
                if (tocBytes[i + j] != prefix[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // Read until null terminator
                int end = i;
                while (end < tocBytes.length && tocBytes[end] != 0) {
                    end++;
                }
                String modelType = new String(tocBytes, i, end - i);
                refs.add(new ModelTypeRef(modelType, i));
                i = end; // skip past this string
            }
        }
        return refs;
    }

    public static List<Long> findTfl3SegmentsForInspection(Path modelPath) throws IOException {
        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return findTfl3Segments(channel, channel.size());
        }
    }

    private static List<Long> findTfl3Segments(FileChannel channel, long fileSize)
            throws IOException {
        List<Long> offsets = new ArrayList<>();
        byte[] chunk = new byte[1024 * 1024]; // 1MB chunks
        ByteBuffer chunkBuf = ByteBuffer.wrap(chunk);

        for (long pos = 0; pos < fileSize - 8; ) {
            chunkBuf.clear();
            int bytesRead = channel.read(chunkBuf, pos);
            if (bytesRead <= 8) break;

            for (int j = 0; j < bytesRead - 8; j++) {
                if (chunk[j + 4] == TFL3_MAGIC[0] && chunk[j + 5] == TFL3_MAGIC[1]
                        && chunk[j + 6] == TFL3_MAGIC[2] && chunk[j + 7] == TFL3_MAGIC[3]) {
                    offsets.add(pos + j);
                    j += 4096; // Skip past model body
                }
            }
            pos += (bytesRead - 8);
        }

        return offsets;
    }

    private static List<SubModelEntry> parseFromFlatBufferToc(
            byte[] tocBytes, FileChannel channel, long fileSize, int headerSize)
            throws IOException {
        // Fallback: try to interpret the TOC as containing offset/size pairs
        // associated with model_type strings
        List<SubModelEntry> entries = new ArrayList<>();
        List<ModelTypeRef> modelTypes = findModelTypeEntries(tocBytes);
        List<Long> tfl3 = findTfl3Segments(channel, fileSize);

        // Simple heuristic: match model types to TFL3 segments in order
        for (int i = 0; i < Math.min(modelTypes.size(), tfl3.size()); i++) {
            long off = tfl3.get(i);
            long sz = (i + 1 < tfl3.size()) ? tfl3.get(i + 1) - off : fileSize - off;
            entries.add(new SubModelEntry(modelTypes.get(i).modelType, off, sz));
        }
        return entries;
    }

    private static boolean matchesMagic(ByteBuffer buf, byte[] magic) {
        if (buf.remaining() < magic.length) return false;
        for (int i = 0; i < magic.length; i++) {
            if (buf.get(i) != magic[i]) return false;
        }
        return true;
    }

    private static boolean matchesTfl3(ByteBuffer buf) {
        if (buf.remaining() < 8) return false;
        return buf.get(4) == 'T' && buf.get(5) == 'F'
                && buf.get(6) == 'L' && buf.get(7) == '3';
    }

    // ===== Internal: Zip/MediaPipe .task parsing =====

    private static ContainerInfo parseZipContainer(Path path, long fileSize) throws IOException {
        log.info("Detected ZIP/MediaPipe container: {}", path.getFileName());
        List<SubModelEntry> subModels = new ArrayList<>();
        Map<String, BufferRef> metadata = new LinkedHashMap<>();
        
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            List<Long> tfl3Offsets = findTfl3Segments(channel, fileSize);
            for (int i = 0; i < tfl3Offsets.size(); i++) {
                long offset = tfl3Offsets.get(i);
                long size = (i + 1 < tfl3Offsets.size()) ? tfl3Offsets.get(i + 1) - offset : fileSize - offset;
                subModels.add(new SubModelEntry("tflite_model_" + i, offset, size));
                log.debug("  Found TFLite model in ZIP at offset 0x{}", Long.toHexString(offset));
            }
        }

        if (subModels.isEmpty()) {
            log.warn("No TFLite models found in ZIP container {}", path);
        }

        return new ContainerInfo(ContainerFormat.TFLITE, 1, subModels, 
                subModels.isEmpty() ? 0 : subModels.get(0).offset(),
                subModels.isEmpty() ? fileSize : subModels.get(0).size(), 
                metadata);
    }

    private static int readUint32(byte[] data, int offset) {
        return (data[offset] & 0xFF)
                | ((data[offset + 1] & 0xFF) << 8)
                | ((data[offset + 2] & 0xFF) << 16)
                | ((data[offset + 3] & 0xFF) << 24);
    }

    private static int readInt32(byte[] data, int offset) {
        return readUint32(data, offset);
    }

    private static int readUint16(byte[] data, int offset) {
        return (data[offset] & 0xFF) | ((data[offset + 1] & 0xFF) << 8);
    }
}
