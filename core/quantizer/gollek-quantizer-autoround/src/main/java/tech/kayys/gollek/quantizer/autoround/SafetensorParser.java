package tech.kayys.gollek.quantizer.autoround;

import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;

/**
 * Simple safetensor parser for AutoRound.
 * Wraps the safetensor-loader infrastructure for direct tensor access.
 */
public class SafetensorParser implements AutoCloseable {
    private final Path filePath;
    private final SafetensorHeader header;
    private final Arena arena;
    private final MemorySegment dataSegment;
    private final long dataBlobOffset;

    public SafetensorParser(Path filePath) throws IOException {
        this.filePath = filePath;
        this.arena = Arena.ofAuto();
        
        try (RandomAccessFile raf = new RandomAccessFile(filePath.toFile(), "r")) {
            FileChannel channel = raf.getChannel();
            
            // Read header JSON
            long headerLen = channel.map(FileChannel.MapMode.READ_ONLY, 0, 8, arena)
                    .get(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, 0);
            
            MemorySegment headerSeg = channel.map(FileChannel.MapMode.READ_ONLY, 8, headerLen, arena);
            String headerJson = headerSeg.getString(0, java.nio.charset.StandardCharsets.UTF_8);
            
            // Parse header
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            Map<String, Object> root = mapper.readValue(headerJson, Map.class);
            
            // Build header manually
            this.header = buildHeader(root);
            this.dataBlobOffset = 8 + headerLen;
            
            // Map data segment
            long fileSize = channel.size();
            long dataSize = fileSize - dataBlobOffset;
            this.dataSegment = channel.map(FileChannel.MapMode.READ_ONLY, dataBlobOffset, dataSize, arena);
        }
    }

    private SafetensorHeader buildHeader(Map<String, Object> root) {
        Map<String, String> metadata = new java.util.HashMap<>();
        Map<String, SafetensorTensorInfo> tensors = new java.util.LinkedHashMap<>();

        for (var entry : root.entrySet()) {
            if ("__metadata__".equals(entry.getKey())) {
                if (entry.getValue() instanceof Map<?, ?> meta) {
                    meta.forEach((k, v) -> metadata.put(k.toString(), v != null ? v.toString() : ""));
                }
            } else if (entry.getValue() instanceof Map<?, ?> tensorMap) {
                String dtype = (String) tensorMap.get("dtype");
                java.util.List<Number> shapeList = (java.util.List<Number>) tensorMap.get("shape");
                java.util.List<Number> offsetList = (java.util.List<Number>) tensorMap.get("data_offsets");
                
                if (dtype != null && shapeList != null && offsetList != null && offsetList.size() == 2) {
                    long[] shape = shapeList.stream().mapToLong(Number::longValue).toArray();
                    long[] offsets = offsetList.stream().mapToLong(Number::longValue).toArray();
                    tensors.put(entry.getKey(), new SafetensorTensorInfo(entry.getKey(), dtype, shape, offsets));
                }
            }
        }

        return SafetensorHeader.of(8L, tensors, metadata);
    }

    public SafetensorHeader getHeader() {
        return header;
    }

    public MemorySegment getTensorData(String tensorName) {
        SafetensorTensorInfo info = header.findTensor(tensorName).orElse(null);
        if (info == null) {
            return null;
        }
        long begin = info.dataBegin();
        long end = info.dataEnd();
        return dataSegment.asSlice(begin, end - begin);
    }

    /** Alias for {@link #getTensorData(String)} for compatibility with loader API. */
    public MemorySegment getTensorSegment(String tensorName) {
        return getTensorData(tensorName);
    }

    public int[] readInt32Tensor(String tensorName) {
        MemorySegment seg = getTensorData(tensorName);
        if (seg == null) return null;
        
        SafetensorTensorInfo info = header.findTensor(tensorName).orElse(null);
        if (info == null) return null;
        
        int numElements = (int) info.numElements();
        int[] result = new int[numElements];
        for (int i = 0; i < numElements; i++) {
            result[i] = seg.get(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 
                    (long) i * Integer.BYTES);
        }
        return result;
    }

    public float[] readFp32Tensor(String tensorName) {
        MemorySegment seg = getTensorData(tensorName);
        if (seg == null) return null;
        
        SafetensorTensorInfo info = header.findTensor(tensorName).orElse(null);
        if (info == null) return null;
        
        int numElements = (int) info.numElements();
        float[] result = new float[numElements];
        for (int i = 0; i < numElements; i++) {
            result[i] = seg.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 
                    (long) i * Float.BYTES);
        }
        return result;
    }

    public short[] readFp16Tensor(String tensorName) {
        MemorySegment seg = getTensorData(tensorName);
        if (seg == null) return null;
        
        SafetensorTensorInfo info = header.findTensor(tensorName).orElse(null);
        if (info == null) return null;
        
        int numElements = (int) info.numElements();
        short[] result = new short[numElements];
        for (int i = 0; i < numElements; i++) {
            result[i] = seg.get(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 
                    (long) i * Short.BYTES);
        }
        return result;
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
