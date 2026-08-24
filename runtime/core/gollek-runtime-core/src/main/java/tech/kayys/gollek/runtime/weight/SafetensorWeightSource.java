package tech.kayys.gollek.runtime.weight;


import tech.kayys.alkhawarizm.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.alkhawarizm.core.memory.CpuBuffer;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.*;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * A weight source that loads weights from a Safetensors file.
 */
public final class SafetensorWeightSource implements WeightSource {
    private final Path filePath;
    private final ComputeBackend backend;
    private final Arena arena;
    private final Map<String, TensorInfo> tensorInfos = new HashMap<>();
    private final MemorySegment dataSegment;

    public SafetensorWeightSource(Path filePath, ComputeBackend backend) throws IOException {
        this.filePath = filePath;
        this.backend = backend;
        this.arena = Arena.ofShared();

        try (RandomAccessFile raf = new RandomAccessFile(filePath.toFile(), "r")) {
            FileChannel channel = raf.getChannel();
            long fileSize = channel.size();

            // Read header length (first 8 bytes, little endian)
            MemorySegment lenSeg = channel.map(FileChannel.MapMode.READ_ONLY, 0, 8, arena);
            long headerLen = lenSeg.get(ValueLayout.JAVA_LONG_UNALIGNED, 0);

            // Read header JSON
            MemorySegment headerSeg = channel.map(FileChannel.MapMode.READ_ONLY, 8, headerLen, arena);
            byte[] headerBytes = headerSeg.toArray(ValueLayout.JAVA_BYTE);
            
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> root = mapper.readValue(headerBytes, Map.class);
            
            parseHeader(root);
            
            long dataOffset = 8 + headerLen;
            this.dataSegment = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, fileSize - dataOffset, arena);
        }
    }

    private void parseHeader(Map<String, Object> root) {
        for (var entry : root.entrySet()) {
            if ("__metadata__".equals(entry.getKey())) continue;
            
            if (entry.getValue() instanceof Map<?, ?> map) {
                String dtypeStr = (String) map.get("dtype");
                List<Number> shapeList = (List<Number>) map.get("shape");
                List<Number> offsets = (List<Number>) map.get("data_offsets");
                
                if (dtypeStr != null && shapeList != null && offsets != null && offsets.size() == 2) {
                    long[] shape = shapeList.stream().mapToLong(Number::longValue).toArray();
                    long start = offsets.get(0).longValue();
                    long end = offsets.get(1).longValue();
                    tensorInfos.put(entry.getKey(), new TensorInfo(dtypeStr, shape, start, end));
                }
            }
        }
    }

    @Override
    public Tensor get(String key) {
        TensorInfo info = tensorInfos.get(key);
        if (info == null) return null;

        DType dtype = mapDType(info.dtype);
        MemorySegment slice = dataSegment.asSlice(info.start, info.end - info.start);
        
        // Wrap in CpuBuffer. Note: We use the shared arena from the source.
        // The tensor's release() will NOT close the arena, but it's part of the source's lifecycle.
        // Actually, to be safe, we might want to use a separate arena or reference count.
        // For simplicity, we assume the source stays open as long as tensors are needed.
        CpuBuffer buffer = new CpuBuffer(slice, arena);
        
        return new DefaultTensor(new Shape(info.shape), dtype, DeviceType.CPU, buffer, backend);
    }

    @Override
    public boolean contains(String key) {
        return tensorInfos.containsKey(key);
    }

    @Override
    public Set<String> keys() {
        return Collections.unmodifiableSet(tensorInfos.keySet());
    }

    @Override
    public void close() throws Exception {
        arena.close();
    }

    private DType mapDType(String sfDType) {
        return switch (sfDType.toUpperCase()) {
            case "F32" -> DType.F32;
            case "F16" -> DType.F16;
            case "BF16" -> DType.BF16;
            case "I32" -> DType.I32;
            case "I8" -> DType.I8;
            default -> throw new InferenceException(ErrorCode.CONFIG_UNSUPPORTED, "Unsupported Safetensor dtype: " + sfDType);
        };
    }

    private record TensorInfo(String dtype, long[] shape, long start, long end) {}
}
