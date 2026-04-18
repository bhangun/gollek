package tech.kayys.gollek.gguf.loader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

/**
 * Representation of a parsed GGUF model in memory.
 */
public final class GGUFModel implements AutoCloseable {
    private final int version;
    private final Map<String, Object> metadata;
    private final List<GGUFTensorInfo> tensors;
    private final long dataStart;
    private final MemorySegment segment;
    private final Arena arena;

    public GGUFModel(int version, Map<String, Object> metadata, List<GGUFTensorInfo> tensors, long dataStart, MemorySegment segment, Arena arena) {
        this.version = version;
        this.metadata = metadata;
        this.tensors = tensors;
        this.dataStart = dataStart;
        this.segment = segment;
        this.arena = arena;
    }

    public int version() { return version; }
    public Map<String, Object> metadata() { return metadata; }
    public List<GGUFTensorInfo> tensors() { return tensors; }
    public long dataStart() { return dataStart; }
    public MemorySegment segment() { return segment; }

    @Override
    public void close() {
        if (arena != null) {
            arena.close();
        }
    }
}
