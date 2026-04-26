package tech.kayys.gollek.converter.java.gguf;

import java.util.*;

/**
 * In-memory representation of a GGUF model file.
 *
 * <p>
 * This mirrors the {@code gguf_context} struct from the C implementation.
 * It owns:
 * <ul>
 * <li>Ordered metadata key-value pairs.</li>
 * <li>Ordered tensor descriptors.</li>
 * <li>An optional byte array of raw tensor data (null when writing
 * lazily).</li>
 * </ul>
 */
public final class GgufModel {

    // ── Constants ─────────────────────────────────────────────────────────

    /** 4-byte ASCII magic: "GGUF" */
    public static final int MAGIC = 0x46554747; // little-endian "GGUF"
    public static final byte[] MAGIC_BYTES = { 'G', 'G', 'U', 'F' };
    /** Current GGUF specification version. */
    public static final int VERSION = 3;
    /** Default alignment for tensor data (bytes). */
    public static final int DEFAULT_ALIGNMENT = 32;

    // ── Fields ────────────────────────────────────────────────────────────

    private final SequencedMap<String, GgufMetaValue> metadata = new LinkedHashMap<>();
    private final List<TensorInfo> tensors = new ArrayList<>();
    /** Raw concatenated tensor data blob; may be null (lazy / streaming write). */
    private byte[] tensorData;

    // ── Metadata API ──────────────────────────────────────────────────────

    public void addMeta(String key, GgufMetaValue value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        metadata.put(key, value);
    }

    public Optional<GgufMetaValue> getMeta(String key) {
        return Optional.ofNullable(metadata.get(key));
    }

    public SequencedMap<String, GgufMetaValue> metadata() {
        return Collections.unmodifiableSequencedMap(metadata);
    }

    public int alignment() {
        return getMeta("general.alignment")
                .map(GgufMetaValue::asUInt32)
                .map(v -> (int) (long) v)
                .orElse(DEFAULT_ALIGNMENT);
    }

    // ── Tensor API ────────────────────────────────────────────────────────

    public void addTensor(TensorInfo info) {
        tensors.add(Objects.requireNonNull(info, "info"));
    }

    public List<TensorInfo> tensors() {
        return Collections.unmodifiableList(tensors);
    }

    public Optional<TensorInfo> findTensor(String name) {
        return tensors.stream().filter(t -> t.name().equals(name)).findFirst();
    }

    // ── Tensor data blob ─────────────────────────────────────────────────

    public byte[] tensorData() {
        return tensorData;
    }

    public void setTensorData(byte[] data) {
        this.tensorData = data;
    }

    // ── Summary helpers ──────────────────────────────────────────────────

    public String architecture() {
        return getMeta("general.architecture")
                .map(GgufMetaValue::asString)
                .orElse("unknown");
    }

    public String modelName() {
        return getMeta("general.name")
                .map(GgufMetaValue::asString)
                .orElse("unnamed");
    }

    @Override
    public String toString() {
        return "GgufModel{arch='%s', name='%s', kv=%d, tensors=%d}"
                .formatted(architecture(), modelName(), metadata.size(), tensors.size());
    }
}
