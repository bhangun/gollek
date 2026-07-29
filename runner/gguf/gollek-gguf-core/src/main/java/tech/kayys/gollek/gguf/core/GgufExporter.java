package tech.kayys.gollek.gguf.core;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.nn.Module;
import tech.kayys.gollek.gguf.writer.GGUFWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Exports a Gollek {@link Module} to GGUF format for llama.cpp
 *
 * <p>
 * Quantization is applied tensor-by-tensor before writing:
 * <ul>
 * <li>{@code NONE / FP16} — stored as F32 (FP16 is a runtime concern in
 * llama.cpp)</li>
 * <li>{@code INT8} — symmetric per-tensor: {@code scale = max(|x|) / 127}</li>
 * <li>{@code INT4 / NF4} — block-wise Q4_0 (32-element blocks):
 * {@code scale = max(|x|) / 7}</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * 
 * <pre>{@code
 * GgufExporter.fromModel(model)
 *         .quantization(Quantization.INT8)
 *         .export(Path.of("model.gguf"));
 * }</pre>
 */
public final class GgufExporter {

    /** GGUF file_type values matching llama.cpp conventions. */
    public enum Quantization {
        NONE(0, GGUFWriter.TensorEncoding.F32),
        FP16(1, GGUFWriter.TensorEncoding.F16),
        INT8(7, GGUFWriter.TensorEncoding.Q8_0),
        INT4(2, GGUFWriter.TensorEncoding.Q4_0),
        NF4(20, null);

        public final long fileType;
        private final GGUFWriter.TensorEncoding tensorEncoding;

        Quantization(long fileType, GGUFWriter.TensorEncoding tensorEncoding) {
            this.fileType = fileType;
            this.tensorEncoding = tensorEncoding;
        }
    }

    private static final Set<String> RESERVED_METADATA_KEYS = Set.of(
            "architecture",
            "general.alignment",
            "general.architecture",
            "general.file_type",
            "general.parameter_count",
            "general.quantization_version");

    private final Module model;
    private final Map<String, Object> metadata;
    private Quantization quantization = Quantization.NONE;

    private GgufExporter(Module model, Map<String, Object> metadata) {
        this.model = model;
        this.metadata = metadata != null ? metadata : Map.of();
    }

    public static GgufExporter fromModel(Module model) {
        return new GgufExporter(model, Map.of());
    }

    public static GgufExporter fromModel(Module model, Map<String, Object> metadata) {
        return new GgufExporter(model, metadata);
    }

    public GgufExporter quantization(Quantization q) {
        this.quantization = q;
        return this;
    }

    public void export(Path outputPath) throws IOException {
        if (quantization == Quantization.NF4) {
            throw new UnsupportedOperationException(
                    "NF4 GGUF export requires IQ4_NL tensor packing; use INT4 for Q4_0 GGUF export.");
        }

        Map<String, Tensor> stateDict = model.namedParameters();
        Map<String, GgufMetaValue> ggufMeta = ggufMetadata();
        GGUFWriter.save(outputPath, stateDict, ggufMeta, quantization.tensorEncoding);
    }

    private Map<String, GgufMetaValue> ggufMetadata() {
        Map<String, GgufMetaValue> ggufMeta = new LinkedHashMap<>();
        ggufMeta.put("general.architecture",
                GgufMetaValue.ofString(metadata.getOrDefault("architecture", "gollek").toString()));
        ggufMeta.put("general.alignment", GgufMetaValue.ofUInt32(GgufModel.DEFAULT_ALIGNMENT));
        ggufMeta.put("general.quantization_version", GgufMetaValue.ofUInt32(2));
        ggufMeta.put("general.file_type", GgufMetaValue.ofUInt32(quantization.fileType));
        ggufMeta.put("general.parameter_count", GgufMetaValue.ofUInt64(model.parameterCount()));
        for (var e : metadata.entrySet()) {
            if (!RESERVED_METADATA_KEYS.contains(e.getKey())) {
                ggufMeta.put(e.getKey(), toMetadataValue(e.getValue()));
            }
        }
        return ggufMeta;
    }

    private static GgufMetaValue toMetadataValue(Object value) {
        if (value instanceof GgufMetaValue ggufValue) {
            return ggufValue;
        }
        if (value instanceof String string) {
            return GgufMetaValue.ofString(string);
        }
        if (value instanceof Boolean bool) {
            return GgufMetaValue.ofBool(bool);
        }
        if (value instanceof Float flt) {
            return GgufMetaValue.ofFloat32(flt);
        }
        if (value instanceof Double dbl) {
            return new GgufMetaValue.Float64Val(dbl);
        }
        if (value instanceof Byte b) {
            return new GgufMetaValue.Int8Val(b);
        }
        if (value instanceof Short s) {
            return new GgufMetaValue.Int16Val(s);
        }
        if (value instanceof Integer i) {
            return GgufMetaValue.ofInt32(i);
        }
        if (value instanceof Long l) {
            return l >= 0 ? GgufMetaValue.ofUInt64(l) : new GgufMetaValue.Int64Val(l);
        }
        if (value instanceof Iterable<?> iterable) {
            return toArrayMetadataValue(iterable);
        }
        return GgufMetaValue.ofString(String.valueOf(value));
    }

    private static GgufMetaValue toArrayMetadataValue(Iterable<?> iterable) {
        List<Object> values = new ArrayList<>();
        for (Object value : iterable) {
            values.add(value);
        }
        if (values.isEmpty()) {
            return GgufMetaValue.ofStringArray(List.of());
        }
        if (values.stream().allMatch(String.class::isInstance)) {
            return GgufMetaValue.ofStringArray(values.stream().map(String.class::cast).toList());
        }
        if (values.stream().allMatch(Integer.class::isInstance)) {
            return GgufMetaValue.ofInt32Array(values.stream().map(Integer.class::cast).toList());
        }
        if (values.stream().allMatch(Float.class::isInstance)) {
            return GgufMetaValue.ofFloat32Array(values.stream().map(Float.class::cast).toList());
        }
        if (values.stream().allMatch(Boolean.class::isInstance)) {
            return GgufMetaValue.ofBoolArray(values.stream().map(Boolean.class::cast).toList());
        }

        List<String> strings = new ArrayList<>(values.size());
        for (Object value : values) {
            strings.add(String.valueOf(value));
        }
        return GgufMetaValue.ofStringArray(strings);
    }
}
