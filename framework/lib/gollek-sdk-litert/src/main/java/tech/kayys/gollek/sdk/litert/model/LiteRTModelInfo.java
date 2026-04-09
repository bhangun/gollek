package tech.kayys.gollek.sdk.litert.model;

import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * LiteRT Model information.
 */
@Data
@Builder
public class LiteRTModelInfo {

    /**
     * Model identifier.
     */
    private String modelId;

    /**
     * Model file path or name.
     */
    private String modelPath;

    /**
     * Model file size in bytes.
     */
    private long modelSizeBytes;

    /**
     * Input tensor information.
     */
    @Builder.Default
    private List<TensorInfo> inputs = List.of();

    /**
     * Output tensor information.
     */
    @Builder.Default
    private List<TensorInfo> outputs = List.of();

    /**
     * Model metadata.
     */
    @Builder.Default
    private Map<String, String> metadata = Map.of();

    /**
     * Tensor information.
     */
    @Data
    @Builder
    public static class TensorInfo {
        private String name;
        private int[] shape;
        private String dataType;
        private long byteSize;
    }
}
