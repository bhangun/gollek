package tech.kayys.golek.model;


import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public record ModelInfo(
    String name,
    String description,
    String modelType,
    String architecture,
    long parameterCount,
    String quantization,
    int contextLength,
    int vocabSize,
    long fileSize,
    Map<String, Object> metadata
) {
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private String name;
        private String description;
        private String modelType;
        private String architecture;
        private long parameterCount;
        private String quantization;
        private int contextLength;
        private int vocabSize;
        private long fileSize;
        private Map<String, Object> metadata = new HashMap<>();
        
        public Builder name(String name) { this.name = name; return this; }
        public Builder description(String desc) { this.description = desc; return this; }
        public Builder modelType(String type) { this.modelType = type; return this; }
        public Builder architecture(String arch) { this.architecture = arch; return this; }
        public Builder parameterCount(long count) { this.parameterCount = count; return this; }
        public Builder quantization(String quant) { this.quantization = quant; return this; }
        public Builder contextLength(int len) { this.contextLength = len; return this; }
        public Builder vocabSize(int size) { this.vocabSize = size; return this; }
        public Builder fileSize(long size) { this.fileSize = size; return this; }
        public Builder metadata(Map<String, Object> meta) { this.metadata = meta; return this; }
        public Builder addMetadata(String key, Object value) { this.metadata.put(key, value); return this; }
        
        public ModelInfo build() {
            return new ModelInfo(name, description, modelType, architecture,
                parameterCount, quantization, contextLength, vocabSize, fileSize, 
                Collections.unmodifiableMap(new HashMap<>(metadata)));
        }
    }
    
    // Core validation
    public void validate() {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalStateException("Model name cannot be null or empty");
        }
        if (vocabSize <= 0) {
            throw new IllegalStateException("Vocabulary size must be positive");
        }
    }
}