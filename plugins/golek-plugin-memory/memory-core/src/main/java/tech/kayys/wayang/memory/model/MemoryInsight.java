package tech.kayys.wayang.memory.model;

import java.util.Map;

public class MemoryInsight {
    private final String type;
    private final String description;
    private final double confidence;
    private final Map<String, Object> metadata;

    public MemoryInsight(
            String type,
            String description,
            double confidence,
            Map<String, Object> metadata) {
        this.type = type;
        this.description = description;
        this.confidence = confidence;
        this.metadata = metadata;
    }

    public String getType() { return type; }
    public String getDescription() { return description; }
    public double getConfidence() { return confidence; }
    public Map<String, Object> getMetadata() { return metadata; }
}