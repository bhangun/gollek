package tech.kayys.wayang.memory.model;

public class MultiModalMemory {
    private TextualMemory textual;
    private VisualMemory visual;      // For images/diagrams
    private StructuredMemory structured; // For code/tables
    private TemporalMemory temporal;   // For time-series

    // Getters and setters
    public TextualMemory getTextual() { return textual; }
    public void setTextual(TextualMemory textual) { this.textual = textual; }

    public VisualMemory getVisual() { return visual; }
    public void setVisual(VisualMemory visual) { this.visual = visual; }

    public StructuredMemory getStructured() { return structured; }
    public void setStructured(StructuredMemory structured) { this.structured = structured; }

    public TemporalMemory getTemporal() { return temporal; }
    public void setTemporal(TemporalMemory temporal) { this.temporal = temporal; }

    // Inner classes for different modalities
    public static class TextualMemory {
        private String content;
        private String language;
        private double confidence;

        public TextualMemory(String content, String language, double confidence) {
            this.content = content;
            this.language = language;
            this.confidence = confidence;
        }

        // Getters and setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }
    }

    public static class VisualMemory {
        private String imageUrl;
        private String description;
        private String[] tags;

        public VisualMemory(String imageUrl, String description, String[] tags) {
            this.imageUrl = imageUrl;
            this.description = description;
            this.tags = tags;
        }

        // Getters and setters
        public String getImageUrl() { return imageUrl; }
        public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String[] getTags() { return tags; }
        public void setTags(String[] tags) { this.tags = tags; }
    }

    public static class StructuredMemory {
        private String type; // code, table, json, xml, etc.
        private String content;
        private String schema;

        public StructuredMemory(String type, String content, String schema) {
            this.type = type;
            this.content = content;
            this.schema = schema;
        }

        // Getters and setters
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getSchema() { return schema; }
        public void setSchema(String schema) { this.schema = schema; }
    }

    public static class TemporalMemory {
        private java.time.Instant startTime;
        private java.time.Instant endTime;
        private String pattern;

        public TemporalMemory(java.time.Instant startTime, java.time.Instant endTime, String pattern) {
            this.startTime = startTime;
            this.endTime = endTime;
            this.pattern = pattern;
        }

        // Getters and setters
        public java.time.Instant getStartTime() { return startTime; }
        public void setStartTime(java.time.Instant startTime) { this.startTime = startTime; }
        public java.time.Instant getEndTime() { return endTime; }
        public void setEndTime(java.time.Instant endTime) { this.endTime = endTime; }
        public String getPattern() { return pattern; }
        public void setPattern(String pattern) { this.pattern = pattern; }
    }
}