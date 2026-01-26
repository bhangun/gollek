package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.*;

/**
 * Content of a read MCP resource.
 * Supports both text and binary data.
 */
public final class MCPResourceContent {

    private final String uri;
    private final String mimeType;
    private final String text;
    private final String blob; // base64 encoded binary
    private final Map<String, Object> metadata;
    private final Instant timestamp;

    @JsonCreator
    public MCPResourceContent(
            @JsonProperty("uri") String uri,
            @JsonProperty("mimeType") String mimeType,
            @JsonProperty("text") String text,
            @JsonProperty("blob") String blob,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("timestamp") Instant timestamp) {
        this.uri = Objects.requireNonNull(uri, "uri");
        this.mimeType = mimeType;
        this.text = text;
        this.blob = blob;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    // Getters
    public String getUri() {
        return uri;
    }

    public String getMimeType() {
        return mimeType;
    }

    public String getText() {
        return text;
    }

    public String getBlob() {
        return blob;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    /**
     * Check if content is text
     */
    public boolean isText() {
        return text != null;
    }

    /**
     * Check if content is binary
     */
    public boolean isBinary() {
        return blob != null;
    }

    /**
     * Get content as string (text or decoded blob)
     */
    public String getContentAsString() {
        if (text != null) {
            return text;
        }
        if (blob != null) {
            // Could decode base64 here if needed
            return "[Binary content: " + blob.length() + " bytes]";
        }
        return "";
    }

    // Factory methods
    public static Builder builder() {
        return new Builder();
    }

    public static MCPResourceContent text(String uri, String text) {
        return builder()
                .uri(uri)
                .mimeType("text/plain")
                .text(text)
                .build();
    }

    public static MCPResourceContent binary(String uri, String blob, String mimeType) {
        return builder()
                .uri(uri)
                .mimeType(mimeType)
                .blob(blob)
                .build();
    }

    public static class Builder {
        private String uri;
        private String mimeType;
        private String text;
        private String blob;
        private final Map<String, Object> metadata = new HashMap<>();
        private Instant timestamp = Instant.now();

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public Builder mimeType(String mimeType) {
            this.mimeType = mimeType;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder blob(String blob) {
            this.blob = blob;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public MCPResourceContent build() {
            Objects.requireNonNull(uri, "uri is required");
            return new MCPResourceContent(uri, mimeType, text, blob, metadata, timestamp);
        }
    }

    @Override
    public String toString() {
        return "MCPResourceContent{" +
                "uri='" + uri + '\'' +
                ", mimeType='" + mimeType + '\'' +
                ", isText=" + isText() +
                ", isBinary=" + isBinary() +
                '}';
    }
}