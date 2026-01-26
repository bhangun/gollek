package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * Represents an MCP resource (file, data source, external content).
 * Immutable and serializable.
 */
public final class MCPResource {

    @NotBlank
    private final String uri;

    private final String name;
    private final String description;
    private final String mimeType;
    private final Map<String, Object> metadata;

    @JsonCreator
    public MCPResource(
            @JsonProperty("uri") String uri,
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("mimeType") String mimeType,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.uri = Objects.requireNonNull(uri, "uri");
        this.name = name;
        this.description = description;
        this.mimeType = mimeType;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Getters
    public String getUri() {
        return uri;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getMimeType() {
        return mimeType;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Create from MCP protocol map
     */
    public static MCPResource fromMap(Map<String, Object> data) {
        return new MCPResource(
                (String) data.get("uri"),
                (String) data.get("name"),
                (String) data.get("description"),
                (String) data.get("mimeType"),
                (Map<String, Object>) data.get("metadata"));
    }

    /**
     * Check if resource is text-based
     */
    public boolean isTextResource() {
        return mimeType != null && (mimeType.startsWith("text/") ||
                mimeType.equals("application/json") ||
                mimeType.equals("application/xml"));
    }

    /**
     * Check if resource is binary
     */
    public boolean isBinaryResource() {
        return mimeType != null && (mimeType.startsWith("image/") ||
                mimeType.startsWith("video/") ||
                mimeType.startsWith("audio/") ||
                mimeType.equals("application/octet-stream"));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof MCPResource that))
            return false;
        return uri.equals(that.uri);
    }

    @Override
    public int hashCode() {
        return Objects.hash(uri);
    }

    @Override
    public String toString() {
        return "MCPResource{uri='" + uri + "', name='" + name + "', mimeType='" + mimeType + "'}";
    }
}