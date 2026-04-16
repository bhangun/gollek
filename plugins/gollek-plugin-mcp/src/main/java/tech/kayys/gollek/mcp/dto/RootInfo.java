package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Root information for MCP roots capability.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Roots expose file system roots to MCP servers.
 * Per spec: URI MUST be file:// scheme.
 * Clients MUST prompt users for consent before exposing roots.
 */
public class RootInfo {
    private final String uri;
    private final String name;

    private RootInfo(Builder builder) {
        this.uri = builder.uri;
        this.name = builder.name;
    }

    /**
     * Root URI.
     * Per spec: MUST be file:// URI.
     */
    public String getUri() { return uri; }
    
    /** Optional human-readable name */
    public String getName() { return name; }

    public static Builder builder() { return new Builder(); }

    public static RootInfo fromMap(Map<String, Object> data) {
        if (data == null) return null;
        return new Builder()
                .uri((String) data.get("uri"))
                .name((String) data.get("name"))
                .build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new java.util.HashMap<>();
        map.put("uri", uri);
        if (name != null) map.put("name", name);
        return map;
    }

    public static class Builder {
        private String uri;
        private String name;

        public Builder uri(String uri) { this.uri = uri; return this; }
        public Builder name(String name) { this.name = name; return this; }

        public RootInfo build() {
            if (uri == null) {
                throw new IllegalStateException("Root URI is required");
            }
            return new RootInfo(this);
        }
    }
}
