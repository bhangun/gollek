package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Represents an MCP resource.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Resources provide context and data to AI assistants.
 * Per spec: MUST NOT transmit resource data externally without explicit user consent.
 */
public class MCPResource {
    private final String uri;
    private final String name;
    private final String description;
    private final String mimeType;
    private final List<ResourceIcon> icons;

    private MCPResource(Builder builder) {
        this.uri = builder.uri;
        this.name = builder.name;
        this.description = builder.description;
        this.mimeType = builder.mimeType;
        this.icons = builder.icons;
    }

    /** Unique identifier for the resource */
    public String getUri() { return uri; }
    
    /** Human-readable name */
    public String getName() { return name; }
    
    /** Optional description */
    public String getDescription() { return description; }
    
    /** Optional MIME type hint */
    public String getMimeType() { return mimeType; }
    
    /** Optional resource icons */
    public List<ResourceIcon> getIcons() { return icons; }

    public static Builder builder() {
        return new Builder();
    }

    @SuppressWarnings("unchecked")
    public static MCPResource fromMap(Map<String, Object> data) {
        Builder builder = new Builder()
                .uri((String) data.get("uri"))
                .name((String) data.get("name"))
                .description((String) data.get("description"))
                .mimeType((String) data.get("mimeType"));
        
        // Parse icons if present
        if (data.containsKey("icons")) {
            List<Map<String, Object>> iconsData = (List<Map<String, Object>>) data.get("icons");
            builder.icons(iconsData.stream()
                    .map(ResourceIcon::fromMap)
                    .toList());
        }
        
        return builder.build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new java.util.HashMap<>();
        map.put("uri", uri);
        map.put("name", name);
        if (description != null) map.put("description", description);
        if (mimeType != null) map.put("mimeType", mimeType);
        if (icons != null && !icons.isEmpty()) {
            map.put("icons", icons.stream().map(ResourceIcon::toMap).toList());
        }
        return map;
    }

    /**
     * Resource icon definition.
     * Per spec: src MUST use https: or data: schemes.
     * Clients MUST support image/png and image/jpeg.
     */
    public static class ResourceIcon {
        private final String src;
        private final String mimeType;
        private final List<String> sizes;
        private final String theme;

        private ResourceIcon(Builder builder) {
            this.src = builder.src;
            this.mimeType = builder.mimeType;
            this.sizes = builder.sizes;
            this.theme = builder.theme;
        }

        /** Icon source URI (https: or data: scheme) */
        public String getSrc() { return src; }
        
        /** Icon MIME type */
        public String getMimeType() { return mimeType; }
        
        /** Icon sizes (e.g., ["16x16", "32x32"]) */
        public List<String> getSizes() { return sizes; }
        
        /** Icon theme: "light" | "dark" */
        public String getTheme() { return theme; }

        public static Builder builder() { return new Builder(); }

        @SuppressWarnings("unchecked")
        public static ResourceIcon fromMap(Map<String, Object> data) {
            if (data == null) return null;
            Builder builder = new Builder()
                    .src((String) data.get("src"))
                    .mimeType((String) data.get("mimeType"))
                    .theme((String) data.get("theme"));
            if (data.containsKey("sizes")) {
                builder.sizes((List<String>) data.get("sizes"));
            }
            return builder.build();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new java.util.HashMap<>();
            map.put("src", src);
            if (mimeType != null) map.put("mimeType", mimeType);
            if (sizes != null && !sizes.isEmpty()) map.put("sizes", sizes);
            if (theme != null) map.put("theme", theme);
            return map;
        }

        public static class Builder {
            private String src;
            private String mimeType;
            private List<String> sizes;
            private String theme;

            public Builder src(String src) { this.src = src; return this; }
            public Builder mimeType(String mimeType) { this.mimeType = mimeType; return this; }
            public Builder sizes(List<String> sizes) { this.sizes = sizes; return this; }
            public Builder theme(String theme) { this.theme = theme; return this; }

            public ResourceIcon build() { return new ResourceIcon(this); }
        }
    }

    public static class Builder {
        private String uri;
        private String name;
        private String description;
        private String mimeType;
        private List<ResourceIcon> icons;

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder mimeType(String mimeType) {
            this.mimeType = mimeType;
            return this;
        }

        public Builder icons(List<ResourceIcon> icons) {
            this.icons = icons;
            return this;
        }

        public MCPResource build() {
            return new MCPResource(this);
        }
    }
}