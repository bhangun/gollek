package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Represents an MCP content block.
 * Compliant with MCP specification 2025-11-25.
 * 
 * <p>Content blocks can be:
 * <ul>
 *   <li>text: plain text content</li>
 *   <li>image: base64-encoded image data</li>
 *   <li>audio: base64-encoded audio data</li>
 *   <li>resource: embedded resource content</li>
 * </ul>
 */
public class MCPContentBlock {
    
    public enum ContentType {
        TEXT("text"),
        IMAGE("image"),
        AUDIO("audio"),
        RESOURCE("resource");
        
        private final String value;
        ContentType(String value) { this.value = value; }
        public String getValue() { return value; }
    }
    
    private final ContentType type;
    private final String text;          // for type=TEXT
    private final String data;          // base64 for type=IMAGE or AUDIO
    private final String mimeType;      // for type=IMAGE, AUDIO, or RESOURCE
    private final ResourceContent resource;  // for type=RESOURCE
    private final String id;            // optional identifier for type=AUDIO

    private MCPContentBlock(Builder builder) {
        this.type = builder.type;
        this.text = builder.text;
        this.data = builder.data;
        this.mimeType = builder.mimeType;
        this.resource = builder.resource;
        this.id = builder.id;
    }

    public ContentType getType() { return type; }
    public String getText() { return text; }
    public String getData() { return data; }
    public String getMimeType() { return mimeType; }
    public ResourceContent getResource() { return resource; }
    public String getId() { return id; }

    /**
     * Check if this is a text content block.
     */
    public boolean isText() {
        return type == ContentType.TEXT;
    }

    /**
     * Get text content, or null if not text type.
     */
    public String getTextOrNull() {
        return isText() ? text : null;
    }

    /**
     * Check if this is an image content block.
     */
    public boolean isImage() {
        return type == ContentType.IMAGE;
    }

    /**
     * Check if this is a resource content block.
     */
    public boolean isResource() {
        return type == ContentType.RESOURCE;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a text content block.
     */
    public static MCPContentBlock text(String text) {
        return builder().type(ContentType.TEXT).text(text).build();
    }

    /**
     * Create an image content block.
     */
    public static MCPContentBlock image(String base64Data, String mimeType) {
        return builder().type(ContentType.IMAGE).data(base64Data).mimeType(mimeType).build();
    }

    @SuppressWarnings("unchecked")
    public static MCPContentBlock fromMap(Map<String, Object> data) {
        if (data == null) return null;
        
        String typeStr = (String) data.get("type");
        Builder builder = new Builder();
        
        if (typeStr != null) {
            builder.type(ContentType.valueOf(typeStr.toUpperCase()));
        }
        
        builder.text((String) data.get("text"))
               .data((String) data.get("data"))
               .mimeType((String) data.get("mimeType"))
               .id((String) data.get("id"));
        
        if (data.containsKey("resource")) {
            Map<String, Object> resourceData = (Map<String, Object>) data.get("resource");
            builder.resource(ResourceContent.fromMap(resourceData));
        }
        
        return builder.build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new java.util.HashMap<>();
        map.put("type", type.getValue());
        if (text != null) map.put("text", text);
        if (data != null) map.put("data", data);
        if (mimeType != null) map.put("mimeType", mimeType);
        if (resource != null) map.put("resource", resource.toMap());
        if (id != null) map.put("id", id);
        return map;
    }

    /**
     * Embedded resource content.
     * Per spec: exactly one of text or blob must be present.
     */
    public static class ResourceContent {
        private final String uri;
        private final String mimeType;
        private final String text;      // text content
        private final String blob;      // base64-encoded binary content

        private ResourceContent(Builder builder) {
            this.uri = builder.uri;
            this.mimeType = builder.mimeType;
            this.text = builder.text;
            this.blob = builder.blob;
        }

        public String getUri() { return uri; }
        public String getMimeType() { return mimeType; }
        public String getText() { return text; }
        public String getBlob() { return blob; }

        public static Builder builder() { return new Builder(); }

        @SuppressWarnings("unchecked")
        public static ResourceContent fromMap(Map<String, Object> data) {
            if (data == null) return null;
            return new Builder()
                    .uri((String) data.get("uri"))
                    .mimeType((String) data.get("mimeType"))
                    .text((String) data.get("text"))
                    .blob((String) data.get("blob"))
                    .build();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new java.util.HashMap<>();
            map.put("uri", uri);
            if (mimeType != null) map.put("mimeType", mimeType);
            if (text != null) map.put("text", text);
            if (blob != null) map.put("blob", blob);
            return map;
        }

        public static class Builder {
            private String uri;
            private String mimeType;
            private String text;
            private String blob;

            public Builder uri(String uri) { this.uri = uri; return this; }
            public Builder mimeType(String mimeType) { this.mimeType = mimeType; return this; }
            public Builder text(String text) { this.text = text; return this; }
            public Builder blob(String blob) { this.blob = blob; return this; }

            public ResourceContent build() { return new ResourceContent(this); }
        }
    }

    public static class Builder {
        private ContentType type;
        private String text;
        private String data;
        private String mimeType;
        private ResourceContent resource;
        private String id;

        public Builder type(ContentType type) { this.type = type; return this; }
        public Builder text(String text) { this.text = text; return this; }
        public Builder data(String data) { this.data = data; return this; }
        public Builder mimeType(String mimeType) { this.mimeType = mimeType; return this; }
        public Builder resource(ResourceContent resource) { this.resource = resource; return this; }
        public Builder id(String id) { this.id = id; return this; }

        public MCPContentBlock build() {
            if (type == null) {
                throw new IllegalStateException("Content block type is required");
            }
            return new MCPContentBlock(this);
        }
    }
}
