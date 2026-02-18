package tech.kayys.gollek.mcp.dto;

import java.util.Base64;

/**
 * Content of an MCP resource.
 */
public class MCPResourceContent {
    private final String uri;
    private final String mimeType;
    private final String text;
    private final String blob; // Base64 encoded binary data

    private MCPResourceContent(Builder builder) {
        this.uri = builder.uri;
        this.mimeType = builder.mimeType;
        this.text = builder.text;
        this.blob = builder.blob;
    }

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

    public String getContentAsString() {
        if (text != null) {
            return text;
        } else if (blob != null) {
            return new String(Base64.getDecoder().decode(blob));
        }
        return "";
    }

    public byte[] getContentAsBytes() {
        if (blob != null) {
            return Base64.getDecoder().decode(blob);
        } else if (text != null) {
            return text.getBytes();
        }
        return new byte[0];
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MCPResourceContent text(String uri, String text) {
        return builder().uri(uri).text(text).build();
    }

    public static MCPResourceContent binary(String uri, byte[] data) {
        return builder().uri(uri).blob(Base64.getEncoder().encodeToString(data)).build();
    }

    public static class Builder {
        private String uri;
        private String mimeType;
        private String text;
        private String blob;

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

        public MCPResourceContent build() {
            return new MCPResourceContent(this);
        }
    }
}