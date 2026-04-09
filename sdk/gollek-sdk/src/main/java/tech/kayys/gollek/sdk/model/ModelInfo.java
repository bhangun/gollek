package tech.kayys.gollek.sdk.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.Collections;
import java.util.Map;

/**
 * Metadata and capability descriptor for a Gollek model.
 *
 * @param id               unique model identifier (e.g. {@code "llama3-8b-q4"})
 * @param format           model file format (e.g. {@code "GGUF"}, {@code "SAFETENSORS"})
 * @param contextWindow    maximum number of tokens the model can process in one call
 * @param supportsStreaming whether the model supports incremental token streaming
 * @param supportsTools    whether the model supports function/tool calling
 * @param capabilities     additional provider-specific capability flags
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonDeserialize(builder = ModelInfo.Builder.class)
public record ModelInfo(
        String id,
        String format,
        int contextWindow,
        boolean supportsStreaming,
        boolean supportsTools,
        Map<String, Object> capabilities
) {

    public static Builder builder() {
        return new Builder();
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private String id;
        private String format;
        private int contextWindow;
        private boolean supportsStreaming;
        private boolean supportsTools;
        private Map<String, Object> capabilities = Collections.emptyMap();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder format(String format) {
            this.format = format;
            return this;
        }

        public Builder contextWindow(int contextWindow) {
            this.contextWindow = contextWindow;
            return this;
        }

        public Builder supportsStreaming(boolean supportsStreaming) {
            this.supportsStreaming = supportsStreaming;
            return this;
        }

        public Builder supportsTools(boolean supportsTools) {
            this.supportsTools = supportsTools;
            return this;
        }

        public Builder capabilities(Map<String, Object> capabilities) {
            this.capabilities = capabilities;
            return this;
        }

        public ModelInfo build() {
            return new ModelInfo(id, format, contextWindow, supportsStreaming, supportsTools, capabilities);
        }
    }
}
