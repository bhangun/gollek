package tech.kayys.gollek.multimodal.model;

import java.util.Map;

/**
 * Represents the response from a multimodal inference request, containing
 * one or more {@link MultimodalContent} output items.
 *
 * @see MultimodalRequest
 * @see MultimodalContent
 */
public class MultimodalResponse {

    private String requestId;
    private MultimodalContent[] outputs = new MultimodalContent[0];
    private Map<String, Object> metadata;

    /**
     * Returns the request identifier that produced this response.
     *
     * @return request ID, or {@code null} if not set
     */
    public String getRequestId() { return requestId; }

    /**
     * Returns the array of generated output content items.
     *
     * @return non-null array of outputs (may be empty)
     */
    public MultimodalContent[] getOutputs() { return outputs; }

    /**
     * Returns provider-specific metadata about the inference (e.g. token counts, latency).
     *
     * @return metadata map, or {@code null}
     */
    public Map<String, Object> getMetadata() { return metadata; }

    /**
     * Creates a new builder for {@link MultimodalResponse}.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link MultimodalResponse}.
     */
    public static class Builder {
        private final MultimodalResponse response = new MultimodalResponse();

        /** @param requestId the originating request identifier */
        public Builder requestId(String requestId) { response.requestId = requestId; return this; }

        /** @param outputs generated content items; {@code null} is treated as empty */
        public Builder outputs(MultimodalContent... outputs) {
            response.outputs = outputs == null ? new MultimodalContent[0] : outputs;
            return this;
        }

        /** @param metadata provider-specific inference metadata */
        public Builder metadata(Map<String, Object> metadata) { response.metadata = metadata; return this; }

        /**
         * Builds the {@link MultimodalResponse}.
         *
         * @return a configured {@link MultimodalResponse}
         */
        public MultimodalResponse build() { return response; }
    }
}
