package tech.kayys.gollek.sdk.multimodal;

import tech.kayys.gollek.multimodal.model.ModalityType;
import tech.kayys.gollek.multimodal.model.MultimodalContent;
import tech.kayys.gollek.multimodal.model.MultimodalRequest;
import tech.kayys.gollek.multimodal.model.MultimodalResponse;

import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Fluent builder for constructing {@link MultimodalRequest} instances.
 *
 * <p>Provides convenience methods for appending inputs of different modalities
 * and configuring generation parameters, without requiring direct interaction
 * with the lower-level {@link MultimodalRequest.Builder}.
 *
 * <pre>{@code
 * MultimodalRequest request = new MultimodalRequestBuilder()
 *     .model("qwen-vl")
 *     .addText("What is in this image?")
 *     .addImage(imageBytes, "image/jpeg")
 *     .maxTokens(512)
 *     .temperature(0.7)
 *     .build();
 * }</pre>
 *
 * @see MultimodalRequest
 * @see MultimodalContent
 */
public class MultimodalRequestBuilder {

    private final MultimodalRequest.Builder builder = MultimodalRequest.builder();
    private final MultimodalRequest.OutputConfig.Builder configBuilder =
        MultimodalRequest.OutputConfig.builder();

    /**
     * Sets the model identifier to use for inference.
     *
     * @param model model identifier
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder model(String model) {
        builder.model(model);
        return this;
    }

    /**
     * Sets the unique request identifier.
     *
     * @param requestId request ID
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder requestId(String requestId) {
        builder.requestId(requestId);
        return this;
    }

    /**
     * Appends a plain text input to the request.
     *
     * @param text the text content to add
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder addText(String text) {
        MultimodalRequest current = builder.build();
        MultimodalContent[] inputs = current.getInputs();
        MultimodalContent[] newInputs = new MultimodalContent[inputs.length + 1];
        System.arraycopy(inputs, 0, newInputs, 0, inputs.length);
        newInputs[inputs.length] = MultimodalContent.ofText(text);
        builder.inputs(newInputs);
        return this;
    }

    /**
     * Appends an image input from raw bytes (Base64-encoded internally).
     *
     * @param imageBytes raw image bytes
     * @param mimeType   MIME type (e.g. {@code "image/jpeg"})
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder addImage(byte[] imageBytes, String mimeType) {
        MultimodalRequest current = builder.build();
        MultimodalContent[] inputs = current.getInputs();
        MultimodalContent[] newInputs = new MultimodalContent[inputs.length + 1];
        System.arraycopy(inputs, 0, newInputs, 0, inputs.length);
        newInputs[inputs.length] = MultimodalContent.ofBase64Image(imageBytes, mimeType);
        builder.inputs(newInputs);
        return this;
    }

    /**
     * Appends an image input from a pre-encoded Base64 string.
     *
     * @param base64Data Base64-encoded image data
     * @param mimeType   MIME type (e.g. {@code "image/png"})
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder addImageBase64(String base64Data, String mimeType) {
        MultimodalRequest current = builder.build();
        MultimodalContent[] inputs = current.getInputs();
        MultimodalContent[] newInputs = new MultimodalContent[inputs.length + 1];
        System.arraycopy(inputs, 0, newInputs, 0, inputs.length);
        newInputs[inputs.length] = MultimodalContent.builder(ModalityType.IMAGE)
            .base64Data(base64Data)
            .mimeType(mimeType)
            .build();
        builder.inputs(newInputs);
        return this;
    }

    /**
     * Appends an image input from a URI (URL or local file path).
     *
     * @param uri      URI pointing to the image
     * @param mimeType MIME type of the image
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder addImageUri(String uri, String mimeType) {
        MultimodalRequest current = builder.build();
        MultimodalContent[] inputs = current.getInputs();
        MultimodalContent[] newInputs = new MultimodalContent[inputs.length + 1];
        System.arraycopy(inputs, 0, newInputs, 0, inputs.length);
        newInputs[inputs.length] = MultimodalContent.ofImageUri(uri, mimeType);
        builder.inputs(newInputs);
        return this;
    }

    /**
     * Appends a document input from raw bytes (Base64-encoded internally).
     *
     * @param documentBytes raw document bytes
     * @param format        format identifier (e.g. {@code "pdf"}, {@code "docx"})
     * @param mimeType      MIME type (e.g. {@code "application/pdf"})
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder addDocument(byte[] documentBytes, String format, String mimeType) {
        MultimodalRequest current = builder.build();
        MultimodalContent[] inputs = current.getInputs();
        MultimodalContent[] newInputs = new MultimodalContent[inputs.length + 1];
        System.arraycopy(inputs, 0, newInputs, 0, inputs.length);
        newInputs[inputs.length] = MultimodalContent.ofDocument(documentBytes, format, mimeType);
        builder.inputs(newInputs);
        return this;
    }

    /**
     * Applies a lambda to an {@link OutputConfigBuilder} for inline output configuration.
     *
     * @param configurer consumer that configures the output
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder outputConfig(Consumer<OutputConfigBuilder> configurer) {
        OutputConfigBuilder ocb = new OutputConfigBuilder();
        configurer.accept(ocb);
        builder.outputConfig(ocb.build());
        return this;
    }

    /**
     * Sets the maximum number of tokens to generate.
     *
     * @param maxTokens token limit; must be &gt; 0
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder maxTokens(int maxTokens) {
        configBuilder.maxTokens(maxTokens);
        return this;
    }

    /**
     * Sets the sampling temperature.
     *
     * @param temperature value in range {@code [0.0, 2.0]}
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder temperature(double temperature) {
        configBuilder.temperature(temperature);
        return this;
    }

    /**
     * Sets the nucleus sampling probability threshold.
     *
     * @param topP value in range {@code (0.0, 1.0]}
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder topP(double topP) {
        configBuilder.topP(topP);
        return this;
    }

    /**
     * Enables or disables incremental token streaming.
     *
     * @param stream {@code true} to stream tokens
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder stream(boolean stream) {
        configBuilder.stream(stream);
        return this;
    }

    /**
     * Sets the desired output modalities.
     *
     * @param modalities one or more {@link ModalityType} values
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder outputModalities(ModalityType... modalities) {
        configBuilder.outputModalities(modalities);
        return this;
    }

    /**
     * Sets provider-specific inference parameters.
     *
     * @param parameters key-value parameter map
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder parameters(Map<String, Object> parameters) {
        builder.parameters(parameters);
        return this;
    }

    /**
     * Sets the tenant identifier for multi-tenant deployments.
     *
     * @param tenantId tenant ID
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder tenantId(String tenantId) {
        builder.tenantId(tenantId);
        return this;
    }

    /**
     * Sets the request timeout.
     *
     * @param timeoutMs timeout in milliseconds; {@code 0} for no timeout
     * @return {@code this} for chaining
     */
    public MultimodalRequestBuilder timeoutMs(long timeoutMs) {
        builder.timeoutMs(timeoutMs);
        return this;
    }

    /**
     * Builds the {@link MultimodalRequest} with all configured inputs and output settings.
     *
     * @return a fully configured {@link MultimodalRequest}
     */
    public MultimodalRequest build() {
        builder.outputConfig(configBuilder.build());
        return builder.build();
    }

    /**
     * Functional interface used by {@link #outputConfig(Consumer)} for inline configuration.
     *
     * @param <T> the type accepted by this consumer
     */
    @FunctionalInterface
    public interface Consumer<T> {
        /**
         * Performs the configuration operation on the given argument.
         *
         * @param t the argument to configure
         */
        void accept(T t);
    }

    /**
     * Fluent builder for {@link MultimodalRequest.OutputConfig}, used within
     * {@link #outputConfig(Consumer)}.
     */
    public static class OutputConfigBuilder {
        private final MultimodalRequest.OutputConfig.Builder builder =
            MultimodalRequest.OutputConfig.builder();

        /** @param maxTokens maximum tokens to generate */
        public OutputConfigBuilder maxTokens(int maxTokens) { builder.maxTokens(maxTokens); return this; }

        /** @param temperature sampling temperature */
        public OutputConfigBuilder temperature(double temperature) { builder.temperature(temperature); return this; }

        /** @param topP nucleus sampling threshold */
        public OutputConfigBuilder topP(double topP) { builder.topP(topP); return this; }

        /** @param stream {@code true} to enable streaming */
        public OutputConfigBuilder stream(boolean stream) { builder.stream(stream); return this; }

        /** @param modalities desired output modalities */
        public OutputConfigBuilder outputModalities(ModalityType... modalities) { builder.outputModalities(modalities); return this; }

        /**
         * Builds the {@link MultimodalRequest.OutputConfig}.
         *
         * @return a configured {@link MultimodalRequest.OutputConfig}
         */
        public MultimodalRequest.OutputConfig build() { return builder.build(); }
    }
}
