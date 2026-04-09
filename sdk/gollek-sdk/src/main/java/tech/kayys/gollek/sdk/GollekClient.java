package tech.kayys.gollek.sdk;

import tech.kayys.gollek.sdk.model.GenerationRequest;
import tech.kayys.gollek.sdk.model.GenerationResponse;
import tech.kayys.gollek.sdk.model.GenerationStream;
import tech.kayys.gollek.sdk.model.EmbeddingRequest;
import tech.kayys.gollek.sdk.model.EmbeddingResponse;
import tech.kayys.gollek.sdk.model.ModelInfo;

/**
 * Main entry point for the Gollek SDK.
 * <p>
 * Provides high-level methods for text generation, streaming, and embeddings.
 * Use the {@link Builder} to create instances.
 * </p>
 */
public interface GollekClient extends AutoCloseable {

    /**
     * Synchronously generate text based on the request.
     *
     * @param request the generation parameters
     * @return the generated text and metadata
     */
    GenerationResponse generate(GenerationRequest request);

    /**
     * Asynchronously generate text based on the request.
     *
     * @param request the generation parameters
     * @return a future resolving to the generated text and metadata
     */
    java.util.concurrent.CompletableFuture<GenerationResponse> generateAsync(GenerationRequest request);

    /**
     * Start a streaming generation session.
     *
     * @param request the generation parameters (should have stream=true)
     * @return a stream handle to register listeners for tokens
     */
    GenerationStream generateStream(GenerationRequest request);

    /**
     * Generate embeddings for the given text.
     *
     * @param request the embedding parameters
     * @return the vector representation
     */
    EmbeddingResponse embed(EmbeddingRequest request);

    /**
     * List all locally available or remote models.
     *
     * @return list of model metadata; may be empty if the backend does not support listing
     */
    java.util.List<ModelInfo> listModels();

    /**
     * Get information about a specific model.
     *
     * @param modelId the model identifier
     * @return model metadata and capabilities
     */
    ModelInfo getModelInfo(String modelId);

    /**
     * Releases any resources held by this client.
     *
     * @throws Exception if cleanup fails
     */
    @Override
    default void close() throws Exception {
        // Default no-op
    }

    /**
     * Create a new client builder.
     */
    static Builder builder() {
        try {
            // First try to load the Discovery builder which handles Embedded vs Remote logic
            return (Builder) Class.forName("tech.kayys.gollek.sdk.internal.GollekClientBuilderImpl")
                    .getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load GollekClient builder. " +
                    "Ensure gollek-sdk is correctly configured on the classpath.", e);
        }
    }

    /**
     * Interface for building {@link GollekClient} instances.
     */
    interface Builder {
        /**
         * Configures the client to connect to a remote Gollek sidecar or server.
         *
         * @param endpoint base URL of the remote server (e.g. {@code "https://gollek.example.com"})
         * @return {@code this} for chaining
         */
        Builder endpoint(String endpoint);

        /**
         * Configures the client to use the local in-process inference engine.
         * Requires {@code gollek-engine} on the classpath and a running Quarkus environment.
         *
         * @return {@code this} for chaining
         */
        Builder local();

        /**
         * Sets the default model identifier used when a request does not specify one.
         *
         * @param model the model identifier
         * @return {@code this} for chaining
         */
        Builder model(String model);

        /**
         * Sets the API key used for bearer-token authentication.
         *
         * @param apiKey the API key
         * @return {@code this} for chaining
         */
        Builder apiKey(String apiKey);

        /**
         * Sets the request timeout in milliseconds.
         *
         * @param timeout timeout in milliseconds
         * @return {@code this} for chaining
         */
        Builder timeoutMillis(long timeout);

        /**
         * Builds and returns the configured {@link GollekClient}.
         *
         * @return a ready-to-use client instance
         * @throws IllegalStateException if neither an endpoint nor a local engine is available
         */
        GollekClient build();
    }
}
