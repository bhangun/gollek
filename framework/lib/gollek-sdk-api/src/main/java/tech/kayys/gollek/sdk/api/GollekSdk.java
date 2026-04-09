package tech.kayys.gollek.sdk.api;

import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalCapability;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;

import java.util.List;
import java.util.Set;
import java.util.concurrent.Flow;

/**
 * High-level SDK interface for model interaction.
 * <p>
 * Provides a stable contract for both local and remote inference,
 * covering text completions, multimodal processing, embeddings,
 * and capability discovery.
 *
 * <h3>Backward Compatibility</h3>
 * New methods are provided with {@code default} implementations that throw
 * {@link UnsupportedOperationException}, so existing {@code GollekSdkProvider}
 * implementations continue to compile without modification.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * GollekSdk sdk = GollekSdk.builder()
 *     .provider("gemini")
 *     .apiKey(System.getenv("GEMINI_API_KEY"))
 *     .model("gemini-2.0-flash")
 *     .build();
 *
 * // Text completion
 * InferenceResponse text = sdk.createCompletion(request);
 *
 * // Multimodal: image + text
 * MultimodalResponse mm = sdk.processMultimodal(
 *     MultimodalRequest.builder()
 *         .model("gemini-2.0-flash")
 *         .inputs(
 *             MultimodalContent.ofText("Describe this:"),
 *             MultimodalContent.ofBase64Image(bytes, "image/jpeg"))
 *         .build());
 *
 * // Embeddings
 * EmbeddingResponse emb = sdk.createEmbedding(
 *     EmbeddingRequest.builder()
 *         .model("text-embedding-3-small")
 *         .input("Hello world")
 *         .build());
 * }</pre>
 */
public interface GollekSdk {

    // ═══════════════════════════════════════════════════════════════════════
    // Text Completion (original contract)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a text completion based on the provided request.
     *
     * @param request the inference request
     * @return the inference response
     */
    InferenceResponse createCompletion(InferenceRequest request);

    // ═══════════════════════════════════════════════════════════════════════
    // Multimodal Processing
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Processes a multimodal request containing mixed modality inputs
     * (text, images, audio, video, documents).
     *
     * @param request the multimodal request with one or more content parts
     * @return the multimodal response
     * @throws UnsupportedOperationException if the provider does not support multimodal
     */
    default MultimodalResponse processMultimodal(MultimodalRequest request) {
        throw new UnsupportedOperationException(
                "This provider does not support multimodal processing. " +
                "Use a multimodal-capable provider (e.g., gemini, openai, anthropic).");
    }

    /**
     * Processes a multimodal request with streaming response.
     * <p>
     * Returns a {@link Flow.Publisher} that emits partial response chunks
     * as they become available from the model.
     *
     * @param request the multimodal request
     * @return a reactive publisher of streaming response chunks
     * @throws UnsupportedOperationException if the provider does not support multimodal streaming
     */
    default Flow.Publisher<MultimodalResponse> processMultimodalStream(MultimodalRequest request) {
        throw new UnsupportedOperationException(
                "This provider does not support multimodal streaming.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Embeddings
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Generates dense vector embeddings for the given inputs.
     *
     * @param request the embedding request with model and input texts
     * @return the embedding response containing vector representations
     * @throws UnsupportedOperationException if the provider does not support embeddings
     */
    default EmbeddingResponse createEmbedding(EmbeddingRequest request) {
        throw new UnsupportedOperationException(
                "This provider does not support embeddings.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Capability Discovery
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Returns the multimodal capabilities of this SDK instance,
     * including supported input/output modalities and size limits.
     *
     * @return list of capability descriptors, one per available model
     */
    default List<MultimodalCapability> capabilities() {
        return List.of();
    }

    /**
     * Checks whether this SDK instance supports the given modality type
     * as an input.
     *
     * @param type the modality to check
     * @return {@code true} if at least one available model accepts this modality
     */
    default boolean supportsModality(ModalityType type) {
        return capabilities().stream()
                .anyMatch(cap -> cap.getInputModalities().contains(type));
    }

    /**
     * Returns the set of all modality types supported as input across
     * all available models.
     *
     * @return set of supported input modality types
     */
    default Set<ModalityType> supportedInputModalities() {
        return capabilities().stream()
                .flatMap(cap -> cap.getInputModalities().stream())
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    /**
     * Returns the provider identifier for this SDK instance
     * (e.g., "gollek-local", "openai", "gemini").
     *
     * @return the provider name, or "unknown" if not set
     */
    default String provider() {
        return "unknown";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a new builder for constructing {@link GollekSdk} instances.
     * <p>
     * The builder uses {@link java.util.ServiceLoader} to discover available
     * {@link GollekSdkProvider} implementations on the classpath and resolves
     * the best match based on the configured provider preference.
     *
     * @return a new SDK builder
     */
    static GollekSdkBuilder builder() {
        return new GollekSdkBuilder();
    }
}
