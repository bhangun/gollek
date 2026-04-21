package tech.kayys.gollek.sdk;


import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import tech.kayys.gollek.spi.model.ModelInfo;

/**
 * Unified Gollek inference client — single entry point regardless of backend
 * (GGUF, ONNX, LiteRT, LibTorch, cloud API).
 *
 * <p>Hides backend complexity behind a clean, streaming-first API.
 * The backend is auto-detected from the model path or endpoint URL,
 * or explicitly configured via {@link #builder()}.
 *
 * <h3>Example — local model</h3>
 * <pre>{@code
 * try (GollekClient client = GollekClient.builder()
 *         .model("~/.gollek/models/qwen2-7b.gguf")
 *         .build()) {
 *
 *     // Streaming generation
 *     client.generateStream("Explain attention mechanisms")
 *           .onToken(System.out::print)
 *           .onComplete(r -> System.out.println("\n[done] " + r.tokenCount() + " tokens"));
 *
 *     // Batch embeddings
 *     float[] emb = client.embed("Hello world");
 * }
 * }</pre>
 *
 * <h3>Example — remote endpoint</h3>
 * <pre>{@code
 * GollekClient client = GollekClient.builder()
 *     .endpoint("http://localhost:8080")
 *     .model("qwen2-7b")
 *     .build();
 * }</pre>
 */
public interface GollekClient extends AutoCloseable {

    // ── Generation ────────────────────────────────────────────────────────

    /**
     * Generates text synchronously.
     *
     * @param prompt input prompt
     * @return complete generation result
     */
    GenerationResult generate(String prompt);

    /**
     * Generates text synchronously with full request configuration.
     *
     * @param request generation request with parameters
     * @return complete generation result
     */
    GenerationResult generate(GenerationRequest request);

    /**
     * Generates text as a reactive stream — tokens are delivered as they are produced.
     *
     * @param prompt input prompt
     * @return {@link GenerationStream} for registering token/completion handlers
     */
    GenerationStream generateStream(String prompt);

    /**
     * Generates text for multiple prompts in a single batched call.
     *
     * @param prompts list of input prompts
     * @return list of results in the same order as inputs
     */
    List<GenerationResult> generateBatch(List<String> prompts);

    // ── Embeddings ────────────────────────────────────────────────────────

    /**
     * Computes a dense embedding vector for the given text.
     *
     * @param text input text
     * @return embedding vector (length = model's hidden size)
     */
    float[] embed(String text);

    /**
     * Computes embeddings for multiple texts in a single batched call.
     *
     * @param texts list of input texts
     * @return list of embedding vectors
     */
    List<float[]> embedBatch(List<String> texts);

    // ── Model info ────────────────────────────────────────────────────────

    /**
     * Returns metadata about the loaded model.
     *
     * @return {@link ModelInfo} with name, architecture, parameter count, etc.
     */
    ModelInfo modelInfo();

    /**
     * Checks whether the backend supports a specific feature.
     *
     * @param feature the feature to check
     * @return {@code true} if supported
     */
    boolean supports(Feature feature);

    // ── Lifecycle ─────────────────────────────────────────────────────────

    @Override void close();

    // ── Factory ───────────────────────────────────────────────────────────

    /**
     * Creates a new builder for {@link GollekClient}.
     *
     * @return builder instance
     */
    static Builder builder() { return new GollekClientImpl.BuilderImpl(); }

    // ── Nested types ──────────────────────────────────────────────────────

    /**
     * Supported backend features for capability negotiation.
     */
    enum Feature {
        /** KV-cache for faster autoregressive generation. */
        KV_CACHE,
        /** Speculative decoding for faster generation. */
        SPECULATIVE_DECODING,
        /** Vision/audio inputs alongside text. */
        MULTIMODAL,
        /** Flash Attention kernel. */
        FLASH_ATTENTION,
        /** INT8 quantized inference. */
        QUANTIZATION_INT8,
        /** FP16 inference. */
        QUANTIZATION_FP16,
        /** Token-by-token streaming. */
        STREAMING,
        /** Batched inference. */
        BATCH_INFERENCE
    }

    /**
     * Generation request with full parameter control.
     *
     * @param prompt      input text
     * @param maxTokens   maximum tokens to generate (default 512)
     * @param temperature sampling temperature (default 0.7)
     * @param topP        nucleus sampling probability (default 0.9)
     * @param stopTokens  list of stop sequences
     */
    record GenerationRequest(
        String prompt,
        int maxTokens,
        float temperature,
        float topP,
        List<String> stopTokens
    ) {
        /** Creates a simple request with default parameters. */
        public static GenerationRequest of(String prompt) {
            return new GenerationRequest(prompt, 512, 0.7f, 0.9f, List.of());
        }
    }

    /**
     * Result of a completed generation.
     *
     * @param text       generated text
     * @param tokenCount number of tokens generated
     * @param promptTokens number of prompt tokens
     * @param durationMs wall-clock time in milliseconds
     */
    record GenerationResult(String text, int tokenCount, int promptTokens, long durationMs) {
        /** Tokens per second throughput. */
        public float tokensPerSecond() {
            return durationMs > 0 ? tokenCount * 1000f / durationMs : 0f;
        }
    }

    /**
     * Model metadata.
     *
     * @deprecated Use {@link tech.kayys.gollek.spi.model.ModelInfo} instead.
     */
    @Deprecated
    record LegacyModelInfo(String name, String architecture, long parameterCount, int contextLength) {}

    /**
     * Reactive stream for token-by-token generation.
     */
    interface GenerationStream {

        /**
         * Registers a handler called for each generated token.
         *
         * @param handler consumer receiving each token string
         * @return this stream for chaining
         */
        GenerationStream onToken(Consumer<String> handler);

        /**
         * Registers a handler called when generation completes.
         *
         * @param handler consumer receiving the final {@link GenerationResult}
         * @return this stream for chaining
         */
        GenerationStream onComplete(Consumer<GenerationResult> handler);

        /**
         * Registers a handler called if generation fails.
         *
         * @param handler consumer receiving the error
         * @return this stream for chaining
         */
        GenerationStream onError(Consumer<Throwable> handler);

        /**
         * Returns a {@link CompletableFuture} that completes with the full result.
         *
         * @return future resolving to {@link GenerationResult}
         */
        CompletableFuture<GenerationResult> toFuture();
    }

    /**
     * Builder for {@link GollekClient}.
     */
    interface Builder {

        /**
         * Sets the model path (local file) or model name (for remote/hub).
         *
         * @param model path to {@code .gguf}, {@code .onnx}, {@code .safetensors},
         *              or a model name like {@code "qwen2-7b"}
         */
        Builder model(String model);

        /**
         * Sets the remote endpoint URL for server-mode inference.
         *
         * @param endpoint base URL, e.g. {@code "http://localhost:8080"}
         */
        Builder endpoint(String endpoint);

        /**
         * Sets the maximum number of tokens to generate by default.
         *
         * @param maxTokens default max tokens (default 512)
         */
        Builder maxTokens(int maxTokens);

        /**
         * Sets the default sampling temperature.
         *
         * @param temperature value in (0, 2] (default 0.7)
         */
        Builder temperature(float temperature);

        /**
         * Explicitly sets the backend type.
         * If not set, the backend is auto-detected from the model file extension.
         *
         * @param backend backend identifier (e.g. {@code "gguf"}, {@code "onnx"}, {@code "litert"})
         */
        Builder backend(String backend);

        /**
         * Builds and initializes the {@link GollekClient}.
         *
         * @return ready-to-use client
         */
        GollekClient build();
    }
}
