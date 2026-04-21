package tech.kayys.gollek.sdk;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Consumer;
import tech.kayys.gollek.spi.model.ModelInfo;

/**
 * Default implementation of {@link GollekClient}.
 *
 * <p>Routes requests to the appropriate backend based on model file extension
 * or explicit backend configuration. Uses JDK 25 virtual threads for
 * non-blocking streaming generation.
 *
 * <p>Backend auto-detection:
 * <ul>
 *   <li>{@code .gguf} → GGUF/llama.cpp runner</li>
 *   <li>{@code .onnx} → ONNX Runtime runner</li>
 *   <li>{@code .safetensors} → LibTorch/SafeTensors runner</li>
 *   <li>{@code .tflite} → LiteRT runner</li>
 *   <li>URL endpoint → HTTP REST client</li>
 * </ul>
 */
final class GollekClientImpl implements GollekClient {

    private final String model;
    private final String endpoint;
    private final int    maxTokens;
    private final float  temperature;
    private final String backend;

    private GollekClientImpl(BuilderImpl b) {
        this.model       = b.model;
        this.endpoint    = b.endpoint;
        this.maxTokens   = b.maxTokens;
        this.temperature = b.temperature;
        this.backend     = b.backend != null ? b.backend : detectBackend(b.model);
    }

    // ── GollekClient interface ────────────────────────────────────────────

    @Override
    public GenerationResult generate(String prompt) {
        return generate(GenerationRequest.of(prompt));
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        long start = System.currentTimeMillis();
        // Delegate to backend runner (stub — real impl calls JNI/HTTP)
        String text = "[" + backend + "] " + request.prompt().substring(0, Math.min(20, request.prompt().length())) + "...";
        int tokens = text.split("\\s+").length;
        return new GenerationResult(text, tokens, request.prompt().split("\\s+").length,
            System.currentTimeMillis() - start);
    }

    @Override
    public GenerationStream generateStream(String prompt) {
        return new StreamImpl(this, GenerationRequest.of(prompt));
    }

    @Override
    public List<GenerationResult> generateBatch(List<String> prompts) {
        // Run in parallel using virtual threads
        List<Future<GenerationResult>> futures = new ArrayList<>();
        try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
            for (String p : prompts) futures.add(exec.submit(() -> generate(p)));
            List<GenerationResult> results = new ArrayList<>(prompts.size());
            for (var f : futures) {
                try { results.add(f.get()); }
                catch (Exception e) { results.add(new GenerationResult("", 0, 0, 0)); }
            }
            return results;
        }
    }

    @Override
    public float[] embed(String text) {
        // Stub — real impl calls embedding model
        float[] emb = new float[768];
        for (int i = 0; i < emb.length; i++) emb[i] = (float) Math.sin(i + text.hashCode());
        return emb;
    }

    @Override
    public List<float[]> embedBatch(List<String> texts) {
        return texts.stream().map(this::embed).toList();
    }

    @Override
    public ModelInfo modelInfo() {
        return ModelInfo.builder()
                .modelId(model)
                .format(backend)
                .contextLength(4096L)
                .build();
    }

    @Override
    public boolean supports(Feature feature) {
        return switch (feature) {
            case STREAMING, BATCH_INFERENCE -> true;
            case KV_CACHE -> "gguf".equals(backend);
            default -> false;
        };
    }

    @Override public void close() { /* release native resources */ }

    // ── Backend detection ─────────────────────────────────────────────────

    private static String detectBackend(String model) {
        if (model == null) return "remote";
        if (model.endsWith(".gguf"))        return "gguf";
        if (model.endsWith(".onnx"))        return "onnx";
        if (model.endsWith(".safetensors")) return "safetensors";
        if (model.endsWith(".tflite"))      return "litert";
        return "remote";
    }

    // ── Stream implementation ─────────────────────────────────────────────

    private static final class StreamImpl implements GenerationStream {
        private final GollekClientImpl client;
        private final GenerationRequest request;
        private Consumer<String>           tokenHandler   = t -> {};
        private Consumer<GenerationResult> completeHandler = r -> {};
        private Consumer<Throwable>        errorHandler   = e -> {};

        StreamImpl(GollekClientImpl client, GenerationRequest request) {
            this.client  = client;
            this.request = request;
            // Start generation in a virtual thread immediately
            Thread.ofVirtual().start(this::run);
        }

        private void run() {
            try {
                GenerationResult result = client.generate(request);
                // Simulate token-by-token delivery
                for (String token : result.text().split("(?<=\\s)")) {
                    tokenHandler.accept(token);
                    Thread.sleep(1); // simulate latency
                }
                completeHandler.accept(result);
            } catch (Exception e) {
                errorHandler.accept(e);
            }
        }

        @Override public GenerationStream onToken(Consumer<String> h)           { tokenHandler = h; return this; }
        @Override public GenerationStream onComplete(Consumer<GenerationResult> h){ completeHandler = h; return this; }
        @Override public GenerationStream onError(Consumer<Throwable> h)         { errorHandler = h; return this; }

        @Override
        public CompletableFuture<GenerationResult> toFuture() {
            CompletableFuture<GenerationResult> future = new CompletableFuture<>();
            onComplete(future::complete).onError(future::completeExceptionally);
            return future;
        }
    }

    // ── Builder ───────────────────────────────────────────────────────────

    static final class BuilderImpl implements GollekClient.Builder {
        String model, endpoint, backend;
        int   maxTokens   = 512;
        float temperature = 0.7f;

        @Override public Builder model(String m)       { this.model = m; return this; }
        @Override public Builder endpoint(String e)    { this.endpoint = e; return this; }
        @Override public Builder maxTokens(int n)      { this.maxTokens = n; return this; }
        @Override public Builder temperature(float t)  { this.temperature = t; return this; }
        @Override public Builder backend(String b)     { this.backend = b; return this; }
        @Override public GollekClient build()          { return new GollekClientImpl(this); }
    }
}
