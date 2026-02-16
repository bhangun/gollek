package tech.kayys.golek.provider.ollama;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderMetadata;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.golek.spi.provider.StreamingProvider;
import tech.kayys.golek.spi.stream.StreamChunk;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Ollama provider implementation for local LLM inference.
 */
@ApplicationScoped
public class OllamaProvider implements StreamingProvider {

        private static final Logger log = Logger.getLogger(OllamaProvider.class);
        private static final String PROVIDER_ID = "ollama";
        private static final String PROVIDER_NAME = "Ollama";
        private static final String VERSION = "1.0.0";

        @Inject
        @RestClient
        OllamaClient client;

        @Inject
        OllamaConfig config;

        private final AtomicInteger requestCounter = new AtomicInteger(0);

        @Override
        public void initialize(ProviderConfig config) {
                log.info("Ollama provider initialized");
        }

        @Override
        public void shutdown() {
                log.info("Ollama provider shutting down");
        }

        @Override
        public Uni<ProviderHealth> health() {
                return client.listModels()
                                .map(models -> ProviderHealth.healthy("Ollama is running"))
                                .onFailure().recoverWithItem(
                                                t -> ProviderHealth.unhealthy("Ollama unavailable: " + t.getMessage()));
        }

        @Override
        public String id() {
                return PROVIDER_ID;
        }

        @Override
        public String name() {
                return PROVIDER_NAME;
        }

        @Override
        public String version() {
                return VERSION;
        }

        @Override
        public ProviderCapabilities capabilities() {
                return ProviderCapabilities.builder()
                                .streaming(true)
                                .functionCalling(false)
                                .multimodal(true)
                                .embeddings(true)
                                .maxContextTokens(128000)
                                .maxOutputTokens(4096)
                                .supportedModels(Set.of(
                                                "llama3.3:70b",
                                                "llama3.2:3b",
                                                "llama3.1:70b",
                                                "llama3.1:8b",
                                                "llama3:8b",
                                                "mistral:7b",
                                                "mixtral:8x7b",
                                                "phi3:medium",
                                                "phi3:mini",
                                                "gemma2:27b",
                                                "gemma2:9b",
                                                "codellama:13b",
                                                "codellama:7b",
                                                "qwen2.5:72b",
                                                "qwen2.5:7b"))
                                .supportedLanguages(List.of("en", "zh", "es", "fr", "de", "ja", "ko"))
                                .build();
        }

        @Override
        public ProviderMetadata metadata() {
                return ProviderMetadata.builder()
                                .providerId(PROVIDER_ID)
                                .name(PROVIDER_NAME)
                                .description("Local inference via Ollama - supports Llama, Mistral, Phi, Gemma, and more")
                                .version(VERSION)
                                .vendor("Ollama")
                                .homepage("https://ollama.ai/docs")
                                .build();
        }

        @Override
        public boolean supports(String model, ProviderRequest request) {
                return capabilities().getSupportedModels().contains(model) ||
                                model.startsWith("llama") ||
                                model.startsWith("mistral") ||
                                model.startsWith("phi") ||
                                model.startsWith("gemma") ||
                                model.startsWith("qwen");
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                log.debugf("[%d] Ollama inference: model=%s", requestId, request.getModel());

                OllamaRequest ollamaRequest = buildOllamaRequest(request);

                return client.chat(ollamaRequest)
                                .map(response -> {
                                        long duration = System.currentTimeMillis() - startTime;
                                        log.debugf("[%d] Ollama response in %dms", requestId, duration);

                                        return InferenceResponse.builder()
                                                        .requestId(request.getRequestId())
                                                        .content(response.getMessage().getContent())
                                                        .model(response.getModel())
                                                        .inputTokens(response.getPromptEvalCount())
                                                        .outputTokens(response.getEvalCount())
                                                        .tokensUsed(response.getPromptEvalCount()
                                                                        + response.getEvalCount())
                                                        .durationMs(duration)
                                                        .metadata("provider", PROVIDER_ID)
                                                        .build();
                                });
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request) {
                AtomicInteger chunkIndex = new AtomicInteger(0);

                OllamaRequest ollamaRequest = buildOllamaRequest(request);
                ollamaRequest.setStream(true);

                return client.chatStream(ollamaRequest)
                                .map(chunk -> {
                                        int index = chunkIndex.getAndIncrement();
                                        String content = chunk.getMessage() != null
                                                        ? chunk.getMessage().getContent()
                                                        : "";

                                        return chunk.isDone()
                                                        ? StreamChunk.finalChunk(request.getRequestId(), index, content)
                                                        : StreamChunk.of(request.getRequestId(), index, content);
                                });
        }

        private OllamaRequest buildOllamaRequest(ProviderRequest request) {
                OllamaRequest ollamaRequest = new OllamaRequest();
                ollamaRequest.setModel(request.getModel());
                ollamaRequest.setStream(request.isStreaming());

                // Convert messages
                if (request.getMessages() != null) {
                        List<OllamaMessage> messages = request.getMessages().stream()
                                        .map(msg -> new OllamaMessage(msg.getRole().name().toLowerCase(),
                                                        msg.getContent()))
                                        .collect(Collectors.toList());
                        ollamaRequest.setMessages(messages);
                }

                // Set options
                OllamaOptions options = new OllamaOptions();
                if (request.getParameters().containsKey("temperature")) {
                        options.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
                }
                if (request.getParameters().containsKey("max_tokens")) {
                        options.setNumPredict(((Number) request.getParameters().get("max_tokens")).intValue());
                }
                if (request.getParameters().containsKey("top_p")) {
                        options.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
                }
                ollamaRequest.setOptions(options);

                return ollamaRequest;
        }
}