package tech.kayys.golek.provider.ollama;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.provider.core.adapter.CloudProviderAdapter;
import tech.kayys.golek.provider.core.spi.StreamingProvider;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Ollama provider adapter for local LLM inference.
 * Connects to Ollama server running locally or on network.
 */
@ApplicationScoped
public class OllamaProvider extends CloudProviderAdapter implements StreamingProvider {

        private static final String PROVIDER_ID = "ollama";
        private static final String PROVIDER_NAME = "Ollama";
        private static final String VERSION = "1.0.0";

        @Inject
        @RestClient
        OllamaClient client;

        @Inject
        OllamaConfig config; // Keep for now if needed, but CloudProviderAdapter has config too

        private final AtomicInteger requestCounter = new AtomicInteger(0);

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
        protected String getDefaultBaseUrl() {
                return config.baseUrl();
        }

        @Override
        protected String getApiKeyEnvironmentVariable() {
                return "OLLAMA_API_KEY";
        }

        @Override
        protected Uni<Boolean> performHealthCheckRequest() {
                return client.listModels()
                                .map(models -> true)
                                .onFailure().recoverWithItem(false);
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
                                .displayName(PROVIDER_NAME)
                                .description("Local inference via Ollama - supports Llama, Mistral, Phi, Gemma, and more")
                                .version(VERSION)
                                .vendor("Ollama")
                                .documentationUrl("https://ollama.ai/docs")
                                .metadata("deployment", "local")
                                .metadata("requires_api_key", "false")
                                .metadata("base_url", getBaseUrl())
                                .build();
        }

        @Override
        public boolean supports(String model, TenantContext context) {
                return capabilities().getSupportedModels().contains(model) ||
                                model.startsWith("llama") ||
                                model.startsWith("mistral") ||
                                model.startsWith("phi") ||
                                model.startsWith("gemma") ||
                                model.startsWith("qwen") ||
                                model.startsWith("codellama");
        }

        @Override
        protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
                trackRequest();
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
                                                        .durationMs(duration)
                                                        .metadata("provider", PROVIDER_ID)
                                                        .metadata("prompt_tokens", response.getPromptEvalCount())
                                                        .metadata("completion_tokens", response.getEvalCount())
                                                        .metadata("total_tokens",
                                                                        response.getPromptEvalCount()
                                                                                        + response.getEvalCount())
                                                        .build();
                                })
                                .onFailure().invoke(this::trackError)
                                .onFailure().transform(this::wrapException);
        }

        @Override
        public Multi<StreamChunk> stream(ProviderRequest request, TenantContext context) {
                // trackRequest(); // Streaming calls might track differently, but good to track
                AtomicInteger chunkIndex = new AtomicInteger(0);

                OllamaRequest ollamaRequest = buildOllamaRequest(request);
                ollamaRequest.setStream(true);

                return client.chatStream(ollamaRequest)
                                .map(chunk -> {
                                        int index = chunkIndex.getAndIncrement();
                                        String content = chunk.getMessage() != null
                                                        ? chunk.getMessage().getContent()
                                                        : "";

                                        return StreamChunk.builder()
                                                        .index(index)
                                                        .delta(content)
                                                        .model(chunk.getModel())
                                                        .isFinal(chunk.isDone())
                                                        .metadata("eval_count", chunk.getEvalCount())
                                                        .build();
                                })
                                .onFailure().invoke(this::trackError)
                                .onFailure().transform(this::wrapException);
        }

        private OllamaRequest buildOllamaRequest(ProviderRequest request) {
                OllamaRequest ollamaRequest = new OllamaRequest();
                ollamaRequest.setModel(request.getModel());
                ollamaRequest.setStream(request.isStreaming());

                // Convert messages
                if (request.getMessages() != null) {
                        List<OllamaMessage> messages = request.getMessages().stream()
                                        .map(msg -> new OllamaMessage(msg.getRole(), msg.getContent()))
                                        .collect(Collectors.toList());
                        ollamaRequest.setMessages(messages);
                }

                // Set options
                OllamaOptions options = new OllamaOptions();
                if (request.getParameters().containsKey("temperature")) {
                        options.setTemperature((Double) request.getParameters().get("temperature"));
                }
                if (request.getParameters().containsKey("max_tokens")) {
                        options.setNumPredict((Integer) request.getParameters().get("max_tokens"));
                }
                if (request.getParameters().containsKey("top_p")) {
                        options.setTopP((Double) request.getParameters().get("top_p"));
                }
                ollamaRequest.setOptions(options);

                return ollamaRequest;
        }

        private Throwable wrapException(Throwable ex) {
                return new RuntimeException("Ollama request failed: " + ex.getMessage(), ex);
        }
}