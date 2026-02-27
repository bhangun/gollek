package tech.kayys.gollek.provider.gemini;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Google Gemini provider adapter for cloud LLM inference.
 * Standardized on OpenAI-compatible API for maximum stability.
 */
@ApplicationScoped
public class GeminiProvider implements StreamingProvider {

        private static final String PROVIDER_ID = "gemini";
        private static final String PROVIDER_NAME = "Google Gemini";
        private static final String VERSION = "1.0.0";
        private static final String GEMINI_OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
        private static final long MAX_AUTO_RETRY_MS = 60_000L;
        private static final Pattern RETRY_SECONDS_PATTERN = Pattern
                        .compile("retry\\s+in\\s+([0-9]+(?:\\.[0-9]+)?)s", Pattern.CASE_INSENSITIVE);
        private static final Logger log = Logger.getLogger(GeminiProvider.class);

        @Inject
        com.fasterxml.jackson.databind.ObjectMapper objectMapper;

        @Inject
        GeminiConfig configDetails;

        private HttpClient httpClient;

        @jakarta.annotation.PostConstruct
        void init() {
                this.httpClient = HttpClient.newBuilder()
                                .connectTimeout(Duration.ofSeconds(10))
                                .proxy(ProxySelector.getDefault())
                                .build();
        }

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
        public void initialize(ProviderConfig config) {
                log.info("Gemini provider initialized");
        }

        @Override
        public void shutdown() {
                log.info("Gemini provider shutting down");
        }

        @Override
        public Uni<ProviderHealth> health() {
                return Uni.createFrom().item(() -> {
                        String currentApiKey = getApiKey(null);
                        return ProviderHealth.healthy(
                                        currentApiKey != null && !currentApiKey.isBlank() ? "Gemini API available"
                                                        : "Gemini initialized (API key missing)");
                });
        }

        @Override
        public ProviderCapabilities capabilities() {
                return ProviderCapabilities.builder()
                                .streaming(true)
                                .functionCalling(true)
                                .multimodal(true)
                                .maxContextTokens(1000000)
                                .build();
        }

        @Override
        public ProviderMetadata metadata() {
                return ProviderMetadata.builder()
                                .providerId(PROVIDER_ID)
                                .name(PROVIDER_NAME)
                                .description("Google Gemini - multimodal AI with large context window")
                                .version(VERSION)
                                .vendor("Google")
                                .homepage("https://ai.google.dev/docs")
                                .defaultModel(configDetails != null ? configDetails.defaultModel() : "gemini-2.5-flash")
                                .build();
        }

        @Override
        public boolean supports(String model, ProviderRequest request) {
                return model != null && model.startsWith("gemini");
        }

        private String getApiKey(ProviderRequest request) {
                if (request != null && request.getApiKey().isPresent()) {
                        return request.getApiKey().get();
                }
                String key = (configDetails != null) ? configDetails.apiKey() : null;
                if (key != null && !key.isBlank() && !"dummy".equals(key)) {
                        return key;
                }
                // Fallback to standard environment variables
                String envKey = System.getenv("GEMINI_API_KEY");
                if (envKey == null || envKey.isBlank()) {
                        envKey = System.getenv("GOOGLE_API_KEY");
                }
                return envKey;
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                GeminiRequest geminiRequest = buildOpenAICompatibleRequest(request);
                String currentApiKey = getApiKey(request);

                if (currentApiKey == null || currentApiKey.isBlank()) {
                        return Uni.createFrom()
                                        .failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                                                        "Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."));
                }

                // Using OpenAI compatible endpoint for Google AI Studio
                String url = GEMINI_OPENAI_COMPAT_URL;

                try {
                        String body = objectMapper.writeValueAsString(geminiRequest);
                        HttpRequest httpRequest = HttpRequest.newBuilder()
                                        .uri(URI.create(url))
                                        .header("Content-Type", "application/json")
                                        .header("Authorization", "Bearer " + currentApiKey)
                                        .POST(HttpRequest.BodyPublishers.ofString(body))
                                        .build();

                        return Uni.createFrom()
                                        .completionStage(sendWithQuotaRetry(httpRequest))
                                        .map(resp -> {
                                                if (resp.statusCode() != 200) {
                                                        throw buildHttpException(resp.statusCode(), resp.body());
                                                }
                                                try {
                                                        com.fasterxml.jackson.databind.JsonNode root = objectMapper
                                                                        .readTree(resp.body());
                                                        GeminiResponse response = objectMapper.treeToValue(root,
                                                                        GeminiResponse.class);
                                                        String content = extractContent(root, false);
                                                        GeminiUsage usage = response != null ? response.getUsage() : null;
                                                        long duration = System.currentTimeMillis() - startTime;
                                                        return InferenceResponse.builder()
                                                                        .requestId(request.getRequestId() != null
                                                                                        ? request.getRequestId()
                                                                                        : String.valueOf(requestId))
                                                                        .content(content)
                                                                        .model(request.getModel())
                                                                        .inputTokens(usage != null ? usage.getPromptTokens()
                                                                                        : 0)
                                                                        .outputTokens(usage != null
                                                                                        ? usage.getCompletionTokens()
                                                                                        : 0)
                                                                        .tokensUsed(usage != null ? usage.getTotalTokens()
                                                                                        : 0)
                                                                        .durationMs(duration)
                                                                        .metadata("provider", PROVIDER_ID)
                                                                        .build();
                                                } catch (Exception e) {
                                                        throw new RuntimeException("Gemini deserialization failed", e);
                                                }
                                        });
                } catch (Exception e) {
                        return Uni.createFrom().failure(e);
                }
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request) {
                AtomicInteger chunkIndex = new AtomicInteger(0);
                GeminiRequest geminiRequest = buildOpenAICompatibleRequest(request);
                geminiRequest.setStream(true);
                String currentApiKey = getApiKey(request);

                if (currentApiKey == null || currentApiKey.isBlank()) {
                        return Multi.createFrom()
                                        .failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                                                        "Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."));
                }

                // Using OpenAI compatible endpoint for Google AI Studio
                String url = GEMINI_OPENAI_COMPAT_URL;

                log.debug("Gemini streaming URL (OpenAI-compatible): " + url);

                try {
                        String body = objectMapper.writeValueAsString(geminiRequest);
                        HttpRequest httpRequest = HttpRequest.newBuilder()
                                        .uri(URI.create(url))
                                        .header("Content-Type", "application/json")
                                        .header("Authorization", "Bearer " + currentApiKey)
                                        .header("Accept", "text/event-stream")
                                        .POST(HttpRequest.BodyPublishers.ofString(body))
                                        .build();

                        return Multi.createFrom().emitter(emitter -> {
                                httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofInputStream())
                                                .thenAccept(resp -> {
                                                        if (resp.statusCode() != 200) {
                                                                if (resp.statusCode() == 429) {
                                                                        emitter.fail(new RuntimeException(
                                                                                        "Gemini quota exceeded (429). Wait before retrying, or use another provider/model."));
                                                                } else {
                                                                        emitter.fail(new RuntimeException(
                                                                                        "Gemini streaming failed (OpenAI-compatible): "
                                                                                                        + resp.statusCode()));
                                                                }
                                                                return;
                                                        }
                                                        try (BufferedReader reader = new BufferedReader(
                                                                        new InputStreamReader(resp.body()))) {
                                                                String line;
                                                                while ((line = reader.readLine()) != null) {
                                                                        if (line.startsWith("data: ")) {
                                                                                String data = line.substring(6).trim();
                                                                                if (!data.isEmpty() && !"[DONE]"
                                                                                                .equals(data)) {
                                                                                        try {
                                                                                                com.fasterxml.jackson.databind.JsonNode root = objectMapper
                                                                                                                .readTree(data);
                                                                                                String content = extractContent(
                                                                                                                root, true);
                                                                                                if (content == null
                                                                                                                || content.isBlank()) {
                                                                                                        continue;
                                                                                                }
                                                                                                int index = chunkIndex
                                                                                                                .getAndIncrement();

                                                                                                emitter.emit(StreamChunk
                                                                                                                .of(request.getRequestId(),
                                                                                                                                index,
                                                                                                                                content));
                                                                                        } catch (Exception e) {
                                                                                                log.warn("Failed to parse Gemini chunk: "
                                                                                                                + data,
                                                                                                                e);
                                                                                        }
                                                                                }
                                                                        }
                                                                }
                                                                emitter.complete();
                                                        } catch (Exception e) {
                                                                emitter.fail(e);
                                                        }
                                                })
                                                .exceptionally(t -> {
                                                        emitter.fail(t);
                                                        return null;
                                                });
                        });
                } catch (Exception e) {
                        return Multi.createFrom().failure(e);
                }
        }

        private GeminiRequest buildOpenAICompatibleRequest(ProviderRequest request) {
                GeminiRequest geminiRequest = new GeminiRequest();
                String model = request.getModel() != null ? request.getModel().trim()
                                : (configDetails != null ? configDetails.defaultModel() : "gemini-2.5-flash");
                model = normalizeModel(model);
                geminiRequest.setModel(model);

                if (request.getMessages() != null) {
                        List<GeminiMessage> messages = request.getMessages().stream()
                                        .map(msg -> new GeminiMessage(mapRole(msg.getRole().toString()),
                                                        msg.getContent()))
                                        .collect(Collectors.toList());
                        geminiRequest.setMessages(messages);
                }

                if (request.getParameters() != null) {
                        if (request.getParameters().containsKey("temperature")) {
                                geminiRequest.setTemperature(
                                                ((Number) request.getParameters().get("temperature")).doubleValue());
                        }
                        if (request.getParameters().containsKey("max_tokens")) {
                                geminiRequest.setMaxTokens(
                                                ((Number) request.getParameters().get("max_tokens")).intValue());
                        } else {
                                geminiRequest.setMaxTokens(request.getMaxTokens());
                        }
                        if (request.getParameters().containsKey("top_p")) {
                                geminiRequest.setTopP(
                                                ((Number) request.getParameters().get("top_p")).doubleValue());
                        }
                }
                return geminiRequest;
        }

        private String normalizeModel(String model) {
                if (model == null || model.isBlank()) {
                        return configDetails != null ? configDetails.defaultModel() : "gemini-2.5-flash";
                }
                String normalized = model.trim();
                // Gemini 1.5 models are deprecated/shut down; route to current default.
                if (normalized.startsWith("gemini-1.5-")) {
                        return configDetails != null ? configDetails.defaultModel() : "gemini-2.5-flash";
                }
                return normalized;
        }

        private String mapRole(String role) {
                String r = role.toLowerCase();
                return switch (r) {
                        case "user" -> "user";
                        case "assistant" -> "assistant";
                        case "system" -> "system";
                        default -> "user";
                };
        }

        private String extractContent(com.fasterxml.jackson.databind.JsonNode root, boolean streaming) {
                if (root == null) {
                        return "";
                }
                com.fasterxml.jackson.databind.JsonNode choices = root.get("choices");
                if (choices == null || !choices.isArray() || choices.isEmpty()) {
                        return "";
                }
                com.fasterxml.jackson.databind.JsonNode choice = choices.get(0);
                com.fasterxml.jackson.databind.JsonNode primary = streaming ? choice.path("delta") : choice.path("message");
                com.fasterxml.jackson.databind.JsonNode fallback = streaming ? choice.path("message") : choice.path("delta");

                String content = readContentNode(primary.path("content"));
                if (!content.isBlank()) {
                        return content;
                }
                return readContentNode(fallback.path("content"));
        }

        private String readContentNode(com.fasterxml.jackson.databind.JsonNode contentNode) {
                if (contentNode == null || contentNode.isMissingNode() || contentNode.isNull()) {
                        return "";
                }
                if (contentNode.isTextual()) {
                        return contentNode.asText("");
                }
                if (contentNode.isObject()) {
                        String text = contentNode.path("text").asText("");
                        if (!text.isBlank()) {
                                return text;
                        }
                        com.fasterxml.jackson.databind.JsonNode parts = contentNode.path("parts");
                        if (parts.isArray()) {
                                return readContentNode(parts);
                        }
                        return "";
                }
                if (contentNode.isArray()) {
                        StringJoiner joiner = new StringJoiner("");
                        for (com.fasterxml.jackson.databind.JsonNode part : contentNode) {
                                if (part == null || part.isNull()) {
                                        continue;
                                }
                                if (part.isTextual()) {
                                        joiner.add(part.asText(""));
                                        continue;
                                }
                                String type = part.path("type").asText("");
                                String text = part.path("text").asText("");
                                if (!text.isBlank()) {
                                        joiner.add(text);
                                        continue;
                                }
                                if ("text".equalsIgnoreCase(type)
                                                || "output_text".equalsIgnoreCase(type)
                                                || "input_text".equalsIgnoreCase(type)
                                                || type.isBlank()) {
                                        String nested = readContentNode(part.path("content"));
                                        if (!nested.isBlank()) {
                                                joiner.add(nested);
                                        }
                                }
                        }
                        return joiner.toString();
                }
                return "";
        }

        private CompletionStage<HttpResponse<String>> sendWithQuotaRetry(HttpRequest httpRequest) {
                return httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString())
                                .thenCompose(resp -> {
                                        if (resp.statusCode() != 429) {
                                                return CompletableFuture.completedFuture(resp);
                                        }
                                        long retryDelayMs = extractRetryDelayMillis(resp.body());
                                        if (retryDelayMs <= 0 || retryDelayMs > MAX_AUTO_RETRY_MS) {
                                                return CompletableFuture.completedFuture(resp);
                                        }
                                        log.warnf("Gemini quota hit; retrying once after %d ms", retryDelayMs);
                                        return CompletableFuture.supplyAsync(
                                                        () -> null,
                                                        CompletableFuture.delayedExecutor(retryDelayMs,
                                                                        TimeUnit.MILLISECONDS))
                                                        .thenCompose(ignored -> httpClient.sendAsync(httpRequest,
                                                                        HttpResponse.BodyHandlers.ofString()));
                                });
        }

        private RuntimeException buildHttpException(int statusCode, String body) {
                if (statusCode == 429) {
                        String message = extractErrorMessage(body);
                        long retryDelayMs = extractRetryDelayMillis(body);
                        if (retryDelayMs > 0) {
                                return new RuntimeException(
                                                "Gemini quota exceeded (429). Retry in about "
                                                                + Math.max(1L, retryDelayMs / 1000L)
                                                                + "s. " + message);
                        }
                        return new RuntimeException("Gemini quota exceeded (429). " + message);
                }
                String message = extractErrorMessage(body);
                return new RuntimeException("Gemini failed: " + statusCode + " - " + message);
        }

        private String extractErrorMessage(String body) {
                if (body == null || body.isBlank()) {
                        return "No error details";
                }
                try {
                        com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(body);
                        String message = root.path("error").path("message").asText("");
                        if (!message.isBlank()) {
                                return message.replace('\n', ' ').trim();
                        }
                } catch (Exception ignored) {
                        // fallback below
                }
                return body.length() > 240 ? body.substring(0, 240) + "..." : body;
        }

        private long extractRetryDelayMillis(String body) {
                if (body == null || body.isBlank()) {
                        return -1L;
                }
                try {
                        com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(body);
                        com.fasterxml.jackson.databind.JsonNode details = root.path("error").path("details");
                        if (details.isArray()) {
                                for (com.fasterxml.jackson.databind.JsonNode detail : details) {
                                        String retryDelay = detail.path("retryDelay").asText("");
                                        long parsed = parseRetryDelayMillis(retryDelay);
                                        if (parsed > 0) {
                                                return parsed;
                                        }
                                }
                        }
                        String message = root.path("error").path("message").asText("");
                        long fromMessage = parseRetryFromMessage(message);
                        if (fromMessage > 0) {
                                return fromMessage;
                        }
                } catch (Exception ignored) {
                        // ignore and try regex fallback
                }
                return parseRetryFromMessage(body);
        }

        private long parseRetryFromMessage(String message) {
                if (message == null || message.isBlank()) {
                        return -1L;
                }
                Matcher m = RETRY_SECONDS_PATTERN.matcher(message);
                if (!m.find()) {
                        return -1L;
                }
                try {
                        double seconds = Double.parseDouble(m.group(1));
                        return Math.max(1L, Math.round(seconds * 1000.0));
                } catch (Exception ignored) {
                        return -1L;
                }
        }

        private long parseRetryDelayMillis(String retryDelay) {
                if (retryDelay == null || retryDelay.isBlank()) {
                        return -1L;
                }
                String s = retryDelay.trim().toLowerCase();
                try {
                        if (s.endsWith("ms")) {
                                double ms = Double.parseDouble(s.substring(0, s.length() - 2).trim());
                                return Math.max(1L, Math.round(ms));
                        }
                        if (s.endsWith("s")) {
                                double sec = Double.parseDouble(s.substring(0, s.length() - 1).trim());
                                return Math.max(1L, Math.round(sec * 1000.0));
                        }
                } catch (Exception ignored) {
                        return -1L;
                }
                return -1L;
        }
}
