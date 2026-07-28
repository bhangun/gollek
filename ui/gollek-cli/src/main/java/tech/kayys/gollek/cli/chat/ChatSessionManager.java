package tech.kayys.gollek.cli.chat;

import io.quarkus.arc.Arc;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.sdk.session.ChatSession;
import tech.kayys.gollek.sdk.session.ChatSessionImpl;
import tech.kayys.gollek.sdk.session.ChatSessionFactory;

import java.io.PrintWriter;
import java.time.Duration;
import java.util.*;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CLI-specific wrapper around SDK ChatSession.
 * Manages UI rendering and CLI hooks.
 */
@Dependent
public class ChatSessionManager {

    private ChatSession sdkSession;
    private final ChatSessionFactory sessionFactory;
    private String modelId;
    private String providerId;
    private String modelPathOverride;
    private boolean enableSession;

    // UI/Output hooks
    private ChatUIRenderer uiRenderer;
    private PrintWriter fileWriter;
    private boolean quiet;
    private boolean autoContinue = true;
    private int maxTokens = 256;
    private double temperature = 0.2;
    private volatile String lastExecutionRoute = "none";
    private volatile String lastProviderDescriptor = "unknown";
    private volatile String lastExecutionError = null;
    private volatile Map<String, Object> lastExecutionMetadata = Map.of();
    private volatile InferenceRequest lastPreparedRequest;
    private volatile boolean lastPreparedRequestStreaming;
    private volatile boolean lastPreparedRequestJsonSse;

    private final GollekSdk sdk;

    @Inject
    public ChatSessionManager(GollekSdk sdk, ChatSessionFactory sessionFactory) {
        this.sdk = sdk;
        this.sessionFactory = sessionFactory;
    }

    public void initialize(String modelId, String providerId, String modelPathOverride, boolean enableSession, boolean forceGguf) {
        this.modelId = modelId;
        this.providerId = providerId;
        this.modelPathOverride = modelPathOverride;
        this.enableSession = enableSession;
        
        this.sdkSession = createSession(modelId, providerId);
    }

    public void reset() {
        if (sdkSession != null) {
            sdkSession.reset();
        }
    }

    public void switchProvider(String providerId) throws SdkException {
        this.providerId = providerId;
        
    }

    public void switchModel(String newModelId) {
        this.modelId = newModelId;
        this.modelPathOverride = null;
        this.sdkSession = createSession(newModelId, providerId);
    }

    public String getModelId() {
        return modelId;
    }

    public String getProviderId() {
        return providerId;
    }

    public String getSessionId() {
        return sdkSession != null ? sdkSession.getSessionId() : null;
    }

    public boolean isSessionEnabled() {
        return sessionEnabledForExecution();
    }

    private ChatSession createSession(String modelId, String providerId) {
        return new ChatSessionImpl(sdk, modelId, providerId, enableSession);
    }

    public void setInferenceParams(boolean autoContinue, int maxTokens, double temperature) {
        this.autoContinue = autoContinue;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        
        if (sdkSession != null) {
            sdkSession.setAutoContinue(autoContinue);
            Map<String, Object> params = new HashMap<>();
            params.put("max_tokens", maxTokens);
            params.put("temperature", temperature);
            sdkSession.setDefaultParameters(params);
        }
    }

    public void setSystemPrompt(String systemPrompt) {
        if (sdkSession != null) {
            sdkSession.setSystemPrompt(systemPrompt);
        }
    }

    public void setUIHooks(ChatUIRenderer uiRenderer, PrintWriter fileWriter, boolean quiet) {
        this.uiRenderer = uiRenderer;
        this.fileWriter = fileWriter;
        this.quiet = quiet;
    }

    public void addMessage(Message message) {
        if (sdkSession != null) {
            sdkSession.addMessage(message);
        }
    }

    public List<Message> getHistory() {
        return sdkSession != null ? sdkSession.getHistory() : List.of();
    }

    public void clearHistory() {
        if (sdkSession != null) {
            sdkSession.reset();
        }
    }

    public void executeInference(InferenceRequest.Builder reqBuilder, boolean stream, boolean enableJsonSse) {
        if (sdkSession == null) {
            uiRenderer.printError("Session not initialized", quiet);
            return;
        }

        reqBuilder.model(modelId)
                .preferredProvider(providerId)
                .maxTokens(maxTokens)
                .temperature(temperature);

        if (modelPathOverride != null && !modelPathOverride.isBlank()) {
            reqBuilder.parameter("model_path", modelPathOverride);
        }

        InferenceRequest request = reqBuilder.build();
        InferenceRequest preparedRequest = ensureSessionBinding(activeSession().prepareRequest(request));
        rememberPreparedRequest(preparedRequest, stream, enableJsonSse);

        try {
            executePreparedRequest(preparedRequest, stream, enableJsonSse, false);
        } catch (SdkException e) {
            uiRenderer.printError("Inference failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Inference interrupted", quiet);
        }
    }

    public void retryLastRequest() {
        if (lastPreparedRequest == null) {
            uiRenderer.printError("No previous request is available to retry", quiet);
            return;
        }
        if (!shouldUseDirectProviderPath(lastPreparedRequest)) {
            uiRenderer.printError("Retry is currently only supported on the local direct provider path", quiet);
            return;
        }

        InferenceRequest retryRequest = ensureSessionBinding(lastPreparedRequest.toBuilder()
                .requestId(java.util.UUID.randomUUID().toString())
                .build());
        try {
            executePreparedRequest(retryRequest, lastPreparedRequestStreaming, lastPreparedRequestJsonSse, true);
        } catch (SdkException e) {
            uiRenderer.printError("Retry failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Retry interrupted", quiet);
        }
    }

    private void executePreparedRequest(
            InferenceRequest preparedRequest,
            boolean stream,
            boolean enableJsonSse,
            boolean replaceLastAssistantResponse) throws InterruptedException, SdkException {
        String plannedRoute = stream ? "sdk-session-stream" : "sdk-session-sync";
        recordExecutionSnapshot(
                plannedRoute,
                providerId != null ? providerId : "sdk",
                Map.of(
                        "execution_stage", "planned",
                        "retry", replaceLastAssistantResponse));

        if (stream) {
            executeStreaming(preparedRequest, enableJsonSse);
        } else {
            executeNonStreaming(preparedRequest);
        }
    }

    private void rememberPreparedRequest(InferenceRequest preparedRequest, boolean stream, boolean enableJsonSse) {
        this.lastPreparedRequest = preparedRequest;
        this.lastPreparedRequestStreaming = stream;
        this.lastPreparedRequestJsonSse = enableJsonSse;
    }

    private void executeStreaming(InferenceRequest request, boolean enableJsonSse) throws InterruptedException, SdkException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        AtomicLong firstTokenTime = new AtomicLong(0);
        AtomicLong lastTokenTime = new AtomicLong(0);
        AtomicLong sumItl = new AtomicLong(0);
        java.util.concurrent.atomic.AtomicReference<Map<String, Object>> metadataRef = new java.util.concurrent.atomic.AtomicReference<>(Map.of());
        StringBuilder fullResponse = new StringBuilder();
        boolean[] quantCachePrinted = { false };

        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet && !enableJsonSse && !jsonMode) {
            spinner.start();
        }

        sdkSession.stream(request)
                .subscribe().with(
                        chunk -> {
                            if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                metadataRef.set(Map.copyOf(chunk.metadata()));
                            }
                            if (!quantCachePrinted[0] && chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                printQuantCacheInfo(chunk.metadata(), enableJsonSse || jsonMode);
                                quantCachePrinted[0] = true;
                            }
                            String delta = chunk.getDelta();
                            if (delta == null) return;
                            
                            fullResponse.append(delta);
                            tokenCount.incrementAndGet();
                            
                            if (delta.isEmpty()) return;

                            long now = System.currentTimeMillis();
                            if (firstTokenTime.compareAndSet(0, now)) {
                                spinner.stop();
                                if (!enableJsonSse && !jsonMode) {
                                    uiRenderer.printAssistantPrefix(quiet, true);
                                }
                            } else {
                                sumItl.addAndGet(now - lastTokenTime.get());
                            }
                            lastTokenTime.set(now);
                            
                            if (fileWriter != null) {
                                fileWriter.print(delta);
                                fileWriter.flush();
                            } else if (enableJsonSse) {
                                printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                            } else if (!jsonMode) {
                                System.out.print(delta);
                                System.out.flush();
                            }
                        },
                        error -> {
                            spinner.stop();
                            recordExecutionFailure("sdk-session-stream", providerId != null ? providerId : "sdk", summarizeThrowable(error));
                            uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                            latch.countDown();
                        },
                        () -> {
                            spinner.stop();
                            long duration = System.currentTimeMillis() - startTime;
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            recordExecutionSnapshot("sdk-session-stream", providerId != null ? providerId : "sdk", metadataRef.get());
                            
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else if (jsonMode) {
                                System.out.println();
                                printJsonModeResponse(request, fullResponse.toString(), tokenCount.get(), duration / 1000.0, tps);
                            } else {
                                System.out.println();
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps,
                                        ttftMillis(metadataRef.get(), startTime, firstTokenTime), quiet);
                            }

                            latch.countDown();
                        });

        latch.await();
    }


    private void executeNonStreaming(InferenceRequest request) throws SdkException {
        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet) spinner.start();
        long startTime = System.currentTimeMillis();
        
        try {
            InferenceResponse response = sdkSession.send(request);
            long duration = System.currentTimeMillis() - startTime;
            double tps = response.getTokensUsed() / (Math.max(1, duration) / 1000.0);
            recordExecutionSnapshot("sdk-session-sync", providerId != null ? providerId : "sdk", response.getMetadata());
            
            boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

            if (jsonMode) {
                printJsonModeResponse(request, response.getContent(), response.getTokensUsed(), duration / 1000.0, tps);
            } else {
                printQuantCacheInfo(response.getMetadata(), false);
                uiRenderer.printAssistantPrefix(quiet, false);
                System.out.println(response.getContent());
                uiRenderer.printStats(response.getTokensUsed(), duration / 1000.0, tps,
                        ttftMillis(response.getMetadata()), quiet);
            }
        } catch (SdkException e) {
            recordExecutionFailure("sdk-session-sync", providerId != null ? providerId : "sdk", summarizeThrowable(e));
            throw e;
        } finally {
            spinner.stop();
        }
    }


    private static Double ttftMillis(Map<String, Object> metadata) {
        return metadataDouble(metadata, "bench.ttft_ms");
    }

    private static Double ttftMillis(Map<String, Object> metadata, long startTimeMs, AtomicLong firstTokenTimeMs) {
        Double metadataTtft = ttftMillis(metadata);
        if (metadataTtft != null) {
            return metadataTtft;
        }
        long first = firstTokenTimeMs != null ? firstTokenTimeMs.get() : 0L;
        if (first <= 0L || first < startTimeMs) {
            return null;
        }
        return (double) (first - startTimeMs);
    }

    private static Double metadataDouble(Map<String, Object> metadata, String key) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        Object value = metadata.get(key);
        return value instanceof Number number ? number.doubleValue() : null;
    }

    private void printQuantCacheInfo(Map<String, Object> metadata, boolean suppressForStructuredOutput) {
        if (suppressForStructuredOutput || metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("quant_cache_state");
        if (state == null) {
            return;
        }
        StringBuilder line = new StringBuilder("Quant cache: ").append(state);
        Object path = metadata.get("quant_cache_path");
        if (path != null) {
            line.append(" (").append(path).append(")");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format(
                "{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                requestId, created, model != null ? model : "", escapeJson(delta));
        System.out.println("data: " + payload);
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format(
                "{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                requestId, created, model != null ? model : "");
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
    }

    private void printJsonModeResponse(InferenceRequest request, String content, int tokens, double duration, double tps) {
        String lastUserPrompt = "";
        List<Message> history = getHistory();
        if (!history.isEmpty()) {
            for (int i = history.size() - 1; i >= 0; i--) {
                if (history.get(i).getRole() == Message.Role.USER) {
                    lastUserPrompt = history.get(i).getContent();
                    break;
                }
            }
        }

        String json = String.format(
                "{\"prompt\":\"%s\",\"model\":\"%s\",\"response\":\"%s\",\"stats\":{\"tokens\":%d,\"duration_s\":%.2f,\"speed_tps\":%.2f}}",
                escapeJson(lastUserPrompt),
                request.getModel() != null ? request.getModel() : "",
                escapeJson(content),
                tokens, duration, tps);
        System.out.println(json);
    }

    public SessionStats getSessionStats() {
        var stats = sdkSession != null ? sdkSession.getStats() : null;
        if (stats == null) {
            return new SessionStats(java.time.Instant.now(), 0, 0, 0, 0, 0, 0, 0, 0, java.util.Map.of(), java.util.Map.of());
        }
        return new SessionStats(
                stats.sessionStart(),
                stats.sessionDurationSeconds(),
                stats.totalRequests(),
                stats.totalTokens(),
                stats.totalDurationMs(),
                stats.totalErrors(),
                0, 0, 0, // Placeholder for TTFT, TPOT, ITL
                stats.perModelStats(),
                stats.perProviderStats()
        );
    }

    public ExecutionDiagnostics getExecutionDiagnostics() {
        Object registry = null;
        return new ExecutionDiagnostics(
                lastExecutionRoute,
                lastProviderDescriptor,
                lastExecutionError,
                lastExecutionMetadata,
                false,
                false);
    }

    public record SessionStats(
            java.time.Instant sessionStart,
            long sessionDurationSeconds,
            int totalRequests,
            int totalTokens,
            long totalDurationMs,
            int totalErrors,
            long avgTtftMs,
            long avgTpotMs,
            long avgItlMs,
            java.util.Map<String, int[]> perModelStats,
            java.util.Map<String, int[]> perProviderStats
    ) {
        public double avgTokensPerRequest() {
            return totalRequests == 0 ? 0 : (double) totalTokens / totalRequests;
        }

        public double avgTokensPerSecond() {
            return totalDurationMs == 0 ? 0 : (totalTokens / (totalDurationMs / 1000.0));
        }
    }

    public record ExecutionDiagnostics(
            String route,
            String providerDescriptor,
            String lastError,
            Map<String, Object> metadata,
            boolean providerRegistryAvailable,
            boolean providerRegistered
    ) {
    }

    private String escapeJson(String val) {
        if (val == null)
            return "";
        return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    private boolean shouldUseDirectProviderPath(InferenceRequest request) { return false; }


    private Object requireProvider() { return null; }

    private Object requireStreamingProvider() { return null; }

    private ChatSessionImpl activeSession() {
        if (sdkSession instanceof ChatSessionImpl impl) {
            return impl;
        }
        throw new IllegalStateException("Chat session is not using the CLI-managed ChatSessionImpl");
    }

    private boolean sessionEnabledForExecution() {
        if (sdkSession instanceof ChatSessionImpl impl) {
            return impl.isSessionEnabled();
        }
        return enableSession;
    }

    private InferenceRequest ensureSessionBinding(InferenceRequest request) {
        boolean sessionEnabled = sessionEnabledForExecution();
        String effectiveSessionId = request.getSessionId()
                .filter(id -> !id.isBlank())
                .orElse(sessionEnabled ? sdkSession.getSessionId() : null);

        Map<String, Object> params = new HashMap<>(request.getParameters());
        params.put("chat_session_enabled", sessionEnabled);
        if (effectiveSessionId != null && !effectiveSessionId.isBlank()) {
            params.put("session_id", effectiveSessionId);
        } else {
            params.remove("session_id");
        }

        boolean metadataMatches = Objects.equals(params, request.getParameters());
        boolean sessionMatches = Objects.equals(effectiveSessionId, request.getSessionId().orElse(null));
        if (metadataMatches && sessionMatches) {
            return request;
        }

        InferenceRequest.Builder builder = request.toBuilder()
                .parameters(params);
        if (effectiveSessionId != null && !effectiveSessionId.isBlank()) {
            builder.sessionId(effectiveSessionId);
        }
        return builder.build();
    }

    private SdkException directProviderFailure(String mode, RuntimeException error) { return new SdkException("error", error); }

    private String describeDirectProvider() { return "sdk"; }

    private String summarizeThrowable(Throwable error) {
        if (error == null) {
            return "unknown error";
        }
        StringBuilder summary = new StringBuilder();
        Throwable current = error;
        int depth = 0;
        while (current != null && depth < 4) {
            if (depth > 0) {
                summary.append(" <- ");
            }
            summary.append(current.getClass().getSimpleName());
            String message = current.getMessage();
            if (message != null && !message.isBlank()) {
                summary.append(": ").append(message.replace('\n', ' ').replace('\r', ' '));
            }
            current = current.getCause();
            depth++;
        }
        return summary.toString();
    }

    private void recordExecutionSnapshot(String route, String providerDescriptor, Map<String, Object> metadata) {
        this.lastExecutionRoute = route != null ? route : "unknown";
        this.lastProviderDescriptor = providerDescriptor != null ? providerDescriptor : "unknown";
        this.lastExecutionError = null;
        this.lastExecutionMetadata = metadata == null || metadata.isEmpty() ? Map.of() : Map.copyOf(metadata);
    }

    private void recordExecutionFailure(String route, String providerDescriptor, String errorSummary) {
        this.lastExecutionRoute = route != null ? route : "unknown";
        this.lastProviderDescriptor = providerDescriptor != null ? providerDescriptor : "unknown";
        this.lastExecutionError = errorSummary;
    }

    private Object providerRegistry() { return null; }
}
