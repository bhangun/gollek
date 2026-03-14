package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Manages chat session state, history, and inference execution.
 */
@Dependent
public class ChatSessionManager {

    private final List<Message> history = new ArrayList<>();
    private String modelId;
    private String providerId;
    private String modelPathOverride;
    private String sessionId;
    
    // UI/Output hooks
    private ChatUIRenderer uiRenderer;
    private PrintWriter fileWriter;
    private boolean quiet;
    private boolean autoContinue = true;
    private int maxTokens = 256;
    private double temperature = 0.2;

    private final GollekSdk sdk;

    public ChatSessionManager(GollekSdk sdk) {
        this.sdk = sdk;
    }

    public void initialize(String modelId, String providerId, String modelPathOverride, boolean enableSession) {
        this.modelId = modelId;
        this.providerId = providerId;
        this.modelPathOverride = modelPathOverride;
        if (enableSession) {
            this.sessionId = UUID.randomUUID().toString();
        }
    }

    public void reset() {
        history.clear();
        this.sessionId = sessionId != null ? UUID.randomUUID().toString() : null;
    }

    public void switchProvider(String providerId) throws SdkException {
        this.providerId = providerId;
        sdk.setPreferredProvider(providerId);
        // If session is active, we might want to reset it or keep it depending on provider compatibility
        // For now, let's keep it but re-initialize if needed
    }

    public void setInferenceParams(boolean autoContinue, int maxTokens, double temperature) {
        this.autoContinue = autoContinue;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    public void setUIHooks(ChatUIRenderer uiRenderer, PrintWriter fileWriter, boolean quiet) {
        this.uiRenderer = uiRenderer;
        this.fileWriter = fileWriter;
        this.quiet = quiet;
    }

    public void addMessage(Message message) {
        history.add(message);
    }

    public List<Message> getHistory() {
        return history;
    }

    public void clearHistory() {
        history.clear();
    }

    public void executeInference(InferenceRequest.Builder reqBuilder, boolean stream, boolean enableJsonSse) {
        reqBuilder.model(modelId)
                .messages(new ArrayList<>(history))
                .preferredProvider(providerId);

        if (sessionId != null) {
            reqBuilder.parameter("session_id", sessionId);
        }
        if (modelPathOverride != null && !modelPathOverride.isBlank()) {
            reqBuilder.parameter("model_path", modelPathOverride);
        }

        InferenceRequest request = reqBuilder.build();

        try {
            if (stream && supportsStreaming(providerId)) {
                executeStreaming(request, enableJsonSse);
            } else {
                executeNonStreaming(request);
            }
        } catch (SdkException e) {
            uiRenderer.printError("Inference failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Inference interrupted", quiet);
        }
    }

    private void executeStreaming(InferenceRequest request, boolean enableJsonSse) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        StringBuilder fullResponse = new StringBuilder();

        if (!enableJsonSse) {
            uiRenderer.printAssistantPrefix(quiet, true);
        }

        sdk.streamCompletion(request)
                .subscribe().with(
                        chunk -> {
                            String delta = chunk.getDelta();
                            if (delta == null) return;
                            if (fileWriter != null) {
                                fileWriter.print(delta);
                                fileWriter.flush();
                            } else if (enableJsonSse) {
                                printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                            } else {
                                System.out.print(delta);
                                System.out.flush();
                            }
                            fullResponse.append(delta);
                            tokenCount.incrementAndGet();
                        },
                        error -> {
                            uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                            latch.countDown();
                        },
                        () -> {
                            long duration = System.currentTimeMillis() - startTime;
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else {
                                System.out.println();
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, quiet);
                            }

                            if (fileWriter != null) {
                                fileWriter.println();
                                fileWriter.printf("\n[Chunks: %d, Duration: %.2fs, Speed: %.2f t/s]%n",
                                        tokenCount.get(), duration / 1000.0, tps);
                            }

                            history.add(Message.assistant(fullResponse.toString()));
                            maybeContinueIfTruncated(fullResponse, true);
                            latch.countDown();
                        });

        latch.await();
    }

    private void executeNonStreaming(InferenceRequest request) throws SdkException {
        long startTime = System.currentTimeMillis();
        InferenceResponse response = sdk.createCompletion(request);
        long duration = System.currentTimeMillis() - startTime;
        String content = response.getContent();

        uiRenderer.printAssistantPrefix(quiet, false);
        System.out.println(content);
        double tps = response.getTokensUsed() / (Math.max(1, duration) / 1000.0);
        uiRenderer.printStats(response.getTokensUsed(), duration / 1000.0, tps, quiet);

        if (fileWriter != null) {
            fileWriter.println("\nAssistant: " + content);
            fileWriter.printf("\n[Duration: %.2fs, Tokens: %d]%n", duration / 1000.0, response.getTokensUsed());
        }

        history.add(Message.assistant(content));
        maybeContinueIfTruncated(new StringBuilder(content), false);
    }

    private void maybeContinueIfTruncated(StringBuilder responseBuilder, boolean stream) {
        if (autoContinue && looksTruncated(responseBuilder.toString())) {
            uiRenderer.printWarning("Response appears cut off; requesting continuation...", quiet);
            String continuation = null;
            try {
                continuation = requestContinuation();
            } catch (SdkException e) {
                uiRenderer.printError("Auto-continuation failed: " + e.getMessage(), quiet);
            }
            
            if (continuation != null && !continuation.isBlank()) {
                if (fileWriter != null) {
                    fileWriter.print(continuation);
                    fileWriter.flush();
                } else {
                    System.out.print(continuation);
                    System.out.flush();
                }
                responseBuilder.append(continuation);
                updateLastAssistantMessage(responseBuilder.toString());
            }
        }
    }

    private String requestContinuation() throws SdkException {
        List<Message> continuationMessages = new ArrayList<>(history);
        continuationMessages.add(Message.user("Continue from exactly where you stopped."));

        InferenceRequest request = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(modelId)
                .messages(continuationMessages)
                .temperature(temperature)
                .maxTokens(maxTokens)
                .preferredProvider(providerId)
                .cacheBypass(true)
                .build();

        return sdk.createCompletion(request).getContent();
    }

    private boolean looksTruncated(String content) {
        if (content == null || content.length() < 24) return false;
        String trimmed = content.trim();
        return !(trimmed.endsWith(".") || trimmed.endsWith("!") || trimmed.endsWith("?") || trimmed.endsWith("```") || trimmed.endsWith("</s>"));
    }

    private void updateLastAssistantMessage(String fullContent) {
        if (!history.isEmpty()) {
            int lastIdx = history.size() - 1;
            if (history.get(lastIdx).getRole() == Message.Role.ASSISTANT) {
                history.set(lastIdx, Message.assistant(fullContent));
            }
        }
    }

    private boolean supportsStreaming(String providerId) {
        return !("djl".equalsIgnoreCase(providerId) || "safetensor".equalsIgnoreCase(providerId));
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format("{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                requestId, created, model != null ? model : "", escapeJson(delta));
        System.out.println("data: " + payload);
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format("{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                requestId, created, model != null ? model : "");
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
    }

    private String escapeJson(String val) {
        if (val == null) return "";
        return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }
}
