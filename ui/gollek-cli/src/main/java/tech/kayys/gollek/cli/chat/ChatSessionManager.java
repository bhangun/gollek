package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;

import java.io.PrintWriter;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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
    private boolean forceGguf;

    // UI/Output hooks
    private ChatUIRenderer uiRenderer;
    private PrintWriter fileWriter;
    private boolean quiet;
    private boolean autoContinue = true;
    private int maxTokens = 256;
    private double temperature = 0.2;

    private final GollekSdk sdk;

    // ── Audit trail ─────────────────────────────────────────────────────
    private final Instant sessionStartTime = Instant.now();
    private int totalRequests = 0;
    private int totalTokens = 0;
    private long totalDurationMs = 0;
    private int totalErrors = 0;
    private final Map<String, int[]> perModelStats = new LinkedHashMap<>();   // modelId -> [requests, tokens, errors]
    private final Map<String, int[]> perProviderStats = new LinkedHashMap<>(); // providerId -> [requests, tokens, errors]

    public ChatSessionManager(GollekSdk sdk) {
        this.sdk = sdk;
    }

    private long totalTtftMs = 0;
    private long totalTpotSumMs = 0; // Sum of average TPOT per request
    private long totalItlSumMs = 0; // Sum of average ITL per request
    private int streamedRequests = 0;

    /**
     * Session audit statistics snapshot.
     */
    public record SessionStats(
            Instant sessionStart,
            long sessionDurationSeconds,
            int totalRequests,
            int totalTokens,
            long totalDurationMs,
            int totalErrors,
            double avgTokensPerRequest,
            double avgTokensPerSecond,
            long avgTtftMs,
            long avgTpotMs,
            long avgItlMs,
            Map<String, int[]> perModelStats,
            Map<String, int[]> perProviderStats
    ) {}

    public SessionStats getSessionStats() {
        long sessionSecs = java.time.Duration.between(sessionStartTime, Instant.now()).toSeconds();
        double avgTpr = totalRequests > 0 ? (double) totalTokens / totalRequests : 0;
        double avgTps = totalDurationMs > 0 ? totalTokens / (totalDurationMs / 1000.0) : 0;
        long avgTtft = streamedRequests > 0 ? totalTtftMs / streamedRequests : 0;
        long avgTpot = streamedRequests > 0 ? totalTpotSumMs / streamedRequests : 0;
        long avgItl = streamedRequests > 0 ? totalItlSumMs / streamedRequests : 0;
        
        return new SessionStats(
                sessionStartTime, sessionSecs,
                totalRequests, totalTokens, totalDurationMs, totalErrors,
                avgTpr, avgTps, avgTtft, avgTpot, avgItl,
                new LinkedHashMap<>(perModelStats),
                new LinkedHashMap<>(perProviderStats)
        );
    }

    private void recordStats(int tokens, long durationMs, boolean error) {
        recordAdvancedStats(tokens, durationMs, error, 0, 0, 0, false);
    }

    private void recordAdvancedStats(int tokens, long durationMs, boolean error, long ttft, long tpot, long itl, boolean isStream) {
        totalRequests++;
        totalTokens += tokens;
        totalDurationMs += durationMs;
        if (error) totalErrors++;

        if (isStream && !error) {
            streamedRequests++;
            totalTtftMs += ttft;
            totalTpotSumMs += tpot;
            totalItlSumMs += itl;
        }

        String model = modelId != null ? modelId : "unknown";
        perModelStats.computeIfAbsent(model, k -> new int[3]);
        perModelStats.get(model)[0]++;
        perModelStats.get(model)[1] += tokens;
        if (error) perModelStats.get(model)[2]++;

        String provider = providerId != null ? providerId : "auto";
        perProviderStats.computeIfAbsent(provider, k -> new int[3]);
        perProviderStats.get(provider)[0]++;
        perProviderStats.get(provider)[1] += tokens;
        if (error) perProviderStats.get(provider)[2]++;
    }

    public void initialize(String modelId, String providerId, String modelPathOverride, boolean enableSession, boolean forceGguf) {
        this.modelId = modelId;
        this.providerId = providerId;
        this.modelPathOverride = modelPathOverride;
        this.forceGguf = forceGguf;
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
    }

    public void switchModel(String newModelId) {
        this.modelId = newModelId;
        // Reset model path override since user is explicitly choosing a new model
        this.modelPathOverride = null;
    }

    public String getModelId() {
        return modelId;
    }

    public String getProviderId() {
        return providerId;
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
        AtomicLong firstTokenTime = new AtomicLong(0);
        AtomicLong lastTokenTime = new AtomicLong(0);
        AtomicLong sumItl = new AtomicLong(0);
        StringBuilder fullResponse = new StringBuilder();

        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet && !enableJsonSse && !jsonMode) {
            spinner.start();
        }

        // NOTE: printAssistantPrefix is intentionally moved into the first-token
        // handler below so the spinner's CLEAR_LINE doesn't erase it.

        sdk.streamCompletion(request)
                .subscribe().with(
                        chunk -> {
                            long now = System.currentTimeMillis();
                            if (firstTokenTime.compareAndSet(0, now)) {
                                spinner.stop();
                                // Print prefix only after spinner has cleared the line
                                if (!enableJsonSse && !jsonMode) {
                                    uiRenderer.printAssistantPrefix(quiet, true);
                                }
                            } else {
                                sumItl.addAndGet(now - lastTokenTime.get());
                            }
                            lastTokenTime.set(now);

                            if (!jsonMode && chunk.metadata() != null && chunk.metadata().containsKey("hardware")) {
                                String hw = (String) chunk.metadata().get("hardware");
                                uiRenderer.printHardwareInfo(hw, quiet);
                            }
                            String delta = chunk.getDelta();
                            if (delta == null)
                                return;
                            
                            // Detect corruption early - stop processing corrupted chunks
                            if (isResponseCorrupted(delta)) {
                                fullResponse.append("[CORRUPTED_CHUNK_DETECTED]");
                                return;
                            }
                            
                            // Strip model control tokens before display
                            String cleanDelta = stripSpecialTokens(delta);
                            fullResponse.append(delta);
                            tokenCount.incrementAndGet();
                            if (cleanDelta.isEmpty()) {
                                return;
                            }
                            if (fileWriter != null) {
                                fileWriter.print(cleanDelta);
                                fileWriter.flush();
                            } else if (enableJsonSse) {
                                printOpenAiSseDelta(request.getRequestId(), request.getModel(), cleanDelta);
                            } else if (jsonMode) {
                                // In streaming JSON mode, we don't print delta to console if not SSE
                            } else {
                                System.out.print(cleanDelta);
                                System.out.flush();
                            }
                        },
                        error -> {
                            long errDuration = System.currentTimeMillis() - startTime;
                            spinner.stop();
                            recordStats(tokenCount.get(), errDuration, true);
                            uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                            latch.countDown();
                        },
                        () -> {
                            spinner.stop(); // no-op if already stopped by first token
                            long duration = System.currentTimeMillis() - startTime;
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            
                            String finalResponse = fullResponse.toString();
                            
                            // Check if response appears corrupted
                            if (isResponseCorrupted(finalResponse)) {
                                uiRenderer.printError("Response appears corrupted or malformed. This may indicate an issue with the model or inference engine.", quiet);
                                recordStats(tokenCount.get(), duration, true);
                                history.add(Message.assistant("[Response corrupted - inference error]"));
                                latch.countDown();
                                return;
                            }
                            
                            long ttft = firstTokenTime.get() > 0 ? firstTokenTime.get() - startTime : 0;
                            long tpot = tokenCount.get() > 1 ? (lastTokenTime.get() - firstTokenTime.get()) / (tokenCount.get() - 1) : 0;
                            long avgItl = tokenCount.get() > 1 ? sumItl.get() / (tokenCount.get() - 1) : 0;

                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else if (request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm) {
                                System.out.println();
                                printJsonModeResponse(request, finalResponse, tokenCount.get(), duration / 1000.0, tps);
                            } else {
                                System.out.println();
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, quiet);
                            }

                            if (fileWriter != null) {
                                fileWriter.println();
                                fileWriter.printf("\n[Chunks: %d, Duration: %.2fs, Speed: %.2f t/s]%n",
                                        tokenCount.get(), duration / 1000.0, tps);
                            }

                            recordAdvancedStats(tokenCount.get(), duration, false, ttft, tpot, avgItl, true);
                            history.add(Message.assistant(finalResponse));
                            maybeContinueIfTruncated(fullResponse, true);
                            latch.countDown();
                        });

        latch.await();
    }

    private void executeNonStreaming(InferenceRequest request) throws SdkException {
        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet) spinner.start();
        long startTime = System.currentTimeMillis();
        InferenceResponse response;
        try {
            response = sdk.createCompletion(request);
        } finally {
            spinner.stop();
        }
        long duration = System.currentTimeMillis() - startTime;
        String content = response.getContent();
        
        // Check if response appears corrupted
        if (isResponseCorrupted(content)) {
            uiRenderer.printError("Response appears corrupted or malformed. This may indicate an issue with the model or inference engine.", quiet);
            recordStats(response.getTokensUsed(), duration, true);
            history.add(Message.assistant("[Response corrupted - inference error]"));
            return;
        }

        double tps = response.getTokensUsed() / (Math.max(1, duration) / 1000.0);
        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        if (jsonMode) {
            printJsonModeResponse(request, content, response.getTokensUsed(), duration / 1000.0, tps);
        } else {
            uiRenderer.printAssistantPrefix(quiet, false);
            System.out.println(content);
            uiRenderer.printStats(response.getTokensUsed(), duration / 1000.0, tps, quiet);
        }

        if (fileWriter != null) {
            fileWriter.println("\nAssistant: " + content);
            fileWriter.printf("\n[Duration: %.2fs, Tokens: %d]%n", duration / 1000.0, response.getTokensUsed());
        }

        recordStats(response.getTokensUsed(), duration, false);
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
        if (content == null || content.length() < 24)
            return false;
        // If the model emitted an EOS token, the response is complete
        if (hasEndOfSequenceToken(content)) {
            return false;
        }
        String trimmed = stripSpecialTokens(content).trim();
        if (trimmed.isEmpty()) {
            return false;
        }
        return !(trimmed.endsWith(".") || trimmed.endsWith("!")
                || trimmed.endsWith("?") || trimmed.endsWith("```")
                || trimmed.endsWith("\n"));
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
        // Both GGUF and Safetensor now support streaming
        return true;
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

    private String escapeJson(String val) {
        if (val == null)
            return "";
        return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    // ── Special token handling ──────────────────────────────────────────

    private static final java.util.regex.Pattern SPECIAL_TOKEN_PATTERN =
            java.util.regex.Pattern.compile(
                    "<\\|im_start\\|>(?:system|user|assistant)?|<\\|im_end\\|>|<\\|endoftext\\|>|</s>|<s>|\\[INST\\]|\\[/INST\\]|<\\|eot_id\\|>|<\\|start_header_id\\|>.*?<\\|end_header_id\\|>");

    /**
     * Strip LLM control/special tokens from a streamed delta so they
     * are never shown to the user in the terminal.
     */
    private String stripSpecialTokens(String delta) {
        if (delta == null || delta.isEmpty()) {
            return "";
        }
        String cleaned = SPECIAL_TOKEN_PATTERN.matcher(delta).replaceAll("");
        return sanitizeResponse(cleaned);
    }

    /**
     * Sanitize response by filtering out invalid/corrupted characters that indicate
     * token decoding errors. This prevents garbled output from model hallucinations
     * or native binding issues.
     */
    private String sanitizeResponse(String response) {
        if (response == null || response.isEmpty()) {
            return "";
        }
        
        // First, sanitize by checking if response is corrupted
        if (isResponseCorrupted(response)) {
            // Attempt to extract readable portions
            return extractReadablePortion(response);
        }
        
        StringBuilder cleaned = new StringBuilder();
        for (char c : response.toCharArray()) {
            // Allow printable ASCII, common Unicode ranges, and whitespace
            if (isPrintableChar(c)) {
                cleaned.append(c);
            } else if (c > 0x007F) {
                // For non-ASCII, be more lenient - only filter obvious corruptions
                if (!Character.isISOControl(c) && !isLikelyCorruptedChar(c)) {
                    cleaned.append(c);
                }
            }
        }
        return cleaned.toString();
    }

    /**
     * Check if a character looks like it's from a corrupted token.
     */
    private boolean isLikelyCorruptedChar(char c) {
        // Detect replacement character which indicates failed UTF-8 decode
        if (c == '\uFFFD') return true;
        
        // Detect certain block characters that appear in corrupted sequences
        if (Character.getType(c) == Character.SURROGATE) return true;
        
        // Allow CJK, Cyrillic, Arabic, Devanagari and other major scripts
        int block = Character.getType(c);
        return Character.getType(c) == Character.UNASSIGNED;
    }

    /**
     * Extract readable portion from potentially corrupted response.
     */
    private String extractReadablePortion(String response) {
        // Split by common corruption patterns and take the longest readable segment
        String[] segments = response.split("[\\p{C}]+"); // Split on control characters
        String longest = "";
        for (String seg : segments) {
            if (seg.length() > longest.length()) {
                longest = seg;
            }
        }
        return longest.isEmpty() ? "[Response corrupted]" : longest;
    }

    private boolean isPrintableChar(char c) {
        int type = Character.getType(c);
        return !Character.isISOControl(c) || c == '\t' || c == '\n' || c == '\r';
    }

    /**
     * Detect if a response appears corrupted based on patterns that indicate
     * token decoding failures. More aggressive detection for early termination.
     */
    private boolean isResponseCorrupted(String response) {
        if (response == null || response.isEmpty()) {
            return false;
        }
        
        // For short chunks/responses, use stricter thresholds
        int suspiciousCount = 0;
        int nonAsciiCount = 0;
        int totalChars = response.length();
        
        for (int i = 0; i < response.length(); i++) {
            char c = response.charAt(i);
            
            // Count non-ASCII characters
            if (c > 0x007F) {
                nonAsciiCount++;
            }
            
            // Count suspicious patterns
            if (Character.isISOControl(c) && c != '\t' && c != '\n' && c != '\r') {
                suspiciousCount++;
            } else if (Character.isDefined(c) && Character.getType(c) == Character.UNASSIGNED) {
                suspiciousCount++;
            } else if (c == '\uFFFD') {
                suspiciousCount++;
            } else if (Character.getType(c) == Character.SURROGATE) {
                suspiciousCount++;
            }
        }
        
        // Check for patterns that strongly indicate corruption:
        // 1. More than 5% control characters
        double controlCharRatio = (suspiciousCount * 100.0 / totalChars);
        if (controlCharRatio > 5.0) {
            return true;
        }
        
        // 2. High non-ASCII ratio indicates potential corruption
        // For small chunks: > 30% non-ASCII mixed with ASCII is suspicious
        // For normal responses: > 50% is OK (for multilingual)
        if (totalChars < 50) {
            // Small chunks: stricter threshold
            if (nonAsciiCount > totalChars * 0.3) {
                return true;
            }
        } else if (totalChars < 200) {
            // Medium chunks: moderate threshold
            if (nonAsciiCount > totalChars * 0.4) {
                return true;
            }
        }
        
        // 3. Check for patterns like multiple scripts mixed in unnatural ways
        if (hasMultipleScriptMixing(response)) {
            return true;
        }
        
        return false;
    }

    /**
     * Detect unnatural mixing of multiple writing systems which indicates corruption.
     */
    private boolean hasMultipleScriptMixing(String text) {
        int latinCount = 0, cjkCount = 0, arabicCount = 0, thaiCount = 0, devanagariCount = 0;
        
        for (char c : text.toCharArray()) {
            if ((c >= 'A' && c <= 'z') || (c >= '0' && c <= '9')) {
                latinCount++;
            } else if (c >= 0x4E00 && c <= 0x9FFF) {
                cjkCount++;  // CJK Unified Ideographs
            } else if ((c >= 0xAC00 && c <= 0xD7AF) || (c >= 0x1100 && c <= 0x11FF)) {
                cjkCount++; // Hangul
            } else if (c >= 0x0600 && c <= 0x06FF) {
                arabicCount++;  // Arabic
            } else if (c >= 0x0E00 && c <= 0x0E7F) {
                thaiCount++;  // Thai
            } else if (c >= 0x0900 && c <= 0x097F) {
                devanagariCount++;  // Devanagari
            }
        }
        
        // If we see mixing of 3+ different script families in short text, it's likely corruption
        int scriptFamilies = (latinCount > 0 ? 1 : 0) + (cjkCount > 0 ? 1 : 0) + 
                             (arabicCount > 0 ? 1 : 0) + (thaiCount > 0 ? 1 : 0) + 
                             (devanagariCount > 0 ? 1 : 0);
        
        if (scriptFamilies >= 3 && text.length() < 100) {
            // Mixed scripts in short text is highly suspicious
            return true;
        }
        
        return false;
    }

    /**
     * Check whether the response contains a recognised end-of-sequence
     * token, meaning the model intentionally stopped generating.
     */
    private boolean hasEndOfSequenceToken(String content) {
        if (content == null) return false;
        return SPECIAL_TOKEN_PATTERN.matcher(content).find();
    }
}
