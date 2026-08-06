package tech.kayys.gollek.sdk.session;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.sdk.exception.SdkException;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Manages an interactive chat session with history and state tracking.
 */
public interface ChatSession extends AutoCloseable {

    String getSessionId();

    String getModelId();

    String getProviderId();

    /**
     * Add a message to the session history.
     */
    void addMessage(Message message);

    /**
     * Get the current session history.
     */
    List<Message> getHistory();

    /**
     * Clear session history and reset session ID if applicable.
     */
    void reset();

    /**
     * Set default inference parameters for this session.
     */
    void setDefaultParameters(Map<String, Object> parameters);

    /**
     * Set the system prompt for this session.
     */
    void setSystemPrompt(String systemPrompt);

    /**
     * Enable or disable automatic continuation for truncated responses.
     */
    void setAutoContinue(boolean autoContinue);

    /**
     * Execute a synchronous completion request.
     * Automatically enriches the request with session history and parameters.
     */
    InferenceResponse send(String prompt) throws SdkException;

    /**
     * Execute a synchronous completion request with custom parameters.
     */
    InferenceResponse send(InferenceRequest request) throws SdkException;

    /**
     * Execute a streaming completion request.
     * Automatically enriches the request with session history and parameters.
     */
    Multi<StreamingInferenceChunk> stream(String prompt) throws SdkException;

    /**
     * Execute a streaming completion request with custom parameters.
     */
    Multi<StreamingInferenceChunk> stream(InferenceRequest request) throws SdkException;

    /**
     * Get session statistics.
     */
    SessionStats getStats();

    /**
     * Session statistics DTO.
     */
    record SessionStats(
            Instant sessionStart,
            long sessionDurationSeconds,
            int totalRequests,
            int totalTokens,
            long totalDurationMs,
            int totalErrors,
            Map<String, int[]> perModelStats,
            Map<String, int[]> perProviderStats
    ) {}

    @Override
    void close();
}
