package tech.kayys.gollek.spi.inference;

import java.time.Instant;
import java.util.Map;

/**
 * Interface for inference responses.
 */
public interface InferenceResponseInterface {
    String getRequestId();

    String getContent();

    String getModel();

    int getTokensUsed();

    int getInputTokens();

    int getOutputTokens();

    long getDurationMs();

    Instant getTimestamp();

    Map<String, Object> getMetadata();

    boolean isStreaming();
}