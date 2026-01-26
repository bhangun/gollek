package tech.kayys.golek.provider.core.streaming;

import tech.kayys.golek.api.stream.StreamChunk;

/**
 * Processes raw chunks into StreamChunk objects
 */
public interface ChunkProcessor {

    /**
     * Process raw chunk data
     */
    StreamChunk process(String rawChunk, String requestId, int index);

    /**
     * Check if chunk indicates end of stream
     */
    boolean isEndOfStream(String rawChunk);

    /**
     * Extract finish reason from chunk
     */
    String extractFinishReason(String rawChunk);
}