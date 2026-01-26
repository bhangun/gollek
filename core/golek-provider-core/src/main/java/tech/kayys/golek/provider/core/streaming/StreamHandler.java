package tech.kayys.golek.provider.core.streaming;

import io.smallrye.mutiny.Multi;

/**
 * Handles streaming responses from providers
 */
public interface StreamHandler {

    /**
     * Handle server-sent events stream
     */
    Multi<String> handleSSE(String url, String data);

    /**
     * Handle WebSocket stream
     */
    Multi<String> handleWebSocket(String url, String data);

    /**
     * Close handler and cleanup resources
     */
    void close();
}