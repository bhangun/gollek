package tech.kayys.gollek.spi.provider;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Utility methods for immutable {@link ProviderRequest} manipulation.
 *
 * <p>Centralises the copy logic so providers can call:
 * <pre>{@code
 *   ProviderRequest dispatched = ProviderRequests.withModel(original, resolvedPath);
 * }</pre>
 * instead of the error-prone positional constructor copy.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // Replace only the model field (resolved path):
 * ProviderRequest dispatched = ProviderRequests.withModel(request, path.toString());
 *
 * // Custom builder (add streaming flag, change timeout):
 * ProviderRequest patched = ProviderRequests.copyOf(request)
 *         .streaming(true)
 *         .timeout(Duration.ofMinutes(2))
 *         .build();
 * }</pre>
 */
public final class ProviderRequests {

    private ProviderRequests() {}

    // ── Static convenience factories ──────────────────────────────────────────

    /**
     * Create a copy of {@code source} with the model field replaced by
     * {@code newModel}.  All other fields are preserved unchanged.
     *
     * @param source   original request
     * @param newModel resolved model path or identifier
     * @return new immutable request
     */
    public static ProviderRequest withModel(ProviderRequest source, String newModel) {
        Objects.requireNonNull(source,   "source");
        Objects.requireNonNull(newModel, "newModel");
        return copyOf(source).model(newModel).build();
    }

    /**
     * Start a fluent builder pre-populated with all fields from {@code source}.
     *
     * @param source request to copy from
     * @return a mutable builder pre-filled from {@code source}
     */
    public static CopyBuilder copyOf(ProviderRequest source) {
        Objects.requireNonNull(source, "source");
        return new CopyBuilder(source);
    }

    // ── CopyBuilder ───────────────────────────────────────────────────────────

    /**
     * Fluent builder for creating modified copies of a {@link ProviderRequest}.
     */
    public static final class CopyBuilder {

        private String     requestId;
        private String     model;
        private List<tech.kayys.gollek.spi.Message> messages;
        private Map<String, Object> parameters;
        private List<tech.kayys.gollek.spi.tool.ToolDefinition> tools;
        private Object     toolChoice;
        private boolean    streaming;
        private Duration   timeout;
        private String     userId;
        private String     sessionId;
        private String     traceId;
        private String     apiKey;
        private Map<String, Object> metadata;

        private CopyBuilder(ProviderRequest src) {
            this.requestId  = src.getRequestId();
            this.model      = src.getModel();
            this.messages   = src.getMessages();
            this.parameters = src.getParameters();
            this.tools      = src.getTools();
            this.toolChoice = src.getToolChoice();
            this.streaming  = src.isStreaming();
            this.timeout    = src.getTimeout();
            this.userId     = src.getUserId().orElse(null);
            this.sessionId  = src.getSessionId().orElse(null);
            this.traceId    = src.getTraceId().orElse(null);
            this.apiKey     = src.getApiKey().orElse(null);
            this.metadata   = src.getMetadata();
        }

        public CopyBuilder model(String model)     { this.model = model; return this; }
        public CopyBuilder requestId(String id)    { this.requestId = id; return this; }
        public CopyBuilder streaming(boolean s)    { this.streaming = s; return this; }
        public CopyBuilder timeout(Duration t)     { this.timeout = t; return this; }
        public CopyBuilder userId(String uid)      { this.userId = uid; return this; }
        public CopyBuilder sessionId(String sid)   { this.sessionId = sid; return this; }
        public CopyBuilder traceId(String tid)     { this.traceId = tid; return this; }
        public CopyBuilder apiKey(String key)      { this.apiKey = key; return this; }

        public CopyBuilder parameters(Map<String, Object> p) { this.parameters = p; return this; }
        public CopyBuilder metadata(Map<String, Object> m)   { this.metadata = m; return this; }

        public ProviderRequest build() {
            return ProviderRequest.builder()
                    .requestId(requestId)
                    .model(model)
                    .messages(messages)
                    .parameters(parameters)
                    .tools(tools)
                    .toolChoice(toolChoice)
                    .streaming(streaming)
                    .timeout(timeout)
                    .userId(userId)
                    .sessionId(sessionId)
                    .traceId(traceId)
                    .apiKey(apiKey)
                    .metadata(metadata)
                    .build();
        }
    }
}
