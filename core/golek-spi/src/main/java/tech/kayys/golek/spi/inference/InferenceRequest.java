package tech.kayys.golek.spi.inference;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.tool.ToolDefinition;

import org.jetbrains.annotations.Nullable;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Immutable inference request.
 * Thread-safe and serializable.
 */
public final class InferenceRequest {

    @NotBlank
    private final String requestId;

    @Nullable
    @Deprecated
    private final String tenantId;

    @NotBlank
    private final String model;

    @NotNull
    private final List<Message> messages;

    @Nullable
    private final List<ToolDefinition> tools;

    @Nullable
    private final Object toolChoice;

    private final Map<String, Object> parameters;
    private final boolean streaming;

    @Nullable
    private final String preferredProvider;

    @Nullable
    private final Duration timeout;

    private final int priority;

    private final boolean cacheBypass;

    @JsonCreator
    public InferenceRequest(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("tenantId") String tenantId,
            @JsonProperty("model") String model,
            @JsonProperty("messages") List<Message> messages,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("tools") List<ToolDefinition> tools,
            @JsonProperty("toolChoice") Object toolChoice,
            @JsonProperty("streaming") boolean streaming,
            @JsonProperty("preferredProvider") String preferredProvider,
            @JsonProperty("timeout") Duration timeout,
            @JsonProperty("priority") int priority,
            @JsonProperty("cacheBypass") boolean cacheBypass) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.tenantId = tenantId;
        this.model = Objects.requireNonNull(model, "model");
        this.messages = Collections.unmodifiableList(new ArrayList<>(
                Objects.requireNonNull(messages, "messages")));
        this.tools = tools != null ? Collections.unmodifiableList(new ArrayList<>(tools)) : null;
        this.toolChoice = toolChoice;
        this.parameters = parameters != null
                ? Collections.unmodifiableMap(new HashMap<>(parameters))
                : Collections.emptyMap();
        this.streaming = streaming;
        this.preferredProvider = preferredProvider;
        this.timeout = timeout;
        this.priority = priority;
        this.cacheBypass = cacheBypass;
    }

    // Getters
    public String getRequestId() {
        return requestId;
    }

    /**
     * @deprecated Tenant ID is resolved server-side from the API key.
     * Client code should not set or rely on this value.
     */
    @Deprecated
    public String getTenantId() {
        return tenantId;
    }

    public String getModel() {
        return model;
    }

    public List<Message> getMessages() {
        return messages;
    }

    public List<ToolDefinition> getTools() {
        return tools;
    }

    public Object getToolChoice() {
        return toolChoice;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public boolean isStreaming() {
        return streaming;
    }

    public Optional<String> getPreferredProvider() {
        return Optional.ofNullable(preferredProvider);
    }

    public Optional<Duration> getTimeout() {
        return Optional.ofNullable(timeout);
    }

    public int getPriority() {
        return priority;
    }

    public boolean isCacheBypass() {
        return cacheBypass;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId = UUID.randomUUID().toString();
        private String tenantId;
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final List<ToolDefinition> tools = new ArrayList<>();
        private Object toolChoice;
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean streaming = false;
        private String preferredProvider;
        private Duration timeout;
        private int priority = 5;
        private boolean cacheBypass = false;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        /**
         * @deprecated Tenant ID is resolved server-side from the API key.
         * Client code should not set or rely on this value.
         */
        @Deprecated
        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder message(Message message) {
            this.messages.add(message);
            return this;
        }

        public Builder messages(List<Message> messages) {
            this.messages.addAll(messages);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder temperature(double temperature) {
            this.parameters.put("temperature", temperature);
            return this;
        }

        public Builder tool(ToolDefinition tool) {
            this.tools.add(tool);
            return this;
        }

        public Builder tools(List<ToolDefinition> tools) {
            this.tools.addAll(tools);
            return this;
        }

        public Builder toolChoice(Object toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.parameters.put("max_tokens", maxTokens);
            return this;
        }

        public Builder topP(double topP) {
            this.parameters.put("top_p", topP);
            return this;
        }

        public Builder topK(int topK) {
            this.parameters.put("top_k", topK);
            return this;
        }

        public Builder minP(double minP) {
            this.parameters.put("min_p", minP);
            return this;
        }

        public Builder seed(int seed) {
            this.parameters.put("seed", seed);
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder preferredProvider(String provider) {
            this.preferredProvider = provider;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public Builder cacheBypass(boolean cacheBypass) {
            this.cacheBypass = cacheBypass;
            return this;
        }

        public InferenceRequest build() {
            Objects.requireNonNull(model, "model is required");
            if (messages.isEmpty()) {
                throw new IllegalStateException("At least one message is required");
            }
            return new InferenceRequest(
                    requestId, tenantId, model, messages, parameters, tools, toolChoice, streaming,
                    preferredProvider, timeout, priority, cacheBypass);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof InferenceRequest that))
            return false;
        return streaming == that.streaming &&
                priority == that.priority &&
                requestId.equals(that.requestId) &&
                model.equals(that.model) &&
                messages.equals(that.messages) &&
                Objects.equals(tools, that.tools) &&
                Objects.equals(toolChoice, that.toolChoice);
    }

    @Override
    public int hashCode() {
        return Objects.hash(requestId, model, messages, tools, toolChoice, streaming, priority);
    }

    @Override
    public String toString() {
        return "InferenceRequest{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", messageCount=" + messages.size() +
                ", streaming=" + streaming +
                ", priority=" + priority +
                '}';
    }
}
