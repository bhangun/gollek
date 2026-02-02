package tech.kayys.golek.api.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.jetbrains.annotations.Nullable;
import tech.kayys.golek.api.Message;
import tech.kayys.golek.api.tool.ToolDefinition;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;
import java.util.*;

/**
 * Normalized provider request.
 * Providers receive this standardized format regardless of original input.
 */
public final class ProviderRequest {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String model;

    @NotNull
    private final List<Message> messages;

    @Nullable
    private final Object toolChoice;

    private final Map<String, Object> parameters;
    private final boolean streaming;
    private final Duration timeout;
    private final TenantContext tenantContext;

    private final List<ToolDefinition> tools;

    private final Map<String, Object> metadata;

    @JsonCreator
    public ProviderRequest(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("model") String model,
            @JsonProperty("messages") List<Message> messages,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("tools") List<ToolDefinition> tools,
            @JsonProperty("toolChoice") Object toolChoice,
            @JsonProperty("streaming") boolean streaming,
            @JsonProperty("timeout") Duration timeout,
            @JsonProperty("tenantContext") TenantContext tenantContext,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.model = Objects.requireNonNull(model, "model");
        this.messages = Collections.unmodifiableList(new ArrayList<>(
                Objects.requireNonNull(messages, "messages")));

        this.tools = tools != null
                ? Collections.unmodifiableList(new ArrayList<>(tools))
                : Collections.emptyList();
        this.toolChoice = toolChoice;
        this.parameters = parameters != null
                ? Collections.unmodifiableMap(new HashMap<>(parameters))
                : Collections.emptyMap();
        this.streaming = streaming;
        this.timeout = timeout != null ? timeout : Duration.ofSeconds(30);
        this.tenantContext = tenantContext;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Getters
    public String getRequestId() {
        return requestId;
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

    public Duration getTimeout() {
        return timeout;
    }

    public TenantContext getTenantContext() {
        return tenantContext;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    // Parameter helpers
    public <T> Optional<T> getParameter(String key, Class<T> type) {
        Object value = parameters.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of(type.cast(value));
        }
        return Optional.empty();
    }

    public double getTemperature() {
        return getParameter("temperature", Number.class)
                .map(Number::doubleValue)
                .orElse(0.7);
    }

    public int getMaxTokens() {
        return getParameter("max_tokens", Number.class)
                .map(Number::intValue)
                .orElse(2048);
    }

    public double getTopP() {
        return getParameter("top_p", Number.class)
                .map(Number::doubleValue)
                .orElse(1.0);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId = UUID.randomUUID().toString();
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final List<ToolDefinition> tools = new ArrayList<>();
        private Object toolChoice;
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean streaming = false;
        private Duration timeout = Duration.ofSeconds(30);
        private TenantContext tenantContext;
        private Map<String, Object> metadata = new HashMap<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
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

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            if (metadata != null) {
                this.metadata.putAll(metadata);
            }
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ProviderRequest build() {
            Objects.requireNonNull(model, "model is required");
            if (messages.isEmpty()) {
                throw new IllegalStateException("At least one message is required");
            }
            return new ProviderRequest(
                    requestId, model, messages, parameters, tools, toolChoice, streaming, timeout, tenantContext,
                    metadata);
        }
    }

    @Override
    public String toString() {
        return "ProviderRequest{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", messageCount=" + messages.size() +
                ", toolCount=" + (tools != null ? tools.size() : 0) +
                ", streaming=" + streaming +
                '}';
    }
}