package tech.kayys.gollek.spi.inference;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.tool.ToolDefinition;

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

    private final Priority priority;

    private final boolean cacheBypass;

    @NotNull
    private final RequestContext requestContext;

    @Nullable
    private final InferenceStage inferenceStage;

    private final int promptTokenCount;

    private final Map<String, Object> metadata;

    @NotNull
    private final List<Attachment> attachments;

    @JsonCreator
    public InferenceRequest(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("apiKey") @JsonAlias("apiKey") String apiKey,
            @JsonProperty("model") String model,
            @JsonProperty("messages") List<Message> messages,
            @JsonProperty("parameters") Map<String, Object> parameters,
            @JsonProperty("tools") List<ToolDefinition> tools,
            @JsonProperty("toolChoice") Object toolChoice,
            @JsonProperty("streaming") boolean streaming,
            @JsonProperty("preferredProvider") String preferredProvider,
            @JsonProperty("timeout") Duration timeout,
            @JsonProperty("priority") Priority priority,
            @JsonProperty("cacheBypass") boolean cacheBypass,
            @JsonProperty("userId") String userId,
            @JsonProperty("sessionId") String sessionId,
            @JsonProperty("traceId") String traceId,
            @JsonProperty("inferenceStage") InferenceStage inferenceStage,
            @JsonProperty("promptTokenCount") int promptTokenCount,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("attachments") List<Attachment> attachments) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.requestContext = new RequestContext() {
            @Override
            public String apiKey() {
                return apiKey;
            }

            @Override
            public String getRequestId() {
                return requestId;
            }

            @Override
            public String userId() {
                return userId;
            }

            @Override
            public String sessionId() {
                return sessionId;
            }

            @Override
            public String traceId() {
                return traceId;
            }
        };
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
        this.inferenceStage = inferenceStage;
        this.promptTokenCount = promptTokenCount;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.attachments = attachments != null
                ? Collections.unmodifiableList(new ArrayList<>(attachments))
                : Collections.emptyList();
    }

    // Getters
    public String getRequestId() {
        return requestId;
    }

    public String getModel() {
        return model;
    }

    public String getApiKey() {
        return requestContext.getApiKey();
    }

    public RequestContext getRequestContext() {
        return requestContext;
    }

    public List<Message> getMessages() {
        return messages;
    }

    /**
     * Returns a stable hash of the current message history to identify
     * possible KV cache locations on execution providers.
     */
    public String prefixHash() {
        if (messages == null || messages.isEmpty()) {
            return "empty";
        }
        // Message.hashCode() is stable and includes role, content, and tool details.
        return Integer.toHexString(messages.hashCode());
    }

    /**
     * Convenience method to get the prompt from parameters or last user message.
     */
    public String getPrompt() {
        String prompt = (String) parameters.get("prompt");
        if (prompt != null && !prompt.isBlank()) {
            return prompt;
        }

        if (!messages.isEmpty()) {
            Message last = messages.get(messages.size() - 1);
            if (last.getRole() == Message.Role.USER) {
                return last.getContent();
            }
        }
        return null;
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

    public Priority getPriority() {
        return priority != null ? priority : Priority.NORMAL;
    }

    public boolean isCacheBypass() {
        return cacheBypass;
    }

    public Optional<String> getUserId() {
        return Optional.ofNullable(requestContext.userId());
    }

    public Optional<String> getSessionId() {
        return Optional.ofNullable(requestContext.sessionId());
    }

    public Optional<String> getTraceId() {
        return Optional.ofNullable(requestContext.traceId());
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public InferenceStage getInferenceStage() {
        return inferenceStage != null ? inferenceStage : InferenceStage.COMBINED;
    }

    public boolean hasExplicitStage() {
        return inferenceStage != null;
    }

    public int getPromptTokenCount() {
        return promptTokenCount;
    }

    public List<Attachment> getAttachments() {
        return attachments;
    }

    // Parameter accessors
    public double getTemperature() {
        Object val = parameters.get("temperature");
        return val instanceof Number n ? n.doubleValue() : 0.2;
    }

    public double getTopP() {
        Object val = parameters.get("top_p");
        return val instanceof Number n ? n.doubleValue() : 0.9;
    }

    public int getTopK() {
        Object val = parameters.get("top_k");
        return val instanceof Number n ? n.intValue() : 40;
    }

    public int getMaxTokens() {
        Object val = parameters.get("max_tokens");
        return val instanceof Number n ? n.intValue() : 256;
    }

    public double getRepeatPenalty() {
        Object val = parameters.get("repeat_penalty");
        return val instanceof Number n ? n.doubleValue() : 1.1;
    }

    public boolean isJsonMode() {
        Object val = parameters.get("json_mode");
        return val instanceof Boolean b ? b : false;
    }

    public int getMirostat() {
        Object val = parameters.get("mirostat");
        return val instanceof Number n ? n.intValue() : 0;
    }

    @Nullable
    public String getGrammar() {
        return (String) parameters.get("grammar");
    }

    // Builder
    public Builder toBuilder() {
        Builder builder = new Builder()
                .requestId(requestId)
                .apiKey(requestContext.apiKey())
                .model(model)
                .messages(messages)
                .parameters(parameters)
                .preferredProvider(preferredProvider)
                .toolChoice(toolChoice)
                .streaming(streaming)
                .priority(priority)
                .cacheBypass(cacheBypass)
                .userId(requestContext.userId())
                .sessionId(requestContext.sessionId())
                .traceId(requestContext.traceId())
                .inferenceStage(inferenceStage)
                .promptTokenCount(promptTokenCount)
                .metadata(metadata)
                .attachments(attachments);

        if (tools != null) {
            builder.tools(tools);
        }

        if (timeout != null) {
            builder.timeout(timeout);
        }

        return builder;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId = UUID.randomUUID().toString();
        private String apiKey = ApiKeyConstants.COMMUNITY_API_KEY;
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final List<ToolDefinition> tools = new ArrayList<>();
        private Object toolChoice;
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean streaming = false;
        private String preferredProvider;
        private Duration timeout;
        private Priority priority = Priority.NORMAL;
        private boolean cacheBypass = false;
        private String userId;
        private String sessionId;
        private String traceId;
        private InferenceStage inferenceStage;
        private int promptTokenCount = -1;
        private final Map<String, Object> metadata = new HashMap<>();
        private final List<Attachment> attachments = new ArrayList<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder prompt(String prompt) {
            this.parameters.put("prompt", prompt);
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

        public Builder repeatPenalty(double repeatPenalty) {
            this.parameters.put("repeat_penalty", repeatPenalty);
            return this;
        }

        public Builder jsonMode(boolean jsonMode) {
            this.parameters.put("json_mode", jsonMode);
            return this;
        }

        public Builder mirostat(int mirostat) {
            this.parameters.put("mirostat", mirostat);
            return this;
        }

        public Builder grammar(String grammar) {
            this.parameters.put("grammar", grammar);
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

        public Builder priority(Priority priority) {
            this.priority = priority;
            return this;
        }

        /**
         * Numeric priority mapping for backward compatibility.
         * 0 -> CRITICAL, 1 -> HIGH, 2-5 -> NORMAL, >5 -> LOW.
         */
        public Builder priority(int priority) {
            if (priority <= 0)
                this.priority = Priority.CRITICAL;
            else if (priority == 1)
                this.priority = Priority.HIGH;
            else if (priority <= 5)
                this.priority = Priority.NORMAL;
            else
                this.priority = Priority.LOW;
            return this;
        }

        public Builder cacheBypass(boolean cacheBypass) {
            this.cacheBypass = cacheBypass;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder traceId(String traceId) {
            this.traceId = traceId;
            return this;
        }

        public Builder inferenceStage(InferenceStage inferenceStage) {
            this.inferenceStage = inferenceStage;
            return this;
        }

        public Builder promptTokenCount(int promptTokenCount) {
            this.promptTokenCount = promptTokenCount;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public Builder attachment(Attachment attachment) {
            this.attachments.add(attachment);
            return this;
        }

        public Builder attachments(List<Attachment> attachments) {
            this.attachments.addAll(attachments);
            return this;
        }

        public InferenceRequest build() {
            Objects.requireNonNull(model, "model is required");
            if (messages.isEmpty()) {
                throw new IllegalStateException("At least one message is required");
            }
            return new InferenceRequest(
                    requestId, apiKey, model, messages, parameters, tools, toolChoice, streaming,
                    preferredProvider, timeout, priority, cacheBypass, userId, sessionId, traceId,
                    inferenceStage, promptTokenCount, metadata, attachments);
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
                Objects.equals(toolChoice, that.toolChoice) &&
                Objects.equals(requestContext.userId(), that.requestContext.userId()) &&
                Objects.equals(requestContext.sessionId(), that.requestContext.sessionId()) &&
                Objects.equals(requestContext.traceId(), that.requestContext.traceId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(requestId, model, messages, tools, toolChoice, streaming, priority, 
                requestContext.userId(), requestContext.sessionId(), requestContext.traceId());
    }

    @Override
    public String toString() {
        return "InferenceRequest{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", messageCount=" + messages.size() +
                ", streaming=" + streaming +
                ", priority=" + priority +
                ", userId='" + requestContext.userId() + '\'' +
                '}';
    }
}
