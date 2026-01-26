package tech.kayys.golek.provider.core.mcp;

import java.util.*;

/**
 * Context for MCP inference processing.
 * Holds request data and intermediate results.
 */
public final class MCPInferenceContext {

    private final String requestId;
    private final String model;
    private final List<Message> messages;
    private final Map<String, Object> parameters;
    private final Map<String, Object> state;

    private MCPInferenceContext(Builder builder) {
        this.requestId = builder.requestId;
        this.model = builder.model;
        this.messages = Collections.unmodifiableList(new ArrayList<>(builder.messages));
        this.parameters = Collections.unmodifiableMap(new HashMap<>(builder.parameters));
        this.state = new HashMap<>(builder.state);
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

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public Map<String, Object> getState() {
        return state;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final Map<String, Object> parameters = new HashMap<>();
        private final Map<String, Object> state = new HashMap<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder messages(List<Message> messages) {
            this.messages.addAll(messages);
            return this;
        }

        public Builder message(Message message) {
            this.messages.add(message);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder state(String key, Object value) {
            this.state.put(key, value);
            return this;
        }

        public MCPInferenceContext build() {
            Objects.requireNonNull(requestId, "requestId is required");
            return new MCPInferenceContext(this);
        }
    }

    @Override
    public String toString() {
        return "MCPInferenceContext{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", messageCount=" + messages.size() +
                '}';
    }
}