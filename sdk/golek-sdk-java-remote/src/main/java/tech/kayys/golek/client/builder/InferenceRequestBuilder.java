package tech.kayys.golek.client.builder;

import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.tool.ToolDefinition;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Utility class for building InferenceRequest objects.
 */
public class InferenceRequestBuilder {
    
    private String requestId = UUID.randomUUID().toString();
    private String model;
    private final List<Message> messages = new ArrayList<>();
    private final List<ToolDefinition> tools = new ArrayList<>();
    private Object toolChoice;
    private final Map<String, Object> parameters = new java.util.HashMap<>();
    private boolean streaming = false;
    private String preferredProvider;
    private Duration timeout;
    private int priority = 5;
    
    public static InferenceRequestBuilder builder() {
        return new InferenceRequestBuilder();
    }
    
    public InferenceRequestBuilder requestId(String requestId) {
        this.requestId = requestId;
        return this;
    }
    
    public InferenceRequestBuilder model(String model) {
        this.model = model;
        return this;
    }
    
    public InferenceRequestBuilder message(Message message) {
        this.messages.add(message);
        return this;
    }
    
    public InferenceRequestBuilder messages(List<Message> messages) {
        this.messages.addAll(messages);
        return this;
    }
    
    public InferenceRequestBuilder systemMessage(String content) {
        this.messages.add(Message.system(content));
        return this;
    }
    
    public InferenceRequestBuilder userMessage(String content) {
        this.messages.add(Message.user(content));
        return this;
    }
    
    public InferenceRequestBuilder assistantMessage(String content) {
        this.messages.add(Message.assistant(content));
        return this;
    }
    
    public InferenceRequestBuilder parameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }
    
    public InferenceRequestBuilder parameters(Map<String, Object> parameters) {
        this.parameters.putAll(parameters);
        return this;
    }
    
    public InferenceRequestBuilder temperature(double temperature) {
        this.parameters.put("temperature", temperature);
        return this;
    }
    
    public InferenceRequestBuilder maxTokens(int maxTokens) {
        this.parameters.put("max_tokens", maxTokens);
        return this;
    }
    
    public InferenceRequestBuilder topP(double topP) {
        this.parameters.put("top_p", topP);
        return this;
    }
    
    public InferenceRequestBuilder tool(ToolDefinition tool) {
        this.tools.add(tool);
        return this;
    }
    
    public InferenceRequestBuilder tools(List<ToolDefinition> tools) {
        this.tools.addAll(tools);
        return this;
    }
    
    public InferenceRequestBuilder toolChoice(Object toolChoice) {
        this.toolChoice = toolChoice;
        return this;
    }
    
    public InferenceRequestBuilder streaming(boolean streaming) {
        this.streaming = streaming;
        return this;
    }
    
    public InferenceRequestBuilder preferredProvider(String provider) {
        this.preferredProvider = provider;
        return this;
    }
    
    public InferenceRequestBuilder timeout(Duration timeout) {
        this.timeout = timeout;
        return this;
    }
    
    public InferenceRequestBuilder priority(int priority) {
        this.priority = priority;
        return this;
    }
    
    public InferenceRequest build() {
        return new InferenceRequest(
            requestId,
            model,
            messages,
            parameters,
            tools,
            toolChoice,
            streaming,
            preferredProvider,
            timeout,
            priority
        );
    }
}