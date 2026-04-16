/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.cloud.openai;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.exception.ProviderException;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * OpenAI Cloud Provider.
 *
 * <p>Provides access to OpenAI models (GPT-4, GPT-3.5-turbo, etc.) via the OpenAI API.</p>
 *
 * <h2>Configuration</h2>
 * <pre>{@code
 * {
 *   "apiKey": "sk-...",  // Required: OpenAI API key
 *   "baseUrl": "https://api.openai.com/v1",  // Optional: Custom endpoint
 *   "organization": "org-..."  // Optional: Organization ID
 * }
 * }</pre>
 *
 * <h2>Supported Models</h2>
 * <ul>
 *   <li>gpt-4</li>
 *   <li>gpt-4-turbo</li>
 *   <li>gpt-4-32k</li>
 *   <li>gpt-3.5-turbo</li>
 *   <li>gpt-3.5-turbo-16k</li>
 *   <li>o1-preview</li>
 *   <li>o1-mini</li>
 * </ul>
 *
 * @since 2.1.0
 */
public class OpenAiCloudProvider implements LLMProvider, StreamingProvider {

    private static final Logger LOG = Logger.getLogger(OpenAiCloudProvider.class);

    /**
     * Provider ID.
     */
    public static final String ID = "openai-cloud-provider";

    /**
     * Provider version.
     */
    public static final String VERSION = "1.0.0";

    /**
     * Supported models.
     */
    private static final List<String> SUPPORTED_MODELS = List.of(
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-32k",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1-preview",
            "o1-mini"
    );

    /**
     * Provider configuration.
     */
    private final Map<String, Object> config = new ConcurrentHashMap<>();

    /**
     * Provider state.
     */
    private volatile OpenAiConfig openAiConfig;
    private volatile OpenAIClient client;
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "OpenAI";
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(ID)
                .name("OpenAI")
                .description("OpenAI - GPT-4, GPT-3.5-Turbo and compatible APIs")
                .version(VERSION)
                .vendor("OpenAI Inc.")
                .homepage("https://openai.com")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .multimodal(true)
                .embeddings(true)
                .structuredOutputs(true)
                .maxContextTokens(128000)
                .maxOutputTokens(16384)
                .supportedModels(new HashSet<>(SUPPORTED_MODELS))
                .supportedLanguages(List.of("en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        // Load configuration
        this.config.putAll(config.getProperties());
        config.getSecret("apiKey").ifPresent(key -> this.config.put("apiKey", key));

        // Parse configuration into typed config object
        this.openAiConfig = new OpenAiConfig(
                (String) this.config.get("apiKey"),
                (String) this.config.getOrDefault("baseUrl", "https://api.openai.com/v1"),
                (String) this.config.get("organization"),
                (Boolean) this.config.getOrDefault("enabled", true),
                (Integer) this.config.getOrDefault("timeoutSeconds", 30),
                (Integer) this.config.getOrDefault("maxRetries", 3)
        );

        // Validate required configuration
        if (!openAiConfig.isValid()) {
            throw new ProviderException.ProviderInitializationException(
                    ID, "OpenAI API key not configured");
        }

        // Initialize HTTP client
        this.client = new OpenAIClient(openAiConfig);
        this.initialized = true;
        
        LOG.infof("OpenAI Cloud Provider initialized (version %s)", VERSION);
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (modelId == null || modelId.isBlank()) {
            return false;
        }
        
        String lower = modelId.toLowerCase();
        return SUPPORTED_MODELS.stream()
                .anyMatch(m -> m.equalsIgnoreCase(lower) || lower.startsWith(m.toLowerCase()));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        if (!initialized) {
            return Uni.createFrom().failure(
                new IllegalStateException("Provider not initialized"));
        }

        if (openAiConfig == null || !openAiConfig.isValid()) {
            return Uni.createFrom().failure(
                new ProviderException.ProviderAuthenticationException(ID, 
                    "OpenAI API key not configured"));
        }

        try {
            // Convert ProviderRequest to OpenAIRequest
            OpenAIRequest openAiRequest = convertToOpenAiRequest(request, false);
            
            LOG.debugf("Processing completion request for model: %s", request.getModel());

            return client.chatCompletions(openAiRequest)
                    .map(response -> convertToInferenceResponse(response, request));
                    
        } catch (Exception e) {
            LOG.errorf("Failed to process OpenAI request: %s", e.getMessage());
            return Uni.createFrom().failure(e);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        if (!initialized) {
            return Multi.createFrom().failure(
                new IllegalStateException("Provider not initialized"));
        }

        if (openAiConfig == null || !openAiConfig.isValid()) {
            return Multi.createFrom().failure(
                new ProviderException.ProviderAuthenticationException(ID, 
                    "OpenAI API key not configured"));
        }

        try {
            // Convert ProviderRequest to OpenAIRequest
            OpenAIRequest openAiRequest = convertToOpenAiRequest(request, true);
            
            LOG.debugf("Processing streaming request for model: %s", request.getModel());

            return client.chatCompletionsStream(openAiRequest)
                    .onItem().transformToMultiAndConcatenate(response -> {
                        if (response.getChoices() == null || response.getChoices().isEmpty()) {
                            return Multi.createFrom().empty();
                        }
                        
                        OpenAIStreamChoice choice = response.getChoices().get(0);
                        String content = extractContent(choice);
                        
                        if (content == null || content.isBlank()) {
                            return Multi.createFrom().empty();
                        }
                        
                        StreamingInferenceChunk chunk = StreamingInferenceChunk.of(
                                request.getRequestId(),
                                choice.getIndex(),
                                content
                        );
                        
                        // Check if this is the final chunk
                        if (choice.getFinishReason() != null) {
                            return Multi.createFrom().items(
                                    chunk,
                                    StreamingInferenceChunk.finalChunk(request.getRequestId(), choice.getIndex() + 1, "")
                            );
                        }
                        
                        return Multi.createFrom().item(chunk);
                    });
                    
        } catch (Exception e) {
            LOG.errorf("Failed to process OpenAI streaming request: %s", e.getMessage());
            return Multi.createFrom().failure(e);
        }
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            if (!initialized) {
                return ProviderHealth.unknown();
            }
            
            if (openAiConfig == null || !openAiConfig.isValid()) {
                return ProviderHealth.degraded("API key not configured");
            }
            
            return ProviderHealth.healthy("OpenAI API available");
        });
    }

    @Override
    public void shutdown() {
        initialized = false;
        config.clear();
        openAiConfig = null;
        client = null;
        LOG.info("OpenAI provider shutdown complete");
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Convert ProviderRequest to OpenAIRequest.
     */
    private OpenAIRequest convertToOpenAiRequest(ProviderRequest request, boolean streaming) {
        OpenAIRequest openAiRequest = new OpenAIRequest();
        
        // Model
        openAiRequest.setModel(request.getModel());
        
        // Messages
        if (request.getMessages() != null) {
            List<OpenAIMessage> messages = request.getMessages().stream()
                    .map(this::convertToOpenAiMessage)
                    .collect(Collectors.toList());
            openAiRequest.setMessages(messages);
        }
        
        // Parameters
        Map<String, Object> params = request.getParameters();
        if (params != null) {
            if (params.containsKey("temperature")) {
                openAiRequest.setTemperature(((Number) params.get("temperature")).doubleValue());
            }
            if (params.containsKey("max_tokens")) {
                openAiRequest.setMaxTokens(((Number) params.get("max_tokens")).intValue());
            }
            if (params.containsKey("top_p")) {
                openAiRequest.setTopP(((Number) params.get("top_p")).doubleValue());
            }
            if (params.containsKey("frequency_penalty")) {
                openAiRequest.setFrequencyPenalty(((Number) params.get("frequency_penalty")).doubleValue());
            }
            if (params.containsKey("presence_penalty")) {
                openAiRequest.setPresencePenalty(((Number) params.get("presence_penalty")).doubleValue());
            }
            if (params.containsKey("stop")) {
                Object stop = params.get("stop");
                if (stop instanceof List) {
                    openAiRequest.setStop((List<String>) stop);
                } else if (stop instanceof String) {
                    openAiRequest.setStop(List.of((String) stop));
                }
            }
        }
        
        // Streaming
        openAiRequest.setStream(streaming);
        
        // Tools
        if (request.getTools() != null && !request.getTools().isEmpty()) {
            List<OpenAITool> tools = request.getTools().stream()
                    .map(this::convertToOpenAiTool)
                    .collect(Collectors.toList());
            openAiRequest.setTools(tools);
            
            if (request.getToolChoice() != null) {
                openAiRequest.setToolChoice(request.getToolChoice());
            }
        }
        
        return openAiRequest;
    }

    /**
     * Convert SPI Message to OpenAIMessage.
     */
    private OpenAIMessage convertToOpenAiMessage(Message message) {
        OpenAIMessage openAiMessage = new OpenAIMessage();
        openAiMessage.setRole(message.getRole().toString().toLowerCase());
        openAiMessage.setContent(message.getContent());
        
        if (message.getName() != null) {
            openAiMessage.setName(message.getName());
        }
        
        if (message.getToolCallId() != null) {
            openAiMessage.setToolCallId(message.getToolCallId());
        }
        
        if (message.getToolCalls() != null && !message.getToolCalls().isEmpty()) {
            List<OpenAIToolCall> toolCalls = message.getToolCalls().stream()
                    .map(tc -> {
                        OpenAIToolCall toolCall = new OpenAIToolCall();
                        toolCall.setId(tc.getId() != null ? tc.getId() : UUID.randomUUID().toString());
                        toolCall.setType("function");
                        
                        OpenAIFunctionCall functionCall = new OpenAIFunctionCall();
                        functionCall.setName(tc.getFunction() != null ? tc.getFunction().getName() : "");
                        try {
                            functionCall.setArguments(new ObjectMapper().writeValueAsString(tc.getArguments()));
                        } catch (Exception e) {
                            functionCall.setArguments("{}");
                        }
                        toolCall.setFunction(functionCall);
                        
                        return toolCall;
                    })
                    .collect(Collectors.toList());
            openAiMessage.setToolCalls(toolCalls);
        }
        
        return openAiMessage;
    }

    /**
     * Convert SPI ToolDefinition to OpenAITool.
     */
    private OpenAITool convertToOpenAiTool(tech.kayys.gollek.spi.tool.ToolDefinition tool) {
        OpenAIFunction function = new OpenAIFunction();
        function.setName(tool.getName());
        function.setDescription(tool.getDescription().orElse(null));
        function.setParameters(tool.getParameters());
        
        return new OpenAITool(function);
    }

    /**
     * Convert OpenAIResponse to InferenceResponse.
     */
    private InferenceResponse convertToInferenceResponse(OpenAIResponse response, ProviderRequest request) {
        InferenceResponse.Builder builder = InferenceResponse.builder()
                .requestId(request.getRequestId())
                .model(response.getModel() != null ? response.getModel() : request.getModel());
        
        // Extract content from choices
        if (response.getChoices() != null && !response.getChoices().isEmpty()) {
            OpenAIResponse.OpenAIChoice choice = response.getChoices().get(0);
            
            if (choice.getMessage() != null) {
                String content = choice.getMessage().getContent();
                builder.content(content != null ? content : "");
                
                // Check for tool calls
                if (choice.getMessage().getToolCalls() != null && 
                    !choice.getMessage().getToolCalls().isEmpty()) {
                    List<InferenceResponse.ToolCall> toolCalls = choice.getMessage().getToolCalls().stream()
                            .map(tc -> {
                                Map<String, Object> arguments = parseArguments(
                                        tc.getFunction().getArguments());
                                return new InferenceResponse.ToolCall(
                                        tc.getFunction().getName(),
                                        arguments
                                );
                            })
                            .collect(Collectors.toList());
                    builder.toolCalls(toolCalls);
                    builder.finishReason(InferenceResponse.FinishReason.TOOL_CALLS);
                }
            }
            
            // Set finish reason
            if (choice.getFinishReason() != null) {
                builder.finishReason(mapFinishReason(choice.getFinishReason()));
            }
        }
        
        // Set usage
        if (response.getUsage() != null) {
            OpenAIUsage usage = response.getUsage();
            builder.inputTokens(usage.getPromptTokens() != null ? usage.getPromptTokens() : 0)
                   .outputTokens(usage.getCompletionTokens() != null ? usage.getCompletionTokens() : 0)
                   .tokensUsed(usage.getTotalTokens() != null ? usage.getTotalTokens() : 0);
        }
        
        // Set metadata
        builder.metadata("provider", ID)
               .metadata("model", response.getModel())
               .timestamp(Instant.now());
        
        return builder.build();
    }

    /**
     * Extract content from streaming choice.
     */
    private String extractContent(OpenAIStreamChoice choice) {
        OpenAIMessage delta = choice.getDelta();
        if (delta != null && delta.getContent() != null) {
            return delta.getContent();
        }
        
        // Fallback to message for non-streaming responses
        OpenAIMessage message = choice.getMessage();
        if (message != null && message.getContent() != null) {
            return message.getContent();
        }
        
        return "";
    }

    /**
     * Parse function arguments from JSON string.
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> parseArguments(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) {
            return new HashMap<>();
        }
        
        try {
            return new ObjectMapper().readValue(argumentsJson, Map.class);
        } catch (Exception e) {
            LOG.warnf("Failed to parse function arguments: %s", argumentsJson);
            return new HashMap<>();
        }
    }

    /**
     * Map OpenAI finish reason to SPI finish reason.
     */
    private InferenceResponse.FinishReason mapFinishReason(String openAiFinishReason) {
        if (openAiFinishReason == null) {
            return InferenceResponse.FinishReason.STOP;
        }
        
        return switch (openAiFinishReason.toLowerCase()) {
            case "stop" -> InferenceResponse.FinishReason.STOP;
            case "length" -> InferenceResponse.FinishReason.LENGTH;
            case "tool_calls", "function_call" -> InferenceResponse.FinishReason.TOOL_CALLS;
            default -> InferenceResponse.FinishReason.ERROR;
        };
    }

    // ───────────────────────────────────────────────────────────────────────
    // Embedding Methods
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Generate embedding vector for a single text.
     *
     * @param text Input text
     * @return Embedding vector
     */
    public float[] embed(String text) {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }

        OpenAIEmbeddingRequest request = new OpenAIEmbeddingRequest(
                "text-embedding-3-small",
                text
        );
        request.setEncodingFormat("float");

        return client.embeddings(request)
                .map(response -> {
                    if (response.getData() == null || response.getData().isEmpty()) {
                        throw new InferenceException("No embeddings returned from OpenAI");
                    }
                    List<Double> embedding = response.getData().get(0).getEmbedding();
                    float[] result = new float[embedding.size()];
                    for (int i = 0; i < embedding.size(); i++) {
                        result[i] = embedding.get(i).floatValue();
                    }
                    return result;
                })
                .await().atMost(java.time.Duration.ofSeconds(30));
    }

    /**
     * Generate embeddings for multiple texts in batch.
     *
     * @param texts Input texts
     * @return Array of embedding vectors
     */
    public float[][] embedBatch(String[] texts) {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }

        OpenAIEmbeddingRequest request = new OpenAIEmbeddingRequest(
                "text-embedding-3-small",
                List.of(texts)
        );
        request.setEncodingFormat("float");

        return client.embeddings(request)
                .map(response -> {
                    if (response.getData() == null || response.getData().isEmpty()) {
                        throw new InferenceException("No embeddings returned from OpenAI");
                    }
                    float[][] embeddings = new float[response.getData().size()][];
                    for (int i = 0; i < response.getData().size(); i++) {
                        List<Double> embedding = response.getData().get(i).getEmbedding();
                        embeddings[i] = new float[embedding.size()];
                        for (int j = 0; j < embedding.size(); j++) {
                            embeddings[i][j] = embedding.get(j).floatValue();
                        }
                    }
                    return embeddings;
                })
                .await().atMost(java.time.Duration.ofSeconds(60));
    }

    /**
     * Get the dimensionality of the embedding vectors.
     *
     * @return Embedding dimension
     */
    public int dimension() {
        // text-embedding-3-small default dimension
        return 1536;
    }

    /**
     * Generate embeddings with custom model and parameters.
     *
     * @param inputs Input texts
     * @param model Embedding model (e.g., text-embedding-3-small, text-embedding-3-large)
     * @param dimensions Output dimensions (optional, for text-embedding-3 models)
     * @return EmbeddingResponse with generated embeddings
     */
    public EmbeddingResponse generateEmbeddings(List<String> inputs, String model, Integer dimensions) {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }

        OpenAIEmbeddingRequest request = new OpenAIEmbeddingRequest(model, inputs);
        request.setEncodingFormat("float");
        if (dimensions != null) {
            request.setDimensions(dimensions);
        }

        return client.embeddings(request)
                .map(response -> {
                    if (response.getData() == null || response.getData().isEmpty()) {
                        throw new InferenceException("No embeddings returned from OpenAI");
                    }

                    List<float[]> embeddings = new ArrayList<>();
                    for (OpenAIEmbeddingData data : response.getData()) {
                        List<Double> embedding = data.getEmbedding();
                        float[] floatEmbedding = new float[embedding.size()];
                        for (int i = 0; i < embedding.size(); i++) {
                            floatEmbedding[i] = embedding.get(i).floatValue();
                        }
                        embeddings.add(floatEmbedding);
                    }

                    int dim = embeddings.isEmpty() ? 0 : embeddings.get(0).length;

                    return new EmbeddingResponse(
                            UUID.randomUUID().toString(),
                            response.getModel() != null ? response.getModel() : model,
                            embeddings,
                            dim,
                            Map.of(
                                    "usage", response.getUsage() != null ? response.getUsage().getTotalTokens() : 0
                            )
                    );
                })
                .await().atMost(java.time.Duration.ofSeconds(60));
    }
}
