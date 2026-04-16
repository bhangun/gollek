/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;
import org.jboss.resteasy.reactive.RestStreamElementType;

import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * OpenAI-compatible inference REST API.
 * Uses reflection to access Engine and HubClient to avoid circular
 * dependencies.
 */
@jakarta.ws.rs.Path("/v1")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Inference", description = "OpenAI-compatible inference endpoints")
public class OpenAiCompatibleResource {

    private static final org.jboss.logging.Logger log = org.jboss.logging.Logger
            .getLogger(OpenAiCompatibleResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> engineInstance;

    @Inject
    jakarta.enterprise.inject.Instance<Object> hubClientInstance;

    @Inject
    ObjectMapper objectMapper;

    /** Loaded model registry: model alias → model path. */
    private final Map<String, java.nio.file.Path> modelRegistry = new ConcurrentHashMap<>();

    public Map<String, java.nio.file.Path> getModelRegistry() {
        return modelRegistry;
    }

    private Object getEngine() {
        try {
            return engineInstance.get();
        } catch (Exception e) {
            log.error("DirectInferenceEngine not found", e);
            return null;
        }
    }

    private Object getHubClient() {
        try {
            return hubClientInstance.get();
        } catch (Exception e) {
            log.error("HuggingFaceHubClient not found", e);
            return null;
        }
    }

    @POST
    @jakarta.ws.rs.Path("/chat/completions")
    @Operation(summary = "Create a chat completion")
    public Object chatCompletions(ChatCompletionRequest req) {
        if (req == null)
            throw new WebApplicationException(
                    Response.status(400).entity(errorBody("invalid_request", "Request body is required")).build());

        String modelAlias = req.model != null ? req.model : "default";
        java.nio.file.Path modelPath = resolveModel(modelAlias);
        GenerationConfig gc = toGenerationConfig(req.temperature, req.maxTokens, req.topP, req.stop, req.stream);
        String prompt = formatMessages(req.messages);

        Object engine = getEngine();
        if (engine == null)
            return Response.status(503).entity(errorBody("service_unavailable", "Engine not available")).build();

        if (Boolean.TRUE.equals(req.stream)) {
            return streamChat(req, modelPath, prompt, gc, engine);
        } else {
            try {
                Uni<?> uni = (Uni<?>) engine.getClass()
                        .getMethod("generate", String.class, java.nio.file.Path.class, GenerationConfig.class)
                        .invoke(engine, prompt, modelPath, gc);

                return uni.map(resp -> {
                    try {
                        String content = (String) resp.getClass().getMethod("getContent").invoke(resp);
                        return buildChatResponse(req, content, estimateTokens(prompt), estimateTokens(content));
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }).onFailure().recoverWithItem(t -> {
                    log.errorf(t, "Chat completion error");
                    throw new WebApplicationException(
                            Response.status(500).entity(errorBody("internal_error", t.getMessage())).build());
                });
            } catch (Exception e) {
                return Response.serverError().entity(errorBody("internal_error", e.getMessage())).build();
            }
        }
    }

    @POST
    @jakarta.ws.rs.Path("/chat/completions/stream")
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestStreamElementType(MediaType.TEXT_PLAIN)
    @Operation(summary = "Stream a chat completion via SSE")
    public Multi<String> streamChatSse(ChatCompletionRequest req) {
        String modelAlias = req.model != null ? req.model : "default";
        java.nio.file.Path modelPath = resolveModel(modelAlias);
        GenerationConfig gc = toGenerationConfig(req.temperature, req.maxTokens, req.topP, req.stop, true);
        String prompt = formatMessages(req.messages);

        Object engine = getEngine();
        if (engine == null)
            return Multi.createFrom().item(buildStreamError("Engine not available"));

        return streamChat(req, modelPath, prompt, gc, engine);
    }

    private Multi<String> streamChat(ChatCompletionRequest req, java.nio.file.Path modelPath, String prompt,
            GenerationConfig gc, Object engine) {
        String completionId = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 28);
        String model = req.model != null ? req.model : "gollek";

        try {
            Multi<?> multi = (Multi<?>) engine.getClass()
                    .getMethod("generateStream", String.class, java.nio.file.Path.class, GenerationConfig.class)
                    .invoke(engine, prompt, modelPath, gc);

            return multi.map(chunk -> {
                try {
                    String delta = (String) chunk.getClass().getMethod("getDelta").invoke(chunk);
                    return buildInferenceChunk(completionId, model, delta);
                } catch (Exception e) {
                    return buildInferenceChunk(completionId, model, "");
                }
            }).onCompletion().continueWith(buildStreamDone())
                    .onFailure().recoverWithMulti(t -> {
                        log.errorf(t, "Stream error");
                        return Multi.createFrom().item(buildStreamError(t.getMessage()));
                    });
        } catch (Exception e) {
            return Multi.createFrom().item(buildStreamError(e.getMessage()));
        }
    }

    @POST
    @jakarta.ws.rs.Path("/completions")
    @Operation(summary = "Create a text completion")
    public Uni<CompletionResponse> textCompletions(CompletionRequest req) {
        String modelAlias = req.model != null ? req.model : "default";
        java.nio.file.Path modelPath = resolveModel(modelAlias);
        GenerationConfig gc = toGenerationConfig(req.temperature, req.maxTokens, req.topP, req.stop, false);
        String prompt = req.prompt != null ? req.prompt : "";

        Object engine = getEngine();
        if (engine == null)
            throw new WebApplicationException(
                    Response.status(503).entity(errorBody("service_unavailable", "Engine not available")).build());

        try {
            Uni<?> uni = (Uni<?>) engine.getClass()
                    .getMethod("generate", String.class, java.nio.file.Path.class, GenerationConfig.class)
                    .invoke(engine, prompt, modelPath, gc);

            return uni.map(resp -> {
                try {
                    String content = (String) resp.getClass().getMethod("getContent").invoke(resp);
                    return new CompletionResponse(
                            "cmpl-" + UUID.randomUUID().toString().substring(0, 8),
                            "text_completion",
                            req.model,
                            (int) (Instant.now().getEpochSecond()),
                            List.of(new CompletionChoice(content, 0, null, "stop")),
                            new Usage(estimateTokens(prompt), estimateTokens(content),
                                    estimateTokens(prompt) + estimateTokens(content)));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @GET
    @jakarta.ws.rs.Path("/models")
    @Operation(summary = "List available models")
    public ModelsResponse listModels() {
        List<ModelObject> models = modelRegistry.keySet().stream()
                .map(alias -> new ModelObject(alias, "model", (int) (Instant.now().getEpochSecond()), "gollek"))
                .toList();
        return new ModelsResponse("list", models);
    }

    @GET
    @jakarta.ws.rs.Path("/models/{model}")
    @Operation(summary = "Describe a specific model")
    public ModelObject getModel(@PathParam("model") String modelId) {
        if (!modelRegistry.containsKey(modelId)) {
            throw new WebApplicationException(Response.status(404)
                    .entity(errorBody("model_not_found", "Model '" + modelId + "' not found")).build());
        }
        return new ModelObject(modelId, "model", (int) Instant.now().getEpochSecond(), "gollek");
    }

    @POST
    @jakarta.ws.rs.Path("/models/load")
    @Operation(summary = "Load a model into memory")
    public Uni<ModelLoadResponse> loadModel(ModelLoadRequest req) {
        return Uni.createFrom().item(() -> {
            java.nio.file.Path modelPath;
            if (req.modelPath != null && !req.modelPath.isBlank()) {
                modelPath = java.nio.file.Path.of(req.modelPath);
            } else if (req.modelId != null && req.modelId.contains("/")) {
                Object hub = getHubClient();
                if (hub == null)
                    throw new RuntimeException("HuggingFaceHubClient not available");
                try {
                    Uni<java.nio.file.Path> uni = (Uni<java.nio.file.Path>) hub.getClass()
                            .getMethod("download", String.class).invoke(hub, req.modelId);
                    modelPath = uni.await().indefinitely();
                } catch (Exception e) {
                    throw new RuntimeException("Download failed: " + e.getMessage(), e);
                }
            } else {
                throw new WebApplicationException(Response.status(400)
                        .entity(errorBody("invalid_request", "Provide 'model_path' or 'model_id'")).build());
            }

            java.nio.file.Path adapterPath = req.adapterPath != null && !req.adapterPath.isBlank()
                    ? java.nio.file.Path.of(req.adapterPath)
                    : null;

            Object engine = getEngine();
            if (engine == null)
                throw new RuntimeException("Engine not available");

            try {
                String key = (String) engine.getClass()
                        .getMethod("loadModel", java.nio.file.Path.class, java.nio.file.Path.class,
                                java.nio.file.Path.class)
                        .invoke(engine, modelPath, adapterPath, null);
                String alias = req.alias != null ? req.alias : modelPath.getFileName().toString();
                modelRegistry.put(alias, modelPath);
                return new ModelLoadResponse(alias, key, "loaded");
            } catch (Exception e) {
                throw new RuntimeException("Load failed: " + e.getMessage(), e);
            }
        }).runSubscriptionOn(java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor());
    }

    @DELETE
    @jakarta.ws.rs.Path("/models/{model}")
    @Operation(summary = "Unload a model from memory")
    public Response unloadModel(@PathParam("model") String modelId) {
        java.nio.file.Path modelPath = modelRegistry.remove(modelId);
        if (modelPath == null) {
            return Response.status(404).entity(errorBody("model_not_found", "Model '" + modelId + "' not found"))
                    .build();
        }
        Object engine = getEngine();
        if (engine != null) {
            try {
                engine.getClass().getMethod("unloadModel", java.nio.file.Path.class).invoke(engine, modelPath);
            } catch (Exception e) {
                log.error("Unload failed", e);
            }
        }
        return Response.ok(Map.of("status", "unloaded", "model", modelId)).build();
    }

    private java.nio.file.Path resolveModel(String alias) {
        java.nio.file.Path path = modelRegistry.get(alias);
        if (path == null) {
            java.nio.file.Path direct = java.nio.file.Path.of(alias);
            if (java.nio.file.Files.exists(direct)) {
                Object engine = getEngine();
                if (engine != null) {
                    try {
                        engine.getClass().getMethod("loadModel", java.nio.file.Path.class, java.nio.file.Path.class,
                                java.nio.file.Path.class).invoke(engine, direct, null, null);
                        modelRegistry.put(alias, direct);
                        return direct;
                    } catch (Exception e) {
                        log.error("Auto-load failed", e);
                    }
                }
            }
            throw new WebApplicationException(Response.status(404)
                    .entity(errorBody("model_not_found", "Model '" + alias + "' not loaded.")).build());
        }
        return path;
    }

    private String formatMessages(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty())
            return "";
        var sb = new StringBuilder();
        for (var msg : messages) {
            sb.append(msg.role != null ? msg.role.toUpperCase() : "USER").append(": ")
                    .append(msg.content != null ? msg.content : "").append('\n');
        }
        return sb.toString();
    }

    private GenerationConfig toGenerationConfig(Double temperature, Integer maxTokens, Double topP, Object stop,
            Boolean stream) {
        float temp = temperature != null ? temperature.floatValue() : 0.7f;
        int maxT = maxTokens != null ? maxTokens : 2048;
        List<String> stopStrings = new ArrayList<>();
        if (stop instanceof String s)
            stopStrings.add(s);
        else if (stop instanceof List<?> l)
            l.forEach(o -> stopStrings.add(o.toString()));

        return GenerationConfig.builder()
                .temperature(temp)
                .strategy(temp <= 0.01f ? GenerationConfig.SamplingStrategy.GREEDY
                        : GenerationConfig.SamplingStrategy.TOP_K_TOP_P)
                .topK(50).topP(topP != null ? topP.floatValue() : 0.95f)
                .maxNewTokens(maxT).stopStrings(stopStrings)
                .useKvCache(true).maxKvCacheTokens(8192).build();
    }

    private ChatCompletionResponse buildChatResponse(ChatCompletionRequest req, String content, int promptTokens,
            int completionTokens) {
        return new ChatCompletionResponse(
                "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 28),
                "chat.completion", (int) Instant.now().getEpochSecond(), req.model != null ? req.model : "gollek",
                List.of(new ChatChoice(new ChatMessage("assistant", content), 0, "stop")),
                new Usage(promptTokens, completionTokens, promptTokens + completionTokens));
    }

    private String buildInferenceChunk(String id, String model, String delta) {
        try {
            return "data: "
                    + objectMapper.writeValueAsString(Map.of("id", id, "object", "chat.completion.chunk", "model",
                            model,
                            "choices", List.of(Map.of("index", 0, "delta",
                                    Map.of("content", delta != null ? delta : ""), "finish_reason", (Object) null))))
                    + "\n\n";
        } catch (Exception e) {
            return "data: {}\n\n";
        }
    }

    private String buildStreamDone() {
        return "data: [DONE]\n\n";
    }

    private String buildStreamError(String message) {
        return "data: {\"error\":{\"message\":\"" + message + "\",\"type\":\"internal_error\"}}\n\n";
    }

    private static int estimateTokens(String text) {
        if (text == null || text.isBlank())
            return 0;
        return (int) Math.ceil(text.length() / 4.0);
    }

    private static Map<String, Object> errorBody(String type, String message) {
        return Map.of("error", Map.of("type", type, "message", message));
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class ChatCompletionRequest {
        @JsonProperty("model")
        public String model;
        @JsonProperty("messages")
        public List<ChatMessage> messages;
        @JsonProperty("temperature")
        public Double temperature;
        @JsonProperty("max_tokens")
        public Integer maxTokens;
        @JsonProperty("top_p")
        public Double topP;
        @JsonProperty("stop")
        public Object stop;
        @JsonProperty("stream")
        public Boolean stream;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class CompletionRequest {
        @JsonProperty("model")
        public String model;
        @JsonProperty("prompt")
        public String prompt;
        @JsonProperty("temperature")
        public Double temperature;
        @JsonProperty("max_tokens")
        public Integer maxTokens;
        @JsonProperty("top_p")
        public Double topP;
        @JsonProperty("stop")
        public Object stop;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class ChatMessage {
        @JsonProperty("role")
        public String role;
        @JsonProperty("content")
        public String content;

        public ChatMessage() {
        }

        public ChatMessage(String role, String content) {
            this.role = role;
            this.content = content;
        }
    }

    public record ChatCompletionResponse(String id, String object, int created, String model, List<ChatChoice> choices,
            Usage usage) {
    }

    public record ChatChoice(ChatMessage message, int index, String finish_reason) {
    }

    public record CompletionResponse(String id, String object, String model, int created,
            List<CompletionChoice> choices, Usage usage) {
    }

    public record CompletionChoice(String text, int index, Object logprobs, String finish_reason) {
    }

    public record Usage(int prompt_tokens, int completion_tokens, int total_tokens) {
    }

    public record ModelsResponse(String object, List<ModelObject> data) {
    }

    public record ModelObject(String id, String object, int created, String owned_by) {
    }

    public static final class ModelLoadRequest {
        public String modelId;
        public String modelPath;
        public String adapterPath;
        public String alias;
    }

    public record ModelLoadResponse(String alias, String key, String status) {
    }
}
