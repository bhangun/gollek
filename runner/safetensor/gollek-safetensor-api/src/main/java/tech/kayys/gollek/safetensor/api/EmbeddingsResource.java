/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.*;
import java.util.concurrent.Executors;

/**
 * OpenAI-compatible /v1/embeddings endpoint.
 * Uses reflection to access EmbeddingEngine to avoid circular dependencies.
 */
@jakarta.ws.rs.Path("/v1/embeddings")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "Embeddings", description = "Text embedding generation")
public class EmbeddingsResource {

    private static final org.jboss.logging.Logger log = org.jboss.logging.Logger.getLogger(EmbeddingsResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> embeddingEngineInstance;

    /** Model registry (shared with OpenAiCompatibleResource via CDI). */
    @Inject
    OpenAiCompatibleResource openAiResource;

    private Object getEngine() {
        try {
            return embeddingEngineInstance.get();
        } catch (Exception e) {
            log.error("EmbeddingEngine not found", e);
            return null;
        }
    }

    @POST
    @Operation(summary = "Create text embeddings")
    public Uni<EmbeddingResponse> createEmbeddings(EmbeddingRequest req) {
        if (req == null || req.input == null) {
            throw new WebApplicationException(
                    jakarta.ws.rs.core.Response.status(400)
                            .entity(Map.of("error", Map.of("type", "invalid_request",
                                    "message", "input is required")))
                            .build());
        }

        return Uni.createFrom().item(() -> {
            List<String> texts = extractTexts(req.input);
            String model = req.model != null ? req.model : "default";

            java.nio.file.Path modelPath = resolveModelPath(model);
            log.debugf("Embeddings: %d texts, model=%s", texts.size(), model);

            Object engine = getEngine();
            if (engine == null)
                throw new RuntimeException("EmbeddingEngine not available");

            try {
                // List<float[]> embedBatch(List<String> texts, Path modelPath)
                List<float[]> embeddings = (List<float[]>) engine.getClass()
                        .getMethod("embedBatch", List.class, java.nio.file.Path.class)
                        .invoke(engine, texts, modelPath);

                List<EmbeddingObject> data = new ArrayList<>();
                int totalTokens = 0;
                for (int i = 0; i < embeddings.size(); i++) {
                    data.add(new EmbeddingObject("embedding", i, embeddings.get(i)));
                    totalTokens += estimateTokens(texts.get(i));
                }

                return new EmbeddingResponse("list", data, model, new EmbeddingUsage(totalTokens, totalTokens));
            } catch (Exception e) {
                throw new RuntimeException("Embedding failed: " + e.getMessage(), e);
            }
        }).runSubscriptionOn(Executors.newVirtualThreadPerTaskExecutor());
    }

    @SuppressWarnings("unchecked")
    private List<String> extractTexts(Object input) {
        if (input instanceof String s)
            return List.of(s);
        if (input instanceof List<?> list) {
            return list.stream().map(Object::toString).toList();
        }
        return List.of(input.toString());
    }

    private java.nio.file.Path resolveModelPath(String alias) {
        Map<String, java.nio.file.Path> registry = openAiResource.getModelRegistry();
        java.nio.file.Path p = registry.get(alias);
        if (p == null) {
            java.nio.file.Path direct = java.nio.file.Path.of(alias);
            if (java.nio.file.Files.exists(direct))
                return direct;
            throw new WebApplicationException(
                    jakarta.ws.rs.core.Response.status(404)
                            .entity(Map.of("error", Map.of("type", "model_not_found",
                                    "message", "Model '" + alias + "' not loaded")))
                            .build());
        }
        return p;
    }

    private static int estimateTokens(String text) {
        return text == null ? 0 : (int) Math.ceil(text.length() / 4.0);
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class EmbeddingRequest {
        @JsonProperty("model")
        public String model;
        @JsonProperty("input")
        public Object input;
    }

    public record EmbeddingResponse(
            @JsonProperty("object") String object,
            @JsonProperty("data") List<EmbeddingObject> data,
            @JsonProperty("model") String model,
            @JsonProperty("usage") EmbeddingUsage usage) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record EmbeddingObject(
            @JsonProperty("object") String object,
            @JsonProperty("index") int index,
            @JsonProperty("embedding") float[] embedding) {
    }

    public record EmbeddingUsage(
            @JsonProperty("prompt_tokens") int promptTokens,
            @JsonProperty("total_tokens") int totalTokens) {
    }
}
