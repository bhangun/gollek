/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * RAG pipeline REST interface — ingest, query, and manage document collections.
 * Uses reflection to access RagPipeline to avoid circular dependencies.
 */
@jakarta.ws.rs.Path("/v1/rag")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "RAG", description = "Retrieval-Augmented Generation pipeline")
public class RagResource {

    private static final org.jboss.logging.Logger log = org.jboss.logging.Logger.getLogger(RagResource.class);

    @Inject
    jakarta.enterprise.inject.Instance<Object> ragInstance;

    @ConfigProperty(name = "gollek.rag.default-embedding-model", defaultValue = "")
    String defaultEmbeddingModel;

    @ConfigProperty(name = "gollek.rag.default-llm-model", defaultValue = "")
    String defaultLlmModel;

    private final Map<String, java.nio.file.Path> modelPaths = new ConcurrentHashMap<>();

    private Object getRag() {
        try {
            return ragInstance.get();
        } catch (Exception e) {
            log.error("RagPipeline not found", e);
            return null;
        }
    }

    @POST
    @jakarta.ws.rs.Path("/ingest")
    @Operation(summary = "Chunk, embed, and index documents into a collection")
    public Uni<IngestResponse> ingest(IngestRequest req) {
        if (req == null || req.documents == null || req.documents.isEmpty())
            throw new WebApplicationException(
                    Response.status(400).entity(err("Documents list is required")).build());

        java.nio.file.Path embModel = resolveModel(req.embeddingModel, defaultEmbeddingModel);
        String collection = req.collection != null ? req.collection : "default";

        Object rag = getRag();
        if (rag == null)
            return Uni.createFrom().failure(new RuntimeException("RagPipeline not available"));

        try {
            Class<?> docClass = Class.forName("tech.kayys.gollek.safetensor.api.RagPipeline$Document");
            List<Object> docs = new ArrayList<>();
            for (var d : req.documents) {
                Object doc = docClass.getConstructor(String.class, String.class)
                        .newInstance(d.id != null ? d.id : UUID.randomUUID().toString(),
                                d.content != null ? d.content : "");
                docs.add(doc);
            }

            // Uni<Integer> ingest(String collection, List<Document> documents, Path
            // embeddingModel)
            Uni<Integer> uni = (Uni<Integer>) rag.getClass()
                    .getMethod("ingest", String.class, List.class, java.nio.file.Path.class)
                    .invoke(rag, collection, docs, embModel);

            return uni.map(count -> new IngestResponse(collection, count, "indexed"))
                    .onFailure().transform(t -> new WebApplicationException(
                            Response.status(500).entity(err(t.getMessage())).build()));
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    @POST
    @jakarta.ws.rs.Path("/query")
    @Operation(summary = "Retrieve relevant chunks and generate an answer")
    public Uni<RagQueryResponse> query(RagQueryRequest req) {
        if (req == null || req.question == null || req.question.isBlank())
            throw new WebApplicationException(
                    Response.status(400).entity(err("question is required")).build());

        java.nio.file.Path embModel = resolveModel(req.embeddingModel, defaultEmbeddingModel);
        java.nio.file.Path llmModel = resolveModel(req.llmModel, defaultLlmModel);
        String collection = req.collection != null ? req.collection : "default";

        GenerationConfig gc = GenerationConfig.builder()
                .temperature(req.temperature != null ? req.temperature.floatValue() : 0.1f)
                .strategy(GenerationConfig.SamplingStrategy.GREEDY)
                .maxNewTokens(req.maxTokens != null ? req.maxTokens : 1024)
                .useKvCache(false)
                .build();

        Object rag = getRag();
        if (rag == null)
            return Uni.createFrom().failure(new RuntimeException("RagPipeline not available"));

        try {
            // Uni<QueryResult> query(String collection, String question, Path embModel,
            // Path llmModel, GenerationConfig gc)
            Uni<?> uni = (Uni<?>) rag.getClass()
                    .getMethod("query", String.class, String.class, java.nio.file.Path.class, java.nio.file.Path.class,
                            GenerationConfig.class)
                    .invoke(rag, collection, req.question, embModel, llmModel, gc);

            return uni.map(result -> {
                try {
                    String answer = (String) result.getClass().getMethod("answer").invoke(result);
                    boolean contextUsed = (boolean) result.getClass().getMethod("contextUsed").invoke(result);
                    List<?> sources = (List<?>) result.getClass().getMethod("sources").invoke(result);

                    List<SourceDto> sourceDtos = new ArrayList<>();
                    for (Object s : sources) {
                        String sid = (String) s.getClass().getMethod("sourceId").invoke(s);
                        String exc = (String) s.getClass().getMethod("excerpt").invoke(s);
                        float sim = (float) s.getClass().getMethod("similarity").invoke(s);
                        sourceDtos.add(new SourceDto(sid, exc, sim));
                    }
                    return new RagQueryResponse(answer, contextUsed, sourceDtos);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }).onFailure().transform(t -> new WebApplicationException(
                    Response.status(500).entity(err(t.getMessage())).build()));
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    @GET
    @jakarta.ws.rs.Path("/collections")
    @Operation(summary = "List all indexed collections")
    public CollectionsResponse listCollections() {
        Object rag = getRag();
        if (rag == null)
            return new CollectionsResponse(List.of());

        try {
            List<String> names = (List<String>) rag.getClass().getMethod("listCollections").invoke(rag);
            List<CollectionInfo> infos = new ArrayList<>();
            for (String name : names) {
                int count = (int) rag.getClass().getMethod("chunkCount", String.class).invoke(rag, name);
                infos.add(new CollectionInfo(name, count));
            }
            return new CollectionsResponse(infos);
        } catch (Exception e) {
            return new CollectionsResponse(List.of());
        }
    }

    @GET
    @jakarta.ws.rs.Path("/collections/{name}")
    @Operation(summary = "Get collection statistics")
    public CollectionInfo getCollection(@PathParam("name") String name) {
        Object rag = getRag();
        if (rag == null)
            throw new WebApplicationException(Response.status(503).build());

        try {
            List<String> names = (List<String>) rag.getClass().getMethod("listCollections").invoke(rag);
            if (!names.contains(name))
                throw new WebApplicationException(
                        Response.status(404).entity(err("Collection '" + name + "' not found")).build());

            int count = (int) rag.getClass().getMethod("chunkCount", String.class).invoke(rag, name);
            return new CollectionInfo(name, count);
        } catch (Exception e) {
            throw new WebApplicationException(Response.status(500).build());
        }
    }

    @DELETE
    @jakarta.ws.rs.Path("/collections/{name}")
    @Operation(summary = "Drop a collection")
    public Response dropCollection(@PathParam("name") String name) {
        Object rag = getRag();
        if (rag != null) {
            try {
                rag.getClass().getMethod("dropCollection", String.class).invoke(rag, name);
                log.infof("RAG: dropped collection '%s'", name);
            } catch (Exception e) {
                log.error("Drop failed", e);
            }
        }
        return Response.ok(Map.of("status", "dropped", "collection", name)).build();
    }

    @POST
    @jakarta.ws.rs.Path("/models")
    @Operation(summary = "Register a model alias for RAG use")
    public Response registerModel(ModelRegistration req) {
        if (req == null || req.alias == null || req.path == null)
            throw new WebApplicationException(Response.status(400).build());
        modelPaths.put(req.alias, java.nio.file.Path.of(req.path));
        return Response.ok(Map.of("alias", req.alias, "path", req.path, "status", "registered")).build();
    }

    private java.nio.file.Path resolveModel(String alias, String defaultAlias) {
        String key = alias != null && !alias.isBlank() ? alias : defaultAlias;
        if (key == null || key.isBlank())
            throw new WebApplicationException(
                    Response.status(400).entity(err("No model specified and no default configured."))
                            .build());
        java.nio.file.Path p = modelPaths.get(key);
        if (p != null)
            return p;
        p = java.nio.file.Path.of(key);
        if (java.nio.file.Files.exists(p)) {
            modelPaths.put(key, p);
            return p;
        }
        throw new WebApplicationException(Response.status(404).entity(err("Model '" + key + "' not found.")).build());
    }

    private static Map<String, Object> err(String msg) {
        return Map.of("error", Map.of("message", msg));
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class IngestRequest {
        public String collection;
        public String embeddingModel;
        public List<DocInput> documents;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class DocInput {
        public String id;
        public String content;
    }

    public record IngestResponse(String collection, int chunks, String status) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class RagQueryRequest {
        public String collection;
        public String question;
        public String embeddingModel;
        public String llmModel;
        public Integer maxTokens;
        public Double temperature;
        public Integer topK;
    }

    public record RagQueryResponse(String answer, boolean context_used, List<SourceDto> sources) {
    }

    public record SourceDto(String source_id, String excerpt, float similarity) {
    }

    public record CollectionsResponse(List<CollectionInfo> collections) {
    }

    public record CollectionInfo(String name, int chunks) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class ModelRegistration {
        public String alias;
        public String path;
    }
}
