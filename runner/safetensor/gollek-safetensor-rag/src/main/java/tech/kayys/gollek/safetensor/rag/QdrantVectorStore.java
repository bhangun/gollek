/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QdrantVectorStore.java
 * ───────────────────────
 * Qdrant vector database client for the RAG pipeline.
 *
 * Qdrant is a high-performance vector search engine written in Rust.
 * It supports: exact ANN search, filtering, payload storage, quantization,
 * and sharding.
 *
 * API endpoints used:
 *   PUT  /collections/{name}              — create/ensure collection
 *   PUT  /collections/{name}/points       — upsert vectors with payloads
 *   POST /collections/{name}/points/search — k-NN search with optional filters
 *   DELETE /collections/{name}            — drop collection
 *
 * Configuration:
 *   gollek.rag.qdrant.url=http://localhost:6333
 *   gollek.rag.qdrant.api-key=          (optional, for Qdrant Cloud)
 *   gollek.rag.qdrant.timeout-s=30
 */
package tech.kayys.gollek.safetensor.rag;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

/**
 * Qdrant REST API client for vector storage and nearest-neighbour search.
 *
 * <p>
 * Used by {@link RagPipeline} when {@code gollek.rag.backend=qdrant}.
 */
@ApplicationScoped
public class QdrantVectorStore {

    private static final Logger log = Logger.getLogger(QdrantVectorStore.class);

    @ConfigProperty(name = "gollek.rag.qdrant.url", defaultValue = "http://localhost:6333")
    String qdrantUrl;

    @ConfigProperty(name = "gollek.rag.qdrant.api-key", defaultValue = "")
    String apiKey;

    @ConfigProperty(name = "gollek.rag.qdrant.timeout-s", defaultValue = "30")
    int timeoutS;

    @Inject
    ObjectMapper objectMapper;

    private HttpClient httpClient;

    @jakarta.annotation.PostConstruct
    void init() {
        httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(timeoutS))
                .build();
        log.infof("QdrantVectorStore: connected to %s", qdrantUrl);
    }

    // ── Collection management ─────────────────────────────────────────────────

    /**
     * Create a collection if it doesn't already exist.
     *
     * @param name      collection name
     * @param vectorDim embedding dimension
     * @param distance  "Cosine" | "Euclid" | "Dot"
     */
    public void ensureCollection(String name, int vectorDim, String distance) throws IOException {
        // Check if collection exists
        try {
            HttpResponse<String> check = get("/collections/" + name);
            if (check.statusCode() == 200) {
                log.debugf("Qdrant: collection '%s' already exists", name);
                return;
            }
        } catch (Exception e) {
            // ignore — will create below
        }

        // Create
        Map<String, Object> body = Map.of(
                "vectors", Map.of(
                        "size", vectorDim,
                        "distance", distance != null ? distance : "Cosine"));

        put("/collections/" + name, body);
        log.infof("Qdrant: created collection '%s' (dim=%d, distance=%s)",
                name, vectorDim, distance);
    }

    /**
     * Drop a collection.
     */
    public void dropCollection(String name) throws IOException {
        delete("/collections/" + name);
        log.infof("Qdrant: dropped collection '%s'", name);
    }

    // ── Vector upsert ─────────────────────────────────────────────────────────

    /**
     * Upsert a batch of vectors with their payloads.
     *
     * @param collection collection name
     * @param points     list of (id, embedding, payload) to upsert
     */
    public void upsert(String collection, List<QdrantPoint> points) throws IOException {
        if (points.isEmpty())
            return;

        List<Map<String, Object>> pointMaps = points.stream()
                .map(p -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("id", p.id());
                    List<Float> vecList = new ArrayList<>(p.vector().length);
                    for (float f : p.vector())
                        vecList.add(f);
                    m.put("vector", vecList);
                    m.put("payload", p.payload());
                    return m;
                })
                .toList();

        Map<String, Object> body = Map.of("points", pointMaps);
        put("/collections/" + collection + "/points?wait=true", body);
        log.debugf("Qdrant: upserted %d points into '%s'", points.size(), collection);
    }

    // ── Vector search ─────────────────────────────────────────────────────────

    /**
     * Search for nearest neighbours.
     *
     * @param collection     collection name
     * @param queryVec       query embedding
     * @param topK           number of results
     * @param scoreThreshold minimum similarity score (null = no threshold)
     * @return list of (id, score, payload) sorted by descending score
     */
    public List<SearchResult> search(String collection, float[] queryVec,
            int topK, Float scoreThreshold) throws IOException {
        List<Float> vec = new ArrayList<>(queryVec.length);
        for (float f : queryVec)
            vec.add(f);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("vector", vec);
        body.put("limit", topK);
        body.put("with_payload", true);
        body.put("with_vector", false);
        if (scoreThreshold != null)
            body.put("score_threshold", scoreThreshold);

        HttpResponse<String> resp = post(
                "/collections/" + collection + "/points/search", body);

        if (resp.statusCode() != 200) {
            log.warnf("Qdrant search failed: %d %s", resp.statusCode(), resp.body());
            return List.of();
        }

        QdrantSearchResponse parsed = objectMapper.readValue(
                resp.body(), QdrantSearchResponse.class);

        if (parsed.result == null)
            return List.of();

        return parsed.result.stream()
                .map(r -> new SearchResult(r.id, r.score, r.payload != null ? r.payload : Map.of()))
                .toList();
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    private HttpResponse<String> get(String path) throws IOException {
        HttpRequest req = baseRequest(path).GET().build();
        return send(req);
    }

    private HttpResponse<String> put(String path, Object body) throws IOException {
        String json = objectMapper.writeValueAsString(body);
        HttpRequest req = baseRequest(path)
                .PUT(HttpRequest.BodyPublishers.ofString(json))
                .build();
        return send(req);
    }

    private HttpResponse<String> post(String path, Object body) throws IOException {
        String json = objectMapper.writeValueAsString(body);
        HttpRequest req = baseRequest(path)
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        return send(req);
    }

    private HttpResponse<String> delete(String path) throws IOException {
        HttpRequest req = baseRequest(path).DELETE().build();
        return send(req);
    }

    private HttpRequest.Builder baseRequest(String path) {
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(qdrantUrl + path))
                .timeout(Duration.ofSeconds(timeoutS))
                .header("Content-Type", "application/json");
        if (!apiKey.isBlank())
            b.header("api-key", apiKey);
        return b;
    }

    private HttpResponse<String> send(HttpRequest req) throws IOException {
        try {
            return httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Qdrant request interrupted", e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Value types
    // ─────────────────────────────────────────────────────────────────────────

    public record QdrantPoint(
            String id,
            float[] vector,
            Map<String, Object> payload) {
    }

    public record SearchResult(
            String id,
            float score,
            Map<String, Object> payload) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class QdrantSearchResponse {
        @JsonProperty("result")
        List<QdrantHit> result;
        @JsonProperty("status")
        String status;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class QdrantHit {
        @JsonProperty("id")
        String id;
        @JsonProperty("score")
        float score;
        @JsonProperty("payload")
        Map<String, Object> payload;
    }
}
