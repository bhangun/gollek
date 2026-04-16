/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.api;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.nio.file.Path;
import java.util.List;

/**
 * Interface for the RAG pipeline.
 */
public interface RagPipeline {

    Uni<Integer> ingest(String collection, List<Document> documents, Path embeddingModel);

    Uni<RagResult> query(String collection, String question, Path embeddingModel, Path llmModel, GenerationConfig config);

    List<String> listCollections();

    int chunkCount(String collection);

    void dropCollection(String collection);

    record Document(String id, String content) {}

    record RagResult(String answer, boolean contextUsed, List<Source> sources) {}

    record Source(String sourceId, String excerpt, float similarity) {}
}
