/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.rag;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.gollek.core.execution.ExecutionContext;
import tech.kayys.gollek.core.plugin.InferencePhasePlugin;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.plugin.PluginException;

import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.*;

/**
 * Plugin for Retrieval-Augmented Generation (RAG).
 * <p>
 * Bound to {@link InferencePhase#PRE_PROCESSING} (after prompt building).
 * Retrieves relevant external documents, reranks them, and injects into context.
 */
@ApplicationScoped
public class RAGPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(RAGPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/rag";

    private boolean enabled = true;
    private int topK = 5;
    private double similarityThreshold = 0.7;
    private Map<String, Object> config = new HashMap<>();

    @Inject
    Instance<RetrievalService> retrievalServices;

    @Inject
    Instance<Reranker> rerankers;

    @Inject
    Instance<EmbeddingProvider> embeddingProviders;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return 50; // After prompt building (order 10)
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("topK").ifPresent(v -> this.topK = Integer.parseInt(v));
        context.getConfig("similarityThreshold").ifPresent(v -> this.similarityThreshold = Double.parseDouble(v));
        LOG.infof("Initialized %s (topK: %d, threshold: %.2f)", PLUGIN_ID, topK, similarityThreshold);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) return false;
        // Only execute if RAG is requested for this inference
        return context.getVariable("ragEnabled", Boolean.class).orElse(false);
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new PluginException("Request not found in execution context"));

        String query = extractQuery(request);
        if (query == null || query.isBlank()) {
            LOG.debug("No query found for RAG retrieval");
            return;
        }

        try {
            // 1. Get embeddings for the query
            float[] queryEmbedding = getEmbedding(query);

            // 2. Retrieve relevant documents
            List<RetrievalService.RetrievedDocument> documents = retrieve(queryEmbedding);

            // 3. Rerank documents
            documents = rerank(query, documents);

            // 4. Inject documents into context
            if (!documents.isEmpty()) {
                String ragContext = formatDocuments(documents);
                context.putVariable("ragContext", ragContext);
                context.putVariable("ragDocumentCount", documents.size());

                LOG.debugf("Injected %d RAG documents for request %s",
                        documents.size(), request.getRequestId());
            }
        } catch (Exception e) {
            LOG.warnf(e, "RAG retrieval failed, continuing without augmentation");
        }
    }

    private String extractQuery(InferenceRequest request) {
        var messages = request.getMessages();
        if (messages == null || messages.isEmpty()) return null;

        // Find the last user message
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i).getRole() == Message.Role.USER) {
                return messages.get(i).getContent();
            }
        }
        return null;
    }

    private float[] getEmbedding(String text) {
        if (embeddingProviders != null && !embeddingProviders.isUnsatisfied()) {
            return embeddingProviders.get().embed(text);
        }
        return new float[0];
    }

    private List<RetrievalService.RetrievedDocument> retrieve(float[] embedding) {
        if (retrievalServices != null && !retrievalServices.isUnsatisfied()) {
            return retrievalServices.get().retrieve(embedding, topK, similarityThreshold);
        }
        return Collections.emptyList();
    }

    private List<RetrievalService.RetrievedDocument> rerank(String query,
            List<RetrievalService.RetrievedDocument> documents) {
        if (rerankers != null && !rerankers.isUnsatisfied()) {
            return rerankers.get().rerank(query, documents, topK);
        }
        return documents;
    }

    private String formatDocuments(List<RetrievalService.RetrievedDocument> documents) {
        var sb = new StringBuilder("### Relevant Context:\n\n");
        for (int i = 0; i < documents.size(); i++) {
            var doc = documents.get(i);
            sb.append("[Source ").append(i + 1);
            if (doc.source() != null) {
                sb.append(": ").append(doc.source());
            }
            sb.append("]\n").append(doc.content()).append("\n\n");
        }
        return sb.toString();
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.topK = (Integer) newConfig.getOrDefault("topK", 5);
        this.similarityThreshold = (Double) newConfig.getOrDefault("similarityThreshold", 0.7);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "topK", topK,
                "similarityThreshold", similarityThreshold);
    }
}
