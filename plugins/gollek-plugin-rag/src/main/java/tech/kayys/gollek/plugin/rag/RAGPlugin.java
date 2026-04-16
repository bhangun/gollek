package tech.kayys.gollek.plugin.rag;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * RAG (Retrieval-Augmented Generation) plugin that enhances prompts with
 * retrieved context from the knowledge base.
 *
 * <h3>Phases:</h3>
 * <ul>
 * <li>{@code PRE_PROCESSING} - Retrieve relevant context and enhance prompt</li>
 * </ul>
 *
 * <h3>Configuration:</h3>
 * <ul>
 * <li>{@code gollek.rag.enabled} - Enable/disable RAG (default: true)</li>
 * <li>{@code gollek.rag.top-k} - Number of contexts to retrieve (default: 5)</li>
 * <li>{@code gollek.rag.similarity-threshold} - Minimum similarity score (default: 0.7)</li>
 * <li>{@code gollek.rag.auto-enhance} - Automatically enhance prompts (default: true)</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * # Enable RAG for specific requests
 * InferenceRequest request = new InferenceRequest(
 *     "What is our return policy?",
 *     Map.of("rag_enabled", true, "rag_top_k", 3)
 * );
 * }</pre>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class RAGPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(RAGPlugin.class);

    @Inject
    RAGService ragService;

    private static final String PLUGIN_ID = "rag";
    private static final String VERSION = "1.0.0";

    // Plugin configuration
    private boolean enabled = true;
    private int order = 50; // High priority - enhance before other processing
    private int topK = 5;
    private double similarityThreshold = 0.7;
    private boolean autoEnhance = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return order;
    }

    @Override
    public void initialize(tech.kayys.gollek.spi.plugin.PluginContext context) {
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));
        this.topK = Integer.parseInt(context.getConfig("top-k", "5"));
        this.similarityThreshold = Double.parseDouble(context.getConfig("similarity-threshold", "0.7"));
        this.autoEnhance = Boolean.parseBoolean(context.getConfig("auto-enhance", "true"));
        
        LOG.infof("Initialized %s plugin (enabled: %b, topK: %d, similarity: %.2f, auto: %b)", 
            PLUGIN_ID, enabled, topK, similarityThreshold, autoEnhance);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            return false;
        }

        Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
        if (requestOpt.isEmpty()) {
            return false;
        }

        InferenceRequest request = requestOpt.get();
        Map<String, Object> metadata = request.getMetadata();
        
        // Check if RAG is explicitly disabled for this request
        if (Boolean.FALSE.equals(metadata.get("rag_enabled"))) {
            return false;
        }

        // Only process if auto-enhance is enabled or explicitly requested
        return autoEnhance || Boolean.TRUE.equals(metadata.get("rag_enabled"));
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        LOG.debug("RAG plugin processing request");

        try {
            Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
            if (requestOpt.isEmpty()) {
                return;
            }

            InferenceRequest request = requestOpt.get();
            
            // Get query from request
            String query = request.getPrompt() != null && !request.getPrompt().isBlank() 
                ? request.getPrompt() 
                : extractQuery(request);

            if (query == null || query.isBlank()) {
                LOG.debug("No query found, skipping RAG enhancement");
                return;
            }

            // Get top-k from request metadata or use default
            Map<String, Object> metadata = request.getMetadata();
            int k = metadata.containsKey("rag_top_k")
                    ? (Integer) metadata.get("rag_top_k")
                    : topK;

            // Search for relevant context
            List<RAGService.RAGContext> contexts = ragService.search(query, k);

            if (!contexts.isEmpty()) {
                LOG.infof("Retrieved %d contexts for RAG enhancement", contexts.size());

                // Enhance prompt with context
                String enhancedPrompt = ragService.enhancePrompt(query, contexts);

                // Create new request with enhanced prompt
                InferenceRequest redactedRequest = request.toBuilder()
                    .prompt(enhancedPrompt)
                    .metadata("rag_contexts", contexts.size())
                    .metadata("rag_enhanced", true)
                    .build();

                context.putVariable("request", redactedRequest);
                LOG.infof("Enhanced prompt with RAG context for request %s", request.getRequestId());

            } else {
                LOG.debug("No relevant contexts found, using original prompt");
                
                // Track that RAG was attempted but found nothing
                InferenceRequest trackerRequest = request.toBuilder()
                    .metadata("rag_contexts", 0)
                    .build();
                context.putVariable("request", trackerRequest);
            }

        } catch (Exception e) {
            LOG.warnf(e, "RAG enhancement failed, using original prompt");
        }
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "top_k", topK,
                "similarity_threshold", similarityThreshold,
                "auto_enhance", autoEnhance,
                "knowledge_base_size", ragService.getStats().get("documents"));
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", enabled);
        this.topK = (Integer) newConfig.getOrDefault("top-k", topK);
        this.similarityThreshold = (Double) newConfig.getOrDefault("similarity-threshold", similarityThreshold);
        this.autoEnhance = (Boolean) newConfig.getOrDefault("auto-enhance", autoEnhance);
        
        ragService.setTopK(topK);
        ragService.setSimilarityThreshold(similarityThreshold);
    }

    private String extractQuery(InferenceRequest request) {
        // Try to extract query from messages if prompt is not available
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            var lastMessage = request.getMessages().get(request.getMessages().size() - 1);
            if (lastMessage.getContent() != null) {
                return lastMessage.getContent();
            }
        }

        return null;
    }
}
