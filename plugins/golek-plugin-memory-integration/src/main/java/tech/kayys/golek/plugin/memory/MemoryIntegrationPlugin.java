/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.memory;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;
import tech.kayys.wayang.memory.impl.VectorAgentMemory;
import tech.kayys.wayang.memory.spi.MemoryEntry;

import java.util.*;

/**
 * Plugin for integrating long-term memory retrieval.
 * <p>
 * Bound to {@link InferencePhase#PRE_PROCESSING}.
 * Retrieves relevant past conversation context or facts using
 * {@link VectorAgentMemory}
 * and injects them into the current request context.
 */
@ApplicationScoped
public class MemoryIntegrationPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(MemoryIntegrationPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/memory-integration";

    private boolean enabled = true;
    private int maxResults = 5;
    private double minScore = 0.7;
    private Map<String, Object> config = new HashMap<>();

    @Inject
    VectorAgentMemory memoryService;

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
        return 5; // Should run BEFORE PromptControl (order 10) to inject context
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("maxResults").ifPresent(v -> this.maxResults = Integer.parseInt(v));
        context.getConfig("minScore").ifPresent(v -> this.minScore = Double.parseDouble(v));
        LOG.infof("Initialized %s (maxResults: %d, minScore: %.2f)", PLUGIN_ID, maxResults, minScore);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new PluginException("Request not found"));

        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            return;
        }

        // 1. Extract query from the latest user message
        Message lastUserMsg = getLastUserMessage(request.getMessages());
        if (lastUserMsg == null) {
            return;
        }

        String query = lastUserMsg.getContent();
        LOG.debugf("Retrieving memory for query: %s", abbreviate(query, 50));

        // 2. Query Memory Service
        try {
            // VectorAgentMemory uses retrieve(agentId, query, limit) which returns
            // Uni<List<MemoryEntry>>
            // We need to extract agentId from request or context
            String agentId = request.getTenantId(); // Assuming tenantId serves as agentId

            // Call retrieve and block to get results (since we're in a synchronous plugin
            // context)
            List<MemoryEntry> entries = memoryService.retrieve(agentId, query, maxResults)
                    .await().indefinitely();

            if (entries.isEmpty()) {
                LOG.debug("No relevant memories found");
                return;
            }

            // 3. Inject retrieved context
            List<String> memoryContexts = new ArrayList<>();
            for (MemoryEntry entry : entries) {
                // MemoryEntry has content() method
                memoryContexts.add(entry.content());
            }

            if (!memoryContexts.isEmpty()) {
                context.putVariable("retrievedMemories", memoryContexts);

                // Option A: Append to system prompt (handled by PromptControlPlugin via
                // variable)
                // Option B: Inject immediate context message
                Message contextMsg = Message.system(
                        "Relevant Context:\n" + String.join("\n---\n", memoryContexts));

                // We modify the request messages list (assuming valid mutable list or copy)
                // Better approach: PromptControlPlugin should look for "retrievedMemories"
                // variable
                // checking implementation plan: "Inject as Message.system()"
                List<Message> newMessages = new ArrayList<>(request.getMessages());
                newMessages.add(0, contextMsg); // Prepend context

                // Update request object or context variable?
                // Since Request is immutable-ish or shared, we might need a way to pass this
                // For now, let's assume we can modify the list or use a context variable that
                // PromptPlugin reads.
                // PromptPlugin reads "request.getMessages()".
                // We can't easily mutate request here if it's immutable.
                // We'll store it in context variable "injectedContextMessages" defined by
                // convention
                context.putVariable("injectedContextMessages", List.of(contextMsg));

                LOG.debugf("Injected %d memory contexts", memoryContexts.size());
            }

        } catch (Exception e) {
            LOG.errorf("Error retrieving memory: %s", e.getMessage());
            // Don't fail the entire inference request just because memory failed
        }
    }

    private Message getLastUserMessage(List<Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message m = messages.get(i);
            if (m.getRole() == Message.Role.USER) {
                return m;
            }
        }
        return null;
    }

    private String abbreviate(String str, int max) {
        if (str.length() <= max)
            return str;
        return str.substring(0, max) + "...";
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of("enabled", enabled, "maxResults", maxResults);
    }
}
