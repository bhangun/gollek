/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.prompt;

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

import java.util.*;

/**
 * Plugin for prompt construction and context window management.
 * <p>
 * Bound to {@link InferencePhase#PRE_PROCESSING}.
 * Builds the final prompt from structured messages, applies templates,
 * manages context window (truncate/summarize), and injects system instructions.
 */
@ApplicationScoped
public class PromptControlPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(PromptControlPlugin.class);
    private static final String PLUGIN_ID = "tech.kayys/prompt-control";

    private boolean enabled = true;
    private int maxContextTokens = 4096;
    private String defaultSystemPrompt = "";
    private Map<String, Object> config = new HashMap<>();

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
        return 10; // Early in PRE_PROCESSING so other plugins see the built prompt
    }

    @Override
    public void initialize(PluginContext context) {
        context.getConfig("enabled").ifPresent(v -> this.enabled = Boolean.parseBoolean(v));
        context.getConfig("maxContextTokens").ifPresent(v -> this.maxContextTokens = Integer.parseInt(v));
        context.getConfig("defaultSystemPrompt").ifPresent(v -> this.defaultSystemPrompt = v);
        LOG.infof("Initialized %s (maxContextTokens: %d)", PLUGIN_ID, maxContextTokens);
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        return enabled;
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        InferenceRequest request = context.getVariable("request", InferenceRequest.class)
                .orElseThrow(() -> new PluginException("Request not found in execution context"));

        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            LOG.debug("No messages to process");
            return;
        }

        // 1. Inject default system prompt if none present
        messages = injectDefaultSystemPrompt(messages);

        // 2. Manage context window
        messages = manageContextWindow(messages, maxContextTokens);

        // 3. Build the final prompt string
        String prompt = buildPrompt(messages);

        // Store in context for downstream plugins and provider dispatch
        context.putVariable("builtPrompt", prompt);
        context.putVariable("processedMessages", messages);

        LOG.debugf("Built prompt (%d chars, est. %d tokens) for request %s",
                prompt.length(), estimateTokens(prompt), request.getRequestId());
    }

    /**
     * Build a prompt string from structured messages.
     */
    String buildPrompt(List<Message> messages) {
        var sb = new StringBuilder();
        for (Message msg : messages) {
            String roleLabel = formatRole(msg.getRole());
            String content = msg.getContent() != null ? msg.getContent() : "";
            sb.append(roleLabel).append(content).append("\n\n");
        }
        return sb.toString().trim();
    }

    /**
     * Format role prefix for prompt construction.
     */
    private String formatRole(Message.Role role) {
        if (role == null) return "### User:\n";
        return switch (role) {
            case SYSTEM -> "### System:\n";
            case USER -> "### User:\n";
            case ASSISTANT -> "### Assistant:\n";
            case TOOL -> "### Tool Result:\n";
            case FUNCTION -> "### Function:\n";
        };
    }

    /**
     * Inject default system prompt if no system message exists.
     */
    List<Message> injectDefaultSystemPrompt(List<Message> messages) {
        if (defaultSystemPrompt == null || defaultSystemPrompt.isBlank()) {
            return messages;
        }

        boolean hasSystem = messages.stream()
                .anyMatch(m -> m.getRole() == Message.Role.SYSTEM);

        if (!hasSystem) {
            var result = new ArrayList<Message>();
            result.add(Message.system(defaultSystemPrompt));
            result.addAll(messages);
            return result;
        }
        return messages;
    }

    /**
     * Manage context window by truncating oldest messages if total exceeds max tokens.
     * System messages are always preserved.
     */
    List<Message> manageContextWindow(List<Message> messages, int maxTokens) {
        int totalTokens = messages.stream()
                .mapToInt(m -> estimateTokens(m.getContent()))
                .sum();

        if (totalTokens <= maxTokens) {
            return messages;
        }

        LOG.debugf("Context exceeds %d tokens (actual: %d), truncating", maxTokens, totalTokens);

        // Keep system messages and recent messages, truncate middle
        var result = new ArrayList<Message>();
        int budget = maxTokens;

        // Always keep system messages
        for (Message msg : messages) {
            if (msg.getRole() == Message.Role.SYSTEM) {
                result.add(msg);
                budget -= estimateTokens(msg.getContent());
            }
        }

        // Add messages from most recent, skipping system
        var nonSystem = messages.stream()
                .filter(m -> m.getRole() != Message.Role.SYSTEM)
                .toList();

        var reversed = new ArrayList<>(nonSystem);
        Collections.reverse(reversed);

        var recent = new ArrayList<Message>();
        for (Message msg : reversed) {
            int tokens = estimateTokens(msg.getContent());
            if (budget - tokens >= 0) {
                recent.add(0, msg);
                budget -= tokens;
            } else {
                break;
            }
        }

        result.addAll(recent);
        return result;
    }

    /**
     * Rough token estimate: ~4 characters per token.
     */
    int estimateTokens(String text) {
        return text == null ? 0 : (text.length() + 3) / 4;
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.maxContextTokens = (Integer) newConfig.getOrDefault("maxContextTokens", 4096);
        this.defaultSystemPrompt = (String) newConfig.getOrDefault("defaultSystemPrompt", "");
    }

    @Override
    public Map<String, Object> currentConfig() {
        return Map.of(
                "enabled", enabled,
                "maxContextTokens", maxContextTokens,
                "defaultSystemPrompt", defaultSystemPrompt);
    }
}
