/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.prompt;

import tech.kayys.gollek.spi.Message;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Context window management strategies.
 * <p>
 * Provides multiple strategies for fitting conversation history
 * within a model's maximum token budget.
 */
public class ContextWindowManager {

    /**
     * Strategy for managing context window overflow.
     */
    public enum Strategy {
        /** Truncate oldest non-system messages */
        TRUNCATE_OLDEST,
        /** Keep only the most recent N messages */
        SLIDING_WINDOW,
        /** Summarize older messages (placeholder for future summarization backend) */
        SUMMARIZE
    }

    private final Strategy strategy;
    private final int slidingWindowSize;

    public ContextWindowManager(Strategy strategy) {
        this(strategy, 10);
    }

    public ContextWindowManager(Strategy strategy, int slidingWindowSize) {
        this.strategy = strategy;
        this.slidingWindowSize = slidingWindowSize;
    }

    /**
     * Apply the context window strategy to fit messages within the token budget.
     *
     * @param messages  conversation messages
     * @param maxTokens maximum token budget
     * @return messages fitting within the budget
     */
    public List<Message> apply(List<Message> messages, int maxTokens) {
        return switch (strategy) {
            case TRUNCATE_OLDEST -> truncateOldest(messages, maxTokens);
            case SLIDING_WINDOW -> slidingWindow(messages);
            case SUMMARIZE -> summarize(messages, maxTokens);
        };
    }

    private List<Message> truncateOldest(List<Message> messages, int maxTokens) {
        int totalTokens = messages.stream()
                .mapToInt(m -> estimateTokens(m.getContent()))
                .sum();

        if (totalTokens <= maxTokens) {
            return messages;
        }

        var result = new ArrayList<Message>();
        int budget = maxTokens;

        // Keep system messages
        for (Message msg : messages) {
            if (msg.getRole() == Message.Role.SYSTEM) {
                result.add(msg);
                budget -= estimateTokens(msg.getContent());
            }
        }

        // Add from most recent
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

    private List<Message> slidingWindow(List<Message> messages) {
        var result = new ArrayList<Message>();

        // Always keep system messages
        for (Message msg : messages) {
            if (msg.getRole() == Message.Role.SYSTEM) {
                result.add(msg);
            }
        }

        // Keep only last N non-system messages
        var nonSystem = messages.stream()
                .filter(m -> m.getRole() != Message.Role.SYSTEM)
                .toList();

        int start = Math.max(0, nonSystem.size() - slidingWindowSize);
        result.addAll(nonSystem.subList(start, nonSystem.size()));

        return result;
    }

    private List<Message> summarize(List<Message> messages, int maxTokens) {
        // Placeholder: falls back to truncation for now.
        // A real implementation would call a summarization model/service.
        return truncateOldest(messages, maxTokens);
    }

    private int estimateTokens(String text) {
        return text == null ? 0 : (text.length() + 3) / 4;
    }
}
