package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.*;


import java.util.List;
import java.util.Map;

/**
 * Context configuration
 */
public class ContextConfig {

    private int maxMemories = 10;
    private int reservedTokens = 1000; // Reserve for response
    private List<MemoryType> memoryTypes;
    private String systemPrompt;
    private List<String> conversationHistory;
    private String taskInstructions;
    private boolean includeMetadata = true;

    // Getters and setters
    public int getMaxMemories() {
        return maxMemories;
    }

    public void setMaxMemories(int maxMemories) {
        this.maxMemories = maxMemories;
    }

    public int getReservedTokens() {
        return reservedTokens;
    }

    public void setReservedTokens(int reservedTokens) {
        this.reservedTokens = reservedTokens;
    }

    public List<MemoryType> getMemoryTypes() {
        return memoryTypes;
    }

    public void setMemoryTypes(List<MemoryType> memoryTypes) {
        this.memoryTypes = memoryTypes;
    }

    public String getSystemPrompt() {
        return systemPrompt;
    }

    public void setSystemPrompt(String systemPrompt) {
        this.systemPrompt = systemPrompt;
    }

    public List<String> getConversationHistory() {
        return conversationHistory;
    }

    public void setConversationHistory(List<String> conversationHistory) {
        this.conversationHistory = conversationHistory;
    }

    public String getTaskInstructions() {
        return taskInstructions;
    }

    public void setTaskInstructions(String taskInstructions) {
        this.taskInstructions = taskInstructions;
    }

    public boolean isIncludeMetadata() {
        return includeMetadata;
    }

    public void setIncludeMetadata(boolean includeMetadata) {
        this.includeMetadata = includeMetadata;
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final ContextConfig config = new ContextConfig();

        public Builder maxMemories(int maxMemories) {
            config.maxMemories = maxMemories;
            return this;
        }

        public Builder reservedTokens(int reservedTokens) {
            config.reservedTokens = reservedTokens;
            return this;
        }

        public Builder memoryTypes(List<MemoryType> types) {
            config.memoryTypes = types;
            return this;
        }

        public Builder systemPrompt(String prompt) {
            config.systemPrompt = prompt;
            return this;
        }

        public Builder conversationHistory(List<String> history) {
            config.conversationHistory = history;
            return this;
        }

        public Builder taskInstructions(String instructions) {
            config.taskInstructions = instructions;
            return this;
        }

        public Builder includeMetadata(boolean include) {
            config.includeMetadata = include;
            return this;
        }

        public ContextConfig build() {
            return config;
        }
    }
}