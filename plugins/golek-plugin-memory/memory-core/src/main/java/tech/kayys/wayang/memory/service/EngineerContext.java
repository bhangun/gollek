package tech.kayys.wayang.memory.service;


import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Engineered context ready for LLM consumption
 */
public class EngineerContext {

    private final String query;
    private final List<ContextSection> sections;
    private final int totalTokens;
    private final int maxTokens;

    public EngineerContext(
            String query,
            List<ContextSection> sections,
            int totalTokens,
            int maxTokens) {
        this.query = query;
        this.sections = sections;
        this.totalTokens = totalTokens;
        this.maxTokens = maxTokens;
    }

    /**
     * Get formatted prompt ready for LLM
     */
    public String toPrompt() {
        StringBuilder prompt = new StringBuilder();

        // Add sections in order
        for (ContextSection section : sections) {
            if (section.getType().startsWith("memory_")) {
                prompt.append("\n--- Relevant Context ---\n");
            } else if (section.getType().equals("conversation_history")) {
                prompt.append("\n--- Recent Conversation ---\n");
            } else if (section.getType().equals("task_instructions")) {
                prompt.append("\n--- Task ---\n");
            }

            prompt.append(section.getContent());
            prompt.append("\n");
        }

        // Add query at the end
        prompt.append("\n--- Current Query ---\n");
        prompt.append(query);

        return prompt.toString();
    }

    /**
     * Get structured representation for advanced LLMs
     */
    public Map<String, Object> toStructured() {
        Map<String, Object> structured = new HashMap<>();

        // Group sections by type
        for (ContextSection section : sections) {
            String type = section.getType();

            if (type.startsWith("memory_")) {
                List<String> memories = (List<String>) structured.computeIfAbsent(
                        "memories", k -> new ArrayList<String>());
                memories.add(section.getContent());
            } else {
                structured.put(type, section.getContent());
            }
        }

        structured.put("query", query);
        structured.put("metadata", Map.of(
                "totalTokens", totalTokens,
                "maxTokens", maxTokens,
                "sectionCount", sections.size()));

        return structured;
    }

    // Getters
    public String getQuery() {
        return query;
    }

    public List<ContextSection> getSections() {
        return sections;
    }

    public int getTotalTokens() {
        return totalTokens;
    }

    public int getMaxTokens() {
        return maxTokens;
    }

    public double getUtilization() {
        return (double) totalTokens / maxTokens;
    }
}