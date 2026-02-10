/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.prompt;

import java.util.Map;

/**
 * Prompt template with variable substitution support.
 * <p>
 * Templates use {@code {{variable}}} syntax for variable placeholders.
 *
 * @param name      template name (e.g., "agent-default", "tool-prompt", "rag-prompt")
 * @param role      message role (system, user, assistant, tool)
 * @param template  template text with {{variable}} placeholders
 * @param metadata  additional template metadata
 */
public record PromptTemplate(
        String name,
        String role,
        String template,
        Map<String, String> metadata) {

    /**
     * Render the template by substituting variables.
     *
     * @param variables variable name-value pairs
     * @return rendered template
     */
    public String render(Map<String, String> variables) {
        String result = template;
        for (Map.Entry<String, String> entry : variables.entrySet()) {
            result = result.replace("{{" + entry.getKey() + "}}", entry.getValue());
        }
        return result;
    }

    /**
     * Create a system prompt template.
     */
    public static PromptTemplate system(String name, String template) {
        return new PromptTemplate(name, "system", template, Map.of());
    }

    /**
     * Create a user prompt template.
     */
    public static PromptTemplate user(String name, String template) {
        return new PromptTemplate(name, "user", template, Map.of());
    }

    /**
     * Create a tool result template.
     */
    public static PromptTemplate tool(String name, String template) {
        return new PromptTemplate(name, "tool", template, Map.of());
    }
}
