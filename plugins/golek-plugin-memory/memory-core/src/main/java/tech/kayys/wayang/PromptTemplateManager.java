package tech.kayys.gamelan.executor.memory;

import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages prompt templates for different use cases
 */
@ApplicationScoped
public class PromptTemplateManager {

    private static final Logger LOG = LoggerFactory.getLogger(PromptTemplateManager.class);

    private final Map<String, PromptTemplate> templates = new HashMap<>();

    public PromptTemplateManager() {
        initializeDefaultTemplates();
    }

    /**
     * Initialize default templates
     */
    private void initializeDefaultTemplates() {
        // Question answering template
        templates.put("qa", new PromptTemplate(
                "qa",
                """
                        You are a helpful AI assistant. Use the following context to answer the user's question.
                        If you cannot find the answer in the context, say so honestly.

                        Context:
                        {context}

                        Question: {query}

                        Answer:
                        """,
                List.of("context", "query")));

        // Task execution template
        templates.put("task", new PromptTemplate(
                "task",
                """
                        You are an AI agent capable of executing tasks. Based on the context and your capabilities,
                        execute the following task.

                        Available Context:
                        {context}

                        Task: {query}

                        Please provide your response in the following format:
                        1. Analysis: Brief analysis of the task
                        2. Action: The action you will take
                        3. Result: Expected or actual result
                        """,
                List.of("context", "query")));

        // Conversational template
        templates.put("chat", new PromptTemplate(
                "chat",
                """
                        You are a friendly AI assistant having a conversation with a user.
                        Use the conversation history and any relevant context to provide helpful responses.

                        {conversation_history}

                        Relevant Context:
                        {context}

                        User: {query}

                        Assistant:
                        """,
                List.of("conversation_history", "context", "query")));

        LOG.info("Initialized {} prompt templates", templates.size());
    }

    /**
     * Get template by name
     */
    public PromptTemplate getTemplate(String name) {
        return templates.get(name);
    }

    /**
     * Register custom template
     */
    public void registerTemplate(PromptTemplate template) {
        templates.put(template.getName(), template);
        LOG.info("Registered custom template: {}", template.getName());
    }

    /**
     * Apply template with variables
     */
    public String applyTemplate(String templateName, Map<String, String> variables) {
        PromptTemplate template = templates.get(templateName);
        if (template == null) {
            throw new IllegalArgumentException("Template not found: " + templateName);
        }

        return template.apply(variables);
    }
}