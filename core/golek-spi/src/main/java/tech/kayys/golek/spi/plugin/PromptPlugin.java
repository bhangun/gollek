/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */

package tech.kayys.golek.spi.plugin;

import tech.kayys.golek.spi.Message;

import java.util.List;
import java.util.Map;

/**
 * SPI for plugins that handle prompt construction and context window
 * management.
 * <p>
 * Responsible for:
 * <ul>
 * <li>System / user / assistant / tool role formatting</li>
 * <li>Context window management (truncate, summarize, chunk)</li>
 * <li>Prompt templating (agent prompt, tool prompt, RAG prompt)</li>
 * <li>Multi-message conversation history management</li>
 * <li>Instruction injection (policies, schemas, tools)</li>
 * </ul>
 * <p>
 * llama.cpp & LiteRT only see a flat {@code String prompt} — this plugin
 * is responsible for building that string from structured inputs.
 */
public interface PromptPlugin extends GolekPlugin {

    /**
     * Build a prompt string from structured messages and context.
     *
     * @param messages the conversation messages (system, user, assistant, tool)
     * @param context  additional context variables (e.g., RAG documents, tool
     *                 schemas)
     * @return the fully constructed prompt string
     */
    String buildPrompt(List<Message> messages, Map<String, Object> context);

    /**
     * Manage context window by truncating or summarizing history to fit within
     * the model's maximum token limit.
     *
     * @param history   full conversation history
     * @param maxTokens maximum tokens allowed in the context window
     * @return trimmed history that fits within the token budget
     */
    List<Message> manageContext(List<Message> history, int maxTokens);

    /**
     * Inject system-level instructions into the message list.
     * This includes policies, tool definitions, output format schemas, etc.
     *
     * @param messages     existing messages
     * @param instructions system instructions to inject
     * @return messages with injected instructions
     */
    List<Message> injectInstructions(List<Message> messages, List<String> instructions);

    /**
     * Estimate token count for a given text.
     * Used for context window budgeting.
     *
     * @param text the text to estimate tokens for
     * @return estimated token count
     */
    default int estimateTokens(String text) {
        // Rough heuristic: ~4 chars per token for English
        return text == null ? 0 : (text.length() + 3) / 4;
    }
}
