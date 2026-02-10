/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.reasoning;

import tech.kayys.golek.spi.tool.ToolCall;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses LLM output to detect tool calls, final answers, and malformed output.
 * <p>
 * Supports multiple output formats:
 * <ul>
 *   <li>JSON tool call format: {@code {"tool_call": {"name": "...", "arguments": {...}}}}</li>
 *   <li>XML-style: {@code <tool_call>...</tool_call>}</li>
 * </ul>
 */
public class OutputParser {

    private static final Pattern JSON_TOOL_CALL_PATTERN = Pattern.compile(
            "\\{\\s*\"(?:tool_call|function_call)\"\\s*:\\s*\\{\\s*\"name\"\\s*:\\s*\"([^\"]+)\"",
            Pattern.DOTALL);

    private static final Pattern XML_TOOL_CALL_PATTERN = Pattern.compile(
            "<tool_call>\\s*<name>([^<]+)</name>\\s*<arguments>([^<]*)</arguments>\\s*</tool_call>",
            Pattern.DOTALL);

    private static final Pattern MALFORMED_JSON_PATTERN = Pattern.compile(
            "\\{[^}]*\"(?:tool_call|function_call)\"[^}]*$", Pattern.DOTALL);

    /**
     * Parse tool calls from LLM output.
     *
     * @param output the raw LLM output text
     * @return list of detected tool calls, empty if none
     */
    public List<ToolCall> parseToolCalls(String output) {
        if (output == null || output.isBlank()) {
            return Collections.emptyList();
        }

        List<ToolCall> results = new ArrayList<>();

        // Try JSON format
        Matcher jsonMatcher = JSON_TOOL_CALL_PATTERN.matcher(output);
        while (jsonMatcher.find()) {
            String name = jsonMatcher.group(1);
            results.add(ToolCall.builder()
                    .id("call_" + name + "_" + System.nanoTime())
                    .name(name)
                    .build());
        }

        if (!results.isEmpty()) {
            return results;
        }

        // Try XML format
        Matcher xmlMatcher = XML_TOOL_CALL_PATTERN.matcher(output);
        while (xmlMatcher.find()) {
            String name = xmlMatcher.group(1).trim();
            results.add(ToolCall.builder()
                    .id("call_" + name + "_" + System.nanoTime())
                    .name(name)
                    .build());
        }

        return results;
    }

    /**
     * Check if the output appears to be malformed (e.g., truncated JSON, broken tool call).
     */
    public boolean isMalformed(String output) {
        if (output == null) return false;

        // Check for truncated JSON
        if (MALFORMED_JSON_PATTERN.matcher(output).find()) {
            return true;
        }

        // Check for unbalanced braces
        long openBraces = output.chars().filter(c -> c == '{').count();
        long closeBraces = output.chars().filter(c -> c == '}').count();
        if (openBraces > 0 && openBraces != closeBraces) {
            return true;
        }

        return false;
    }

    /**
     * Extract the final answer, stripping any reasoning traces or intermediate content.
     */
    public String extractFinalAnswer(String output) {
        if (output == null) return "";

        // Strip common reasoning trace markers
        String cleaned = output
                .replaceAll("(?s)<think>.*?</think>", "")
                .replaceAll("(?s)<reasoning>.*?</reasoning>", "")
                .replaceAll("(?s)\\[THINKING\\].*?\\[/THINKING\\]", "")
                .trim();

        return cleaned.isEmpty() ? output.trim() : cleaned;
    }
}
