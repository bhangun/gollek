/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.reasoning;

import java.util.Optional;

/**
 * Self-repair strategies for malformed LLM output.
 * <p>
 * Attempts to fix:
 * <ul>
 *   <li>Truncated JSON (close missing braces/brackets)</li>
 *   <li>Incomplete tool calls</li>
 *   <li>Missing quotes in JSON strings</li>
 * </ul>
 */
public class SelfRepairStrategy {

    /**
     * Attempt to repair malformed output.
     *
     * @param malformedOutput the malformed output text
     * @return repaired output if repair was possible, empty otherwise
     */
    public Optional<String> repair(String malformedOutput) {
        if (malformedOutput == null || malformedOutput.isBlank()) {
            return Optional.empty();
        }

        // Try closing unclosed JSON
        String repaired = tryCloseJson(malformedOutput);
        if (repaired != null) {
            return Optional.of(repaired);
        }

        // Try fixing unquoted strings in JSON
        repaired = tryFixQuotes(malformedOutput);
        if (repaired != null) {
            return Optional.of(repaired);
        }

        return Optional.empty();
    }

    /**
     * Try to close unclosed JSON by adding missing closing braces/brackets.
     */
    private String tryCloseJson(String text) {
        int openBraces = 0;
        int openBrackets = 0;
        boolean inString = false;
        char prev = 0;

        for (char c : text.toCharArray()) {
            if (c == '"' && prev != '\\') {
                inString = !inString;
            } else if (!inString) {
                switch (c) {
                    case '{' -> openBraces++;
                    case '}' -> openBraces--;
                    case '[' -> openBrackets++;
                    case ']' -> openBrackets--;
                }
            }
            prev = c;
        }

        if (openBraces == 0 && openBrackets == 0) {
            return null; // Nothing to fix
        }

        var sb = new StringBuilder(text);

        // Close unclosed strings
        if (inString) {
            sb.append('"');
        }

        // Close brackets first (inner), then braces (outer)
        for (int i = 0; i < openBrackets; i++) {
            sb.append(']');
        }
        for (int i = 0; i < openBraces; i++) {
            sb.append('}');
        }

        return sb.toString();
    }

    /**
     * Try to fix unquoted string values in JSON.
     * Simple heuristic: look for key: value patterns without quotes around values.
     */
    private String tryFixQuotes(String text) {
        // Simple: if text contains JSON-like patterns but is invalid,
        // try wrapping unquoted values. This is a placeholder for more
        // sophisticated JSON repair.
        if (text.contains("\"name\"") && text.contains("\"arguments\"")) {
            // Likely a tool call with minor JSON issues
            String fixed = text
                    .replaceAll(":\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*([,}])", ": \"$1\"$2")
                    .replaceAll(",\\s*}", "}");
            if (!fixed.equals(text)) {
                return fixed;
            }
        }
        return null;
    }
}
