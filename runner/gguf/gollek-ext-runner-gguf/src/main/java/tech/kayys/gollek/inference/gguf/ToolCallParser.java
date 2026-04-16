package tech.kayys.gollek.inference.gguf;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses tool/function calls from model output.
 * Supports multiple formats used by different models.
 */
public class ToolCallParser {

    private static final Logger log = Logger.getLogger(ToolCallParser.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    // Common tool call patterns
    private static final Pattern QWEN_PATTERN = Pattern.compile(
            "<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>",
            Pattern.DOTALL);

    private static final Pattern FUNCTION_CALL_PATTERN = Pattern.compile(
            "<function_call>\\s*(\\{.*?\\})\\s*</function_call>",
            Pattern.DOTALL);

    private static final Pattern ACTION_PATTERN = Pattern.compile(
            "Action:\\s*(\\w+)\\s*Action Input:\\s*(\\{.*?\\})",
            Pattern.DOTALL);

    // JSON-only pattern for models that output raw JSON
    private static final Pattern JSON_TOOL_PATTERN = Pattern.compile(
            "\\{\\s*\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"arguments\"\\s*:\\s*(\\{.*?\\})\\s*\\}",
            Pattern.DOTALL);

    /**
     * Parse tool calls from model output.
     * Returns empty list if no tool calls found.
     */
    public List<ToolCall> parse(String output) {
        if (output == null || output.isBlank()) {
            return Collections.emptyList();
        }

        List<ToolCall> results = new ArrayList<>();

        // Try Qwen format first
        results.addAll(parseWithPattern(output, QWEN_PATTERN, true));
        if (!results.isEmpty())
            return results;

        // Try generic function_call format
        results.addAll(parseWithPattern(output, FUNCTION_CALL_PATTERN, true));
        if (!results.isEmpty())
            return results;

        // Try ReAct-style Action format
        results.addAll(parseReActFormat(output));
        if (!results.isEmpty())
            return results;

        // Try raw JSON format
        results.addAll(parseWithPattern(output, JSON_TOOL_PATTERN, false));

        return results;
    }

    /**
     * Check if output contains tool calls without fully parsing.
     */
    public boolean hasToolCalls(String output) {
        if (output == null)
            return false;
        return output.contains("<tool_call>") ||
                output.contains("<function_call>") ||
                output.contains("Action:") ||
                (output.contains("\"name\"") && output.contains("\"arguments\""));
    }

    /**
     * Extract the content before any tool calls (the "thinking" part).
     */
    public String extractContentBeforeToolCalls(String output) {
        if (output == null)
            return "";

        int toolStart = output.indexOf("<tool_call>");
        if (toolStart == -1)
            toolStart = output.indexOf("<function_call>");
        if (toolStart == -1)
            toolStart = output.indexOf("Action:");

        if (toolStart > 0) {
            return output.substring(0, toolStart).trim();
        }
        return output;
    }

    private List<ToolCall> parseWithPattern(String output, Pattern pattern, boolean isWrappedJson) {
        List<ToolCall> results = new ArrayList<>();
        Matcher matcher = pattern.matcher(output);

        while (matcher.find()) {
            try {
                if (isWrappedJson) {
                    // The entire captured group is JSON
                    String json = matcher.group(1).trim();
                    Map<String, Object> parsed = mapper.readValue(json, new TypeReference<>() {
                    });

                    String name = (String) parsed.get("name");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> arguments = (Map<String, Object>) parsed.getOrDefault("arguments",
                            parsed.getOrDefault("parameters", Collections.emptyMap()));

                    if (name != null && !name.isBlank()) {
                        results.add(new ToolCall(name, arguments));
                    }
                } else {
                    // Name and arguments captured separately
                    String name = matcher.group(1).trim();
                    String argsJson = matcher.group(2).trim();
                    Map<String, Object> arguments = mapper.readValue(argsJson, new TypeReference<>() {
                    });
                    results.add(new ToolCall(name, arguments));
                }
            } catch (Exception e) {
                log.debugf("Failed to parse tool call: %s", e.getMessage());
            }
        }
        return results;
    }

    private List<ToolCall> parseReActFormat(String output) {
        List<ToolCall> results = new ArrayList<>();
        Matcher matcher = ACTION_PATTERN.matcher(output);

        while (matcher.find()) {
            try {
                String name = matcher.group(1).trim();
                String argsJson = matcher.group(2).trim();
                Map<String, Object> arguments = mapper.readValue(argsJson, new TypeReference<>() {
                });
                results.add(new ToolCall(name, arguments));
            } catch (Exception e) {
                log.debugf("Failed to parse ReAct action: %s", e.getMessage());
            }
        }
        return results;
    }

    /**
     * Represents a parsed tool call.
     */
    public record ToolCall(String name, Map<String, Object> arguments) {
        public ToolCall {
            arguments = arguments != null ? Map.copyOf(arguments) : Collections.emptyMap();
        }
    }
}
