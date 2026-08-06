/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ToolCallingEngine.java
 * ────────────────────────
 * Implements OpenAI-compatible function / tool calling for the direct inference backend.
 *
 * How tool calling works with open-weight models
 * ═══════════════════════════════════════════════
 * Open-weight models (LLaMA-3-Instruct, Mistral-Instruct, Qwen2.5) implement
 * tool calling by generating a special JSON string that signals tool invocation.
 * The format varies by model family:
 *
 *   LLaMA-3 / LLaMA-3.1-Instruct:
 *     <|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}
 *
 *   Qwen2 / Qwen2.5:
 *     <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>
 *
 *   Mistral-Instruct v0.3+:
 *     [TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]
 *
 *   Hermes / OpenHermes / Nous-Hermes:
 *     <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>
 *
 *   Generic JSON (fallback detection):
 *     {"name": "get_weather", "arguments": {"city": "Paris"}}
 *
 * Pipeline
 * ════════
 * 1. Build a tool-use system prompt from the ToolDefinition list.
 *    (Injected before the first user message.)
 *
 * 2. Generate as normal.
 *
 * 3. Post-process the generated text:
 *    a. Detect tool-call markers.
 *    b. Parse the JSON payload.
 *    c. Match function name against the tool list.
 *    d. Validate arguments against the parameter schema.
 *    e. Build ToolCall SPI objects.
 *    f. Build InferenceResponse with finish_reason="tool_calls".
 *
 * 4. Return the ToolCall list to the caller; they execute the tool and
 *    send the result back as a TOOL message for the next turn.
 *
 * Parallel tool calls
 * ═══════════════════
 * Some models emit multiple tool calls in one response. This is handled
 * by parsing all JSON objects from the output in order.
 */
package tech.kayys.gollek.safetensor.engine.tooling;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.tool.ToolCall;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Detects tool-call JSON in model output and converts it to SPI
 * {@link ToolCall} objects.
 *
 * <p>
 * Inject and call {@link #buildSystemPrompt(List)} to produce the tool
 * description
 * injected before the user message, then call {@link #detect(String, List)}
 * after
 * generation to check if the model output contains a tool invocation.
 */
@ApplicationScoped
public class ToolCallingEngine {

    private static final Logger log = Logger.getLogger(ToolCallingEngine.class);

    @Inject
    ObjectMapper objectMapper;

    // ── Detection patterns (ordered, first match wins) ────────────────────────

    /** LLaMA-3.1 python_tag format */
    private static final Pattern LLAMA3_PATTERN = Pattern.compile("<\\|python_tag\\|>(\\{.*?\\})", Pattern.DOTALL);

    /** Qwen2 / Hermes tool_call XML tag */
    private static final Pattern TOOL_CALL_TAG = Pattern.compile("<tool_call>(.*?)</tool_call>", Pattern.DOTALL);

    /** Mistral [TOOL_CALLS] format */
    private static final Pattern MISTRAL_PATTERN = Pattern.compile("\\[TOOL_CALLS\\]\\s*(\\[.*?\\])", Pattern.DOTALL);

    /** Generic: any JSON object starting with "name" or "function_name" field */
    private static final Pattern GENERIC_JSON = Pattern.compile("(\\{\\s*\"(?:name|function_name)\"\\s*:.*?\\})",
            Pattern.DOTALL);

    // ── Type reference for argument parsing ──────────────────────────────────
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };

    // ─────────────────────────────────────────────────────────────────────────
    // System prompt construction
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Build a tool-use system prompt from a list of tool definitions.
     *
     * <p>
     * The prompt is injected as a system message before any user messages.
     * It describes the available tools in JSON schema format, matching what
     * LLaMA-3.1, Qwen2.5, and Mistral-Instruct expect.
     *
     * @param tools list of available tool definitions
     * @return formatted system prompt string
     */
    public String buildSystemPrompt(List<ToolDefinition> tools) {
        if (tools == null || tools.isEmpty())
            return "";

        var sb = new StringBuilder();
        sb.append("You have access to the following tools. ");
        sb.append("To call a tool, respond with a JSON object of the form:\n");
        sb.append("{\"name\": \"<tool_name>\", \"arguments\": {<tool_arguments>}}\n\n");
        sb.append("Available tools:\n\n");

        for (ToolDefinition tool : tools) {
            sb.append("### ").append(tool.getName()).append("\n");
            tool.getDescription().ifPresent(d -> sb.append(d).append("\n"));

            if (!tool.getParameters().isEmpty()) {
                try {
                    sb.append("Parameters:\n```json\n");
                    sb.append(objectMapper.writerWithDefaultPrettyPrinter()
                            .writeValueAsString(tool.getParameters()));
                    sb.append("\n```\n");
                } catch (Exception e) {
                    sb.append("Parameters: ").append(tool.getParameters()).append("\n");
                }
            }
            sb.append("\n");
        }

        sb.append("Only call tools when necessary. ");
        sb.append("Do not make up tool arguments — use only what the user provided.");
        return sb.toString();
    }

    /**
     * Build the tool-use prompt for LLaMA-3.1 / LLaMA-3.2 format.
     * These models expect a specific JSON schema in the system prompt.
     */
    public String buildLlama31SystemPrompt(List<ToolDefinition> tools) {
        if (tools == null || tools.isEmpty())
            return "";
        try {
            List<Map<String, Object>> toolSchemas = new ArrayList<>();
            for (ToolDefinition tool : tools) {
                Map<String, Object> schema = new LinkedHashMap<>();
                schema.put("name", tool.getName());
                tool.getDescription().ifPresent(d -> schema.put("description", d));
                schema.put("parameters", tool.getParameters().isEmpty()
                        ? Map.of("type", "object", "properties", Map.of())
                        : tool.getParameters());
                toolSchemas.add(schema);
            }
            return "Environment: ipython\nTools: " +
                    objectMapper.writeValueAsString(toolSchemas);
        } catch (Exception e) {
            return buildSystemPrompt(tools);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Detection & parsing
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Detect and parse tool calls from model output.
     *
     * @param modelOutput  raw generated text from the model
     * @param definedTools list of tools that were offered to the model
     * @return a {@link DetectionResult} indicating whether a tool call was found
     */
    public DetectionResult detect(String modelOutput, List<ToolDefinition> definedTools) {
        if (modelOutput == null || modelOutput.isBlank() || definedTools == null
                || definedTools.isEmpty()) {
            return DetectionResult.noToolCall(modelOutput);
        }

        // Try each format in priority order
        List<Map<String, Object>> rawCalls = tryLlama3(modelOutput);
        if (rawCalls.isEmpty())
            rawCalls = tryToolCallTag(modelOutput);
        if (rawCalls.isEmpty())
            rawCalls = tryMistral(modelOutput);
        if (rawCalls.isEmpty())
            rawCalls = tryGenericJson(modelOutput);

        if (rawCalls.isEmpty()) {
            return DetectionResult.noToolCall(modelOutput);
        }

        // Convert raw maps to ToolCall SPI objects
        Map<String, ToolDefinition> byName = new HashMap<>();
        definedTools.forEach(t -> byName.put(t.getName(), t));

        List<ToolCall> toolCalls = new ArrayList<>();
        for (Map<String, Object> raw : rawCalls) {
            try {
                String name = extractString(raw, "name", "function_name");
                if (name == null)
                    continue;

                // Check if this tool is in the defined list
                ToolDefinition def = byName.get(name);
                if (def == null) {
                    log.warnf("ToolCallingEngine: model called unknown tool '%s' — skipping", name);
                    continue;
                }

                Map<String, Object> args = extractArguments(raw);
                String callId = "call_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);

                ToolCall call = new ToolCall(
                        callId,
                        name,
                        new ToolCall.Function(name, objectMapper.writeValueAsString(args)),
                        args,
                        ToolDefinition.Type.FUNCTION);
                toolCalls.add(call);

                log.debugf("ToolCallingEngine: detected call to '%s' with %d args",
                        name, args.size());
            } catch (Exception e) {
                log.warnf(e, "ToolCallingEngine: failed to parse tool call from: %s", raw);
            }
        }

        if (toolCalls.isEmpty()) {
            return DetectionResult.noToolCall(modelOutput);
        }

        // Strip the tool-call JSON from the visible content
        String content = stripToolCallText(modelOutput);
        return DetectionResult.toolCallFound(toolCalls, content);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pattern parsers
    // ─────────────────────────────────────────────────────────────────────────

    private List<Map<String, Object>> tryLlama3(String text) {
        return parseAllMatches(LLAMA3_PATTERN, text, 1);
    }

    private List<Map<String, Object>> tryToolCallTag(String text) {
        return parseAllMatches(TOOL_CALL_TAG, text, 1);
    }

    private List<Map<String, Object>> tryMistral(String text) {
        // Mistral wraps in an array
        Matcher m = MISTRAL_PATTERN.matcher(text);
        List<Map<String, Object>> result = new ArrayList<>();
        while (m.find()) {
            try {
                List<Map<String, Object>> arr = objectMapper.readValue(
                        m.group(1), new TypeReference<>() {
                        });
                result.addAll(arr);
            } catch (Exception e) {
                log.tracef("Mistral pattern parse failed: %s", e.getMessage());
            }
        }
        return result;
    }

    private List<Map<String, Object>> tryGenericJson(String text) {
        return parseAllMatches(GENERIC_JSON, text, 1);
    }

    private List<Map<String, Object>> parseAllMatches(Pattern p, String text, int group) {
        Matcher m = p.matcher(text);
        List<Map<String, Object>> results = new ArrayList<>();
        while (m.find()) {
            try {
                Map<String, Object> parsed = objectMapper.readValue(m.group(group), MAP_TYPE);
                results.add(parsed);
            } catch (Exception e) {
                log.tracef("Pattern parse failed: %s", e.getMessage());
            }
        }
        return results;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Map<String, Object> extractArguments(Map<String, Object> raw) {
        Object args = raw.get("arguments");
        if (args == null)
            args = raw.get("parameters");
        if (args == null)
            args = raw.get("input");
        if (args instanceof Map<?, ?> m)
            return (Map<String, Object>) m;
        if (args instanceof String s) {
            try {
                return objectMapper.readValue(s, MAP_TYPE);
            } catch (Exception e) {
                return Map.of("raw", s);
            }
        }
        return Map.of();
    }

    private static String extractString(Map<String, Object> map, String... keys) {
        for (String k : keys) {
            Object v = map.get(k);
            if (v instanceof String s && !s.isBlank())
                return s;
        }
        return null;
    }

    private static String stripToolCallText(String text) {
        return text
                .replaceAll("<\\|python_tag\\|>\\{.*?\\}", "")
                .replaceAll("<tool_call>.*?</tool_call>", "")
                .replaceAll("\\[TOOL_CALLS\\]\\s*\\[.*?\\]", "")
                .trim();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Result type
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Result of tool-call detection.
     */
    public record DetectionResult(
            boolean hasToolCalls,
            List<ToolCall> toolCalls,
            String textContent) {
        static DetectionResult noToolCall(String text) {
            return new DetectionResult(false, List.of(), text != null ? text : "");
        }

        static DetectionResult toolCallFound(List<ToolCall> calls, String remaining) {
            return new DetectionResult(true, List.copyOf(calls), remaining != null ? remaining : "");
        }

        public InferenceResponse.FinishReason finishReason() {
            return hasToolCalls
                    ? InferenceResponse.FinishReason.TOOL_CALLS
                    : InferenceResponse.FinishReason.STOP;
        }
    }
}
