/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.tooling;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.*;

import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for {@link ToolCallingEngine} — system prompt generation and
 * tool-call detection across all supported model output formats.
 */
@QuarkusTest
class ToolCallingEngineTest {

    @Inject
    ToolCallingEngine toolEngine;

    private static List<ToolDefinition> sampleTools;

    @BeforeAll
    static void buildTools() {
        sampleTools = List.of(
                ToolDefinition.builder()
                        .name("get_weather")
                        .type(ToolDefinition.Type.FUNCTION)
                        .description("Get current weather for a city")
                        .parameters(Map.of("type", "object", "properties", Map.of(
                                "city", Map.of("type", "string"),
                                "units", Map.of("type", "string", "enum", List.of("celsius", "fahrenheit"))),
                                "required", List.of("city")))
                        .build(),
                ToolDefinition.builder()
                        .name("search_web")
                        .type(ToolDefinition.Type.FUNCTION)
                        .description("Search the web for information")
                        .parameters(Map.of("type", "object", "properties", Map.of(
                                "query", Map.of("type", "string")),
                                "required", List.of("query")))
                        .build());
    }

    // ── System prompt ─────────────────────────────────────────────────────────

    @Test
    @DisplayName("buildSystemPrompt: contains tool names")
    void testSystemPromptContainsToolNames() {
        String prompt = toolEngine.buildSystemPrompt(sampleTools);
        assertThat(prompt).contains("get_weather").contains("search_web");
    }

    @Test
    @DisplayName("buildSystemPrompt: contains tool descriptions")
    void testSystemPromptContainsDescriptions() {
        String prompt = toolEngine.buildSystemPrompt(sampleTools);
        assertThat(prompt).contains("Get current weather for a city");
    }

    @Test
    @DisplayName("buildSystemPrompt: contains JSON call format")
    void testSystemPromptContainsFormat() {
        String prompt = toolEngine.buildSystemPrompt(sampleTools);
        assertThat(prompt).contains("\"name\"").contains("\"arguments\"");
    }

    @Test
    @DisplayName("buildSystemPrompt: empty tools returns empty string")
    void testEmptyToolsEmptyPrompt() {
        assertThat(toolEngine.buildSystemPrompt(List.of())).isEmpty();
        assertThat(toolEngine.buildSystemPrompt(null)).isEmpty();
    }

    // ── Detection: no tool call ───────────────────────────────────────────────

    @Test
    @DisplayName("detect: plain text → noToolCall")
    void testPlainTextNoToolCall() {
        var r = toolEngine.detect("The weather in Paris is sunny and 22°C.", sampleTools);
        assertThat(r.hasToolCalls()).isFalse();
        assertThat(r.toolCalls()).isEmpty();
        assertThat(r.textContent()).contains("Paris");
    }

    @Test
    @DisplayName("detect: null output → noToolCall")
    void testNullOutputNoToolCall() {
        var r = toolEngine.detect(null, sampleTools);
        assertThat(r.hasToolCalls()).isFalse();
    }

    @Test
    @DisplayName("detect: unknown tool name → skipped")
    void testUnknownToolSkipped() {
        String output = """
                {"name": "fly_to_moon", "arguments": {"destination": "luna"}}
                """;
        var r = toolEngine.detect(output, sampleTools);
        // Unknown tool should be ignored → no tool calls
        assertThat(r.hasToolCalls()).isFalse();
    }

    // ── Detection: LLaMA-3.1 format ──────────────────────────────────────────

    @Test
    @DisplayName("detect: LLaMA-3.1 python_tag format")
    void testLlama31PythonTag() {
        String output = "<|python_tag|>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.hasToolCalls()).isTrue();
        assertThat(r.toolCalls()).hasSize(1);
        assertThat(r.toolCalls().get(0).getFunction().getName()).isEqualTo("get_weather");
        assertThat(r.toolCalls().get(0).getArguments()).containsKey("city");
        assertThat(r.toolCalls().get(0).getArguments().get("city")).isEqualTo("Paris");
        assertThat(r.finishReason()).isEqualTo(
                tech.kayys.gollek.spi.inference.InferenceResponse.FinishReason.TOOL_CALLS);
    }

    // ── Detection: Qwen2 / Hermes format ─────────────────────────────────────

    @Test
    @DisplayName("detect: tool_call XML tag format (Qwen2/Hermes)")
    void testToolCallTag() {
        String output = "I'll search for that.\n<tool_call>" +
                "{\"name\": \"search_web\", \"arguments\": {\"query\": \"quantum computing\"}}" +
                "</tool_call>";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.hasToolCalls()).isTrue();
        assertThat(r.toolCalls().get(0).getFunction().getName()).isEqualTo("search_web");
        assertThat(r.toolCalls().get(0).getArguments().get("query"))
                .isEqualTo("quantum computing");
        // Text before the tool call should be retained
        assertThat(r.textContent()).contains("I'll search for that");
    }

    // ── Detection: Mistral format ─────────────────────────────────────────────

    @Test
    @DisplayName("detect: Mistral [TOOL_CALLS] format")
    void testMistralToolCalls() {
        String output = "[TOOL_CALLS] " +
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\", \"units\": \"celsius\"}}]";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.hasToolCalls()).isTrue();
        assertThat(r.toolCalls()).hasSize(1);
        assertThat(r.toolCalls().get(0).getArguments().get("city")).isEqualTo("London");
    }

    // ── Detection: Generic JSON ───────────────────────────────────────────────

    @Test
    @DisplayName("detect: generic JSON with name field")
    void testGenericJson() {
        String output = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.hasToolCalls()).isTrue();
        assertThat(r.toolCalls().get(0).getArguments().get("city")).isEqualTo("Tokyo");
    }

    // ── Multiple tool calls ───────────────────────────────────────────────────

    @Test
    @DisplayName("detect: multiple tool_call tags → parallel tool calls")
    void testMultipleToolCalls() {
        String output = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
                + "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"Eiffel Tower\"}}</tool_call>";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.hasToolCalls()).isTrue();
        assertThat(r.toolCalls()).hasSize(2);
        assertThat(r.toolCalls()).extracting("name")
                .containsExactlyInAnyOrder("get_weather", "search_web");
    }

    // ── ToolCall IDs ─────────────────────────────────────────────────────────

    @Test
    @DisplayName("detect: each tool call gets a unique call_* ID")
    void testToolCallUniqueIds() {
        String output = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"
                + "<tool_call>{\"name\":\"search_web\",\"arguments\":{\"query\":\"test\"}}</tool_call>";
        var r = toolEngine.detect(output, sampleTools);
        assertThat(r.toolCalls()).hasSize(2);
        String id0 = r.toolCalls().get(0).getId();
        String id1 = r.toolCalls().get(1).getId();
        assertThat(id0).startsWith("call_");
        assertThat(id1).startsWith("call_");
        assertThat(id0).isNotEqualTo(id1);
    }

    // ── LLaMA-3.1 system prompt ───────────────────────────────────────────────

    @Test
    @DisplayName("buildLlama31SystemPrompt: produces JSON schema format")
    void testLlama31SystemPrompt() throws Exception {
        String prompt = toolEngine.buildLlama31SystemPrompt(sampleTools);
        assertThat(prompt).contains("ipython");
        assertThat(prompt).contains("get_weather");
        // Should be valid JSON-containing string
        assertThat(prompt).contains("[");
    }
}
