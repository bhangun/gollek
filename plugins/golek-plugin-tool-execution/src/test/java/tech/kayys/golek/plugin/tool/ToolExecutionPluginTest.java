/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.tool;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.registry.ToolRegistry;
import tech.kayys.golek.spi.tool.Tool;
import tech.kayys.golek.spi.tool.ToolCall;
import tech.kayys.wayang.tool.impl.DefaultToolExecutor;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests for {@link ToolExecutionPlugin}.
 */
@ExtendWith(MockitoExtension.class)
class ToolExecutionPluginTest {

    @Mock
    DefaultToolExecutor toolExecutor;

    @Mock
    PluginContext pluginContext;

    @Mock
    ExecutionContext executionContext;

    @Mock
    ToolRegistry toolRegistry;

    @Mock
    Tool tool;

    @Mock
    ObjectMapper objectMapper;

    @InjectMocks
    ToolExecutionPlugin plugin;

    @BeforeEach
    void setUp() {
        // Default behavior
    }

    @Test
    void initialize_loadsConfig() {
        when(pluginContext.getConfig("enabled")).thenReturn(Optional.of("false"));
        plugin.initialize(pluginContext);

        // Should be disabled based on config
        assertFalse(plugin.shouldExecute(executionContext));
    }

    @Test
    void execute_noToolCalls_doesNothing() {
        when(executionContext.getVariable("detectedToolCalls", List.class))
                .thenReturn(Optional.empty());

        plugin.execute(executionContext, null);

        verifyNoInteractions(toolExecutor);
    }

    @Test
    void execute_runsDetectedTools() throws Exception {
        // Mock a tool call
        ToolCall call = ToolCall.builder()
                .id("call_123")
                .name("search")
                .argument("query", "golek")
                .build();

        when(executionContext.getVariable("detectedToolCalls", List.class))
                .thenReturn(Optional.of(List.of(call)));

        when(executionContext.getVariable("conversationHistory", List.class))
                .thenReturn(Optional.of(new java.util.ArrayList<>()));

        // Mock ToolRegistry to return a mock Tool
        when(toolRegistry.getTool("search")).thenReturn(io.smallrye.mutiny.Uni.createFrom().item(tool));
        when(tool.inputSchema()).thenReturn(Map.of()); // Empty schema for validation

        // Mock ObjectMapper
        when(objectMapper.readValue(anyString(), eq(Map.class))).thenReturn(Map.of("query", "golek"));

        // Mock executor result
        ToolExecutionResult result = ToolExecutionResult.success(
                "search",
                Map.of("result", "Golek is awesome"),
                100L);

        when(toolExecutor.execute(eq("search"), anyMap(), anyMap()))
                .thenReturn(io.smallrye.mutiny.Uni.createFrom().item(result));

        plugin.execute(executionContext, null);

        // Verify execution
        verify(toolExecutor, times(1)).execute(eq("search"), anyMap(), anyMap());

        // Verify results stored in context
        verify(executionContext).putVariable(eq("toolResults"), anyList());
        verify(executionContext).putVariable(eq("hasToolResults"), eq(true));
        verify(executionContext).putVariable(eq("reasoningLoopContinue"), eq(true));
    }
}
