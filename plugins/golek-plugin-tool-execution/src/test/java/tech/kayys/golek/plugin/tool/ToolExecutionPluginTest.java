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

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.tool.ToolCall;
import tech.kayys.wayang.tool.impl.DefaultToolExecutor;
import tech.kayys.wayang.tool.dto.ToolExecutionResult;
import tech.kayys.wayang.tool.dto.ToolExecuteRequest;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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

    @InjectMocks
    ToolExecutionPlugin plugin;

    @BeforeEach
    void setUp() {
        // Default behavior
        when(pluginContext.getConfig("enabled")).thenReturn(Optional.empty());
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
    void execute_runsDetectedTools() {
        // Mock a tool call
        ToolCall call = ToolCall.builder()
                .name("search")
                .argument("query", "golek")
                .build();

        when(executionContext.getVariable("detectedToolCalls", List.class))
                .thenReturn(Optional.of(List.of(call)));

        // Mock executor result
        ToolExecutionResult result = new ToolExecutionResult();
        result.setSuccess(true);
        result.setOutput("Golek is awesome");
        when(toolExecutor.execute(any(ToolExecuteRequest.class))).thenReturn(result);

        plugin.execute(executionContext, null);

        // Verify execution
        verify(toolExecutor, times(1)).execute(any(ToolExecuteRequest.class));
        
        // Verify results stored in context
        verify(executionContext).putVariable(eq("toolResults"), anyList());
        verify(executionContext).putVariable("hasToolResults", true);
        verify(executionContext).putVariable("reasoningLoopContinue", true);
    }
}
