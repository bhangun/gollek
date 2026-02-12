/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.memory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.wayang.memory.impl.VectorAgentMemory;
import tech.kayys.wayang.memory.spi.MemoryEntry;
import io.smallrye.mutiny.Uni;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * Tests for {@link MemoryIntegrationPlugin}.
 */
@ExtendWith(MockitoExtension.class)
class MemoryIntegrationPluginTest {

    @Mock
    VectorAgentMemory memoryService;

    @Mock
    PluginContext pluginContext;

    @Mock
    ExecutionContext executionContext;

    @InjectMocks
    MemoryIntegrationPlugin plugin;

    @BeforeEach
    void setUp() {
        when(pluginContext.getConfig("enabled")).thenReturn(Optional.empty());
        when(pluginContext.getConfig("maxResults")).thenReturn(Optional.empty());
        when(pluginContext.getConfig("minScore")).thenReturn(Optional.empty());
    }

    @Test
    void initialize_loadsConfig() {
        when(pluginContext.getConfig("maxResults")).thenReturn(Optional.of("10"));
        plugin.initialize(pluginContext);

        Map<String, Object> config = plugin.currentConfig();
        assertEquals(10, config.get("maxResults"));
    }

    @Test
    void execute_noUserMessage_skipsRetrieval() {
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-1")
                .model("model")
                .messages(List.of(Message.system("Sys")))
                .build();
        when(executionContext.getVariable("request", InferenceRequest.class))
                .thenReturn(Optional.of(request));

        plugin.execute(executionContext, null);

        verifyNoInteractions(memoryService);
    }

    @Test
    void execute_performsRetrievalAndInjectsContext() {
        // Setup request
        InferenceRequest request = InferenceRequest.builder()
                .requestId("req-1")
                .model("model")
                .messages(List.of(Message.user("Hello Golek")))
                .build();
        when(executionContext.getVariable("request", InferenceRequest.class))
                .thenReturn(Optional.of(request));

        // Mock memory response - VectorAgentMemory.retrieve() returns
        // Uni<List<MemoryEntry>>
        MemoryEntry entry = new MemoryEntry("id-1", "Golek is a puppet", Instant.now(), Map.of());
        when(memoryService.retrieve(anyString(), anyString(), anyInt()))
                .thenReturn(Uni.createFrom().item(List.of(entry)));

        plugin.execute(executionContext, null);

        // Verify context injection
        verify(executionContext).putVariable(eq("retrievedMemories"), anyList());
        verify(executionContext).putVariable(eq("injectedContextMessages"), anyList());
    }
}
