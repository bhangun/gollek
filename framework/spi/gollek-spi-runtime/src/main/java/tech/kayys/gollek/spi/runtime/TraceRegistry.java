/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Registry for collecting and managing inference traces across the distributed fabric.
 * Enables post-mortem analysis and deterministic replay of inference sessions.
 */
@ApplicationScoped
public class TraceRegistry {

    private final Map<String, List<InferenceTrace>> sessionTraces = new ConcurrentHashMap<>();

    /**
     * Records a new trace for an inference session.
     */
    public void record(InferenceTrace trace) {
        sessionTraces.computeIfAbsent(trace.requestId(), k -> new CopyOnWriteArrayList<>())
                    .add(trace);
    }

    /**
     * Retrieves the complete trace of an inference session.
     */
    public List<InferenceTrace> getSessionTrace(String requestId) {
        return sessionTraces.getOrDefault(requestId, List.of());
    }

    /**
     * Clears traces for a finished session to free memory.
     */
    public void clear(String requestId) {
        sessionTraces.remove(requestId);
    }

    /**
     * Placeholder for deterministic replay logic.
     * In a production system, this would orchestrate the re-execution
     * of the exact node/kernel path captured in the trace.
     */
    public void replay(String requestId) {
        List<InferenceTrace> trace = getSessionTrace(requestId);
        if (trace.isEmpty()) {
            throw new IllegalArgumentException("No trace found for request: " + requestId);
        }
        // Replay orchestration logic goes here
    }
}
