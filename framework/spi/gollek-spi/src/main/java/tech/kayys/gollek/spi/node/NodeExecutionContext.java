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

package tech.kayys.gollek.spi.node;

import io.smallrye.mutiny.Uni;

import java.util.Map;
import java.util.Optional;

/**
 * Reactive execution context for a single node invocation within a graph.
 *
 * <p>When the graph executor fires a node, it creates a {@code NodeExecutionContext}
 * populated with the resolved input values (from upstream wires) and graph-level
 * shared state. The node writes its results to the output map, which the executor
 * then routes to downstream nodes.</p>
 *
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>Graph executor resolves upstream outputs → populates {@link #inputs()}</li>
 *   <li>Node's {@link NodePlugin#execute(NodeExecutionContext)} is called</li>
 *   <li>Node reads inputs, performs work, writes to {@link #setOutput(String, Object)}</li>
 *   <li>Graph executor reads outputs → feeds to downstream nodes</li>
 * </ol>
 *
 * @since 3.0.0
 */
public interface NodeExecutionContext {

    // ──────────────────────────────────────────────
    // Input access
    // ──────────────────────────────────────────────

    /**
     * Get all resolved input values keyed by port name.
     *
     * @return unmodifiable map of input port name → value
     */
    Map<String, Object> inputs();

    /**
     * Get a single input value by port name, cast to the expected type.
     *
     * @param portName the input port name
     * @param type     expected Java type
     * @param <T>      type parameter
     * @return the input value, or empty if the port is unconnected and has no default
     */
    <T> Optional<T> getInput(String portName, Class<T> type);

    /**
     * Get a required input value. Throws if missing.
     *
     * @param portName the input port name
     * @param type     expected Java type
     * @param <T>      type parameter
     * @return the input value
     * @throws NodeExecutionException if the required input is missing
     */
    <T> T requireInput(String portName, Class<T> type) throws NodeExecutionException;

    // ──────────────────────────────────────────────
    // Output writing
    // ──────────────────────────────────────────────

    /**
     * Set an output value for routing to downstream nodes.
     *
     * @param portName the output port name
     * @param value    the value to emit
     */
    void setOutput(String portName, Object value);

    /**
     * Get all outputs written so far.
     *
     * @return unmodifiable map of output port name → value
     */
    Map<String, Object> outputs();

    // ──────────────────────────────────────────────
    // Graph-level shared state
    // ──────────────────────────────────────────────

    /**
     * Access graph-scoped state that persists across all node executions
     * within a single graph run (e.g., conversation ID, accumulated context).
     *
     * @return mutable map of graph state
     */
    Map<String, Object> graphState();

    /**
     * Get a typed value from graph state.
     */
    <T> Optional<T> getGraphState(String key, Class<T> type);

    /**
     * Put a value into graph state.
     */
    void putGraphState(String key, Object value);

    // ──────────────────────────────────────────────
    // Node instance configuration
    // ──────────────────────────────────────────────

    /**
     * Get the per-instance configuration for this node.
     * This is the configuration set by the user in the Flutter UI
     * when they place and configure this specific node instance.
     *
     * @return unmodifiable configuration map
     */
    Map<String, Object> nodeConfig();

    /**
     * Get a typed configuration value.
     */
    <T> Optional<T> getConfig(String key, Class<T> type);

    // ──────────────────────────────────────────────
    // Execution metadata
    // ──────────────────────────────────────────────

    /**
     * Unique execution ID for this graph run.
     */
    String executionId();

    /**
     * The unique instance ID of this node within the graph.
     */
    String nodeInstanceId();

    /**
     * Current retry attempt (0-based). Used by the executor for retry logic.
     */
    int attempt();

    // ──────────────────────────────────────────────
    // Error handling
    // ──────────────────────────────────────────────

    /**
     * Signal that this node execution has failed.
     * The graph executor will use this to decide on retry or error routing.
     */
    void fail(Throwable error);

    /**
     * Check if a failure has been signaled.
     */
    boolean hasFailed();

    /**
     * Get the failure if present.
     */
    Optional<Throwable> failure();

    // ──────────────────────────────────────────────
    // Logging / Observability
    // ──────────────────────────────────────────────

    /**
     * Log a message scoped to this node execution.
     * These logs are surfaced in the Flutter UI's execution trace viewer.
     *
     * @param level   log level (DEBUG, INFO, WARN, ERROR)
     * @param message log message
     */
    void log(String level, String message);

    /**
     * Emit a progress update (0.0 to 1.0) for long-running nodes.
     * The Flutter UI can display this as a progress indicator on the node.
     *
     * @param progress value between 0.0 and 1.0
     */
    void reportProgress(double progress);

    /**
     * Exception thrown when a required input is missing or type-cast fails.
     */
    class NodeExecutionException extends RuntimeException {
        public NodeExecutionException(String message) {
            super(message);
        }

        public NodeExecutionException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
