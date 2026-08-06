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
import tech.kayys.gollek.spi.plugin.GollekPlugin;

import java.util.List;
import java.util.Map;

/**
 * Core SPI for visual, node-based workflow execution.
 *
 * <p>A {@code NodePlugin} represents a single computational unit in a directed
 * acyclic graph (DAG) that the Flutter visual editor renders and the Gollek
 * engine executes reactively. Each node declares typed I/O ports, a category,
 * and a reactive {@link #execute(NodeExecutionContext)} method.</p>
 *
 * <h2>Design Principles</h2>
 * <ul>
 *   <li><b>Reactive-first</b>: {@code execute()} returns {@code Uni<Void>},
 *       keeping the Vert.x event loop non-blocking.</li>
 *   <li><b>Type-safe wiring</b>: Ports carry {@link NodePort.DataType} so the
 *       Flutter UI and the graph executor can validate connections at design
 *       time and runtime.</li>
 *   <li><b>Wrapper-friendly</b>: Existing {@link tech.kayys.gollek.spi.inference.InferencePhasePlugin}
 *       and {@link tech.kayys.gollek.spi.plugin.StreamingPlugin} implementations
 *       can be wrapped into {@code NodePlugin} adapters without modification.</li>
 *   <li><b>Self-describing</b>: {@link #descriptor()} returns a complete
 *       {@link NodeDescriptor} that the Flutter client serializes for its palette.</li>
 * </ul>
 *
 * <h2>Example Implementation</h2>
 * <pre>{@code
 * public class LLMInferenceNode implements NodePlugin {
 *
 *     @Override
 *     public String id() {
 *         return "gollek/llm-inference";
 *     }
 *
 *     @Override
 *     public NodeDescriptor descriptor() {
 *         return NodeDescriptor.builder("gollek/llm-inference", "LLM Inference", NodeCategory.INFERENCE)
 *             .description("Sends a prompt to an LLM provider and returns the response")
 *             .inputs(
 *                 NodePort.requiredInput("prompt", NodePort.DataType.STRING),
 *                 NodePort.builder("messages", NodePort.Direction.INPUT, NodePort.DataType.MESSAGE_LIST)
 *                     .displayName("Chat History")
 *                     .description("Optional conversation history for multi-turn")
 *                     .build()
 *             )
 *             .outputs(
 *                 NodePort.output("response", NodePort.DataType.INFERENCE_RESPONSE),
 *                 NodePort.output("text", NodePort.DataType.STRING)
 *             )
 *             .defaultConfig(Map.of("model", "default", "temperature", 0.7))
 *             .tags("llm", "inference", "chat", "completion")
 *             .build();
 *     }
 *
 *     @Override
 *     public Uni<Void> execute(NodeExecutionContext ctx) {
 *         String prompt = ctx.requireInput("prompt", String.class);
 *         // ... build request, call provider ...
 *         return provider.infer(request)
 *             .invoke(response -> {
 *                 ctx.setOutput("response", response);
 *                 ctx.setOutput("text", response.getText());
 *             })
 *             .replaceWithVoid();
 *     }
 * }
 * }</pre>
 *
 * @see NodeDescriptor
 * @see NodePort
 * @see NodeExecutionContext
 * @since 3.0.0
 */
public interface NodePlugin extends GollekPlugin {

    // ──────────────────────────────────────────────
    // Visual Metadata
    // ──────────────────────────────────────────────

    /**
     * Return the complete visual and behavioral descriptor for this node type.
     *
     * <p>The returned descriptor is serialized to JSON and sent to the Flutter
     * visual editor to render the node card, populate the palette, and validate
     * connections.</p>
     *
     * @return immutable node descriptor
     */
    NodeDescriptor descriptor();

    /**
     * Convenience: get the node's category from its descriptor.
     */
    default NodeCategory category() {
        return descriptor().category();
    }

    /**
     * Convenience: get the node's input ports from its descriptor.
     */
    default List<NodePort> inputs() {
        return descriptor().inputs();
    }

    /**
     * Convenience: get the node's output ports from its descriptor.
     */
    default List<NodePort> outputs() {
        return descriptor().outputs();
    }

    // ──────────────────────────────────────────────
    // Reactive Execution
    // ──────────────────────────────────────────────

    /**
     * Execute this node's logic within the given context.
     *
     * <p>The graph executor calls this method after all required input ports
     * have been resolved. The implementation reads inputs from the context,
     * performs its computation (non-blocking), and writes outputs back.</p>
     *
     * <p>Implementations <b>MUST NOT</b> block the calling thread. Use
     * {@code Uni.createFrom().item(...)} for synchronous transformations or
     * delegate to a worker pool via {@code emitOn(Infrastructure.getDefaultWorkerPool())}
     * for CPU-bound work.</p>
     *
     * @param context the execution context with resolved inputs and output slots
     * @return a Uni that completes when the node finishes
     */
    Uni<Void> execute(NodeExecutionContext context);

    // ──────────────────────────────────────────────
    // Validation
    // ──────────────────────────────────────────────

    /**
     * Validate a node instance configuration at design time.
     *
     * <p>Called by the Flutter UI when a user edits a node's configuration panel.
     * Returns a list of validation error messages (empty = valid).</p>
     *
     * @param config the proposed configuration map
     * @return list of validation error messages (empty if valid)
     */
    default List<String> validateConfig(Map<String, Object> config) {
        return List.of();
    }

    /**
     * Check if this node can execute given the current context state.
     *
     * <p>The graph executor calls this before {@link #execute(NodeExecutionContext)}
     * to allow conditional skipping (e.g., a node that only runs when a feature
     * flag is set in graph state).</p>
     *
     * @param context the execution context
     * @return true if this node should execute, false to skip
     */
    default boolean shouldExecute(NodeExecutionContext context) {
        return true;
    }

    // ──────────────────────────────────────────────
    // Error Handling
    // ──────────────────────────────────────────────

    /**
     * Maximum retry attempts for this node if execution fails.
     * Default: 0 (no retries).
     *
     * @return max retry count
     */
    default int maxRetries() {
        return 0;
    }

    /**
     * Whether this node's failure should halt the entire graph execution.
     * If false, the graph executor will mark this node as failed but continue
     * executing other branches.
     *
     * @return true if failure is critical
     */
    default boolean failureIsCritical() {
        return true;
    }

    /**
     * Optional fallback logic invoked when execution fails after all retries.
     * Default: propagates the error.
     *
     * @param context the execution context
     * @param error   the error that caused the failure
     * @return a Uni that completes with fallback logic, or fails to propagate
     */
    default Uni<Void> onFailure(NodeExecutionContext context, Throwable error) {
        return Uni.createFrom().failure(error);
    }
}
