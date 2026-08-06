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

package tech.kayys.gollek.plugin.core.node;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.infrastructure.Infrastructure;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.node.NodeCategory;
import tech.kayys.gollek.spi.node.NodeDescriptor;
import tech.kayys.gollek.spi.node.NodeExecutionContext;
import tech.kayys.gollek.spi.node.NodePlugin;
import tech.kayys.gollek.spi.node.NodePort;
import tech.kayys.gollek.spi.plugin.PluginContext;

import java.util.List;
import java.util.Map;

/**
 * Adapter that wraps an existing {@link InferencePhasePlugin} as a {@link NodePlugin}.
 *
 * <p>This enables all existing phase-based plugins to appear in the Flutter visual
 * editor without any code changes to the plugin itself. The adapter maps the
 * phase plugin's synchronous {@code execute(ExecutionContext, EngineContext)} call
 * into the reactive {@code Uni<Void>} pipeline, running it on a worker thread
 * to avoid blocking the event loop.</p>
 *
 * <h2>Port Mapping</h2>
 * <ul>
 *   <li><b>Input</b>: {@code execution_context} (type: JSON) — the serialized
 *       execution context from the upstream node or graph trigger</li>
 *   <li><b>Output</b>: {@code execution_context} (type: JSON) — the mutated
 *       execution context after this phase plugin has run</li>
 * </ul>
 *
 * @since 3.0.0
 */
public class InferencePhaseNodeAdapter implements NodePlugin {

    private final InferencePhasePlugin delegate;
    private final NodeDescriptor cachedDescriptor;

    /**
     * Create an adapter for the given phase plugin.
     *
     * @param delegate the existing inference phase plugin to wrap
     */
    public InferencePhaseNodeAdapter(InferencePhasePlugin delegate) {
        this.delegate = delegate;
        this.cachedDescriptor = buildDescriptor();
    }

    @Override
    public String id() {
        return delegate.id();
    }

    @Override
    public int order() {
        return delegate.order();
    }

    @Override
    public String version() {
        return delegate.version();
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        return delegate.initialize(context);
    }

    @Override
    public Uni<Void> start() {
        return delegate.start();
    }

    @Override
    public Uni<Void> stop() {
        return delegate.stop();
    }

    @Override
    public Uni<Void> shutdown() {
        return delegate.shutdown();
    }

    @Override
    public boolean isHealthy() {
        return delegate.isHealthy();
    }

    @Override
    public NodeDescriptor descriptor() {
        return cachedDescriptor;
    }

    @Override
    public Uni<Void> execute(NodeExecutionContext context) {
        return Uni.createFrom().voidItem()
                .emitOn(Infrastructure.getDefaultWorkerPool())
                .invoke(() -> {
                    ExecutionContext execCtx = context.requireInput(
                            "execution_context", ExecutionContext.class);
                    EngineContext engineCtx = context.getInput(
                            "engine_context", EngineContext.class).orElse(null);

                    try {
                        delegate.execute(execCtx, engineCtx);
                        context.setOutput("execution_context", execCtx);
                    } catch (Exception e) {
                        context.fail(e);
                    }
                });
    }

    @Override
    public boolean shouldExecute(NodeExecutionContext context) {
        return context.getInput("execution_context", ExecutionContext.class)
                .map(delegate::shouldExecute)
                .orElse(false);
    }

    @Override
    public int maxRetries() {
        return delegate.phase().isRetryable() ? 1 : 0;
    }

    @Override
    public boolean failureIsCritical() {
        return delegate.phase().isCritical();
    }

    /**
     * Get the wrapped delegate plugin.
     */
    public InferencePhasePlugin delegate() {
        return delegate;
    }

    private NodeDescriptor buildDescriptor() {
        String phaseName = delegate.phase().getDisplayName();
        NodeCategory category = mapPhaseToCategory();

        return NodeDescriptor.builder(
                        delegate.id(),
                        phaseName + ": " + delegate.getClass().getSimpleName(),
                        category
                )
                .description("Phase plugin [" + phaseName + "] — " +
                        delegate.getClass().getName())
                .inputs(
                        NodePort.requiredInput("execution_context", NodePort.DataType.JSON),
                        NodePort.builder("engine_context", NodePort.Direction.INPUT, NodePort.DataType.JSON)
                                .displayName("Engine Context")
                                .description("Global engine context (optional, injected by executor)")
                                .build()
                )
                .outputs(
                        NodePort.output("execution_context", NodePort.DataType.JSON)
                )
                .defaultConfig(Map.of(
                        "phase", delegate.phase().name(),
                        "order", delegate.order()
                ))
                .tags("phase", phaseName.toLowerCase().replace(" ", "-"),
                        delegate.getClass().getSimpleName().toLowerCase())
                .build();
    }

    private NodeCategory mapPhaseToCategory() {
        return switch (delegate.phase()) {
            case PRE_VALIDATE, VALIDATE -> NodeCategory.VALIDATION;
            case AUTHORIZE -> NodeCategory.VALIDATION;
            case ROUTE -> NodeCategory.CONTROL_FLOW;
            case PRE_PROCESSING -> NodeCategory.PROCESSING;
            case PROVIDER_DISPATCH -> NodeCategory.INFERENCE;
            case POST_PROCESSING -> NodeCategory.PROCESSING;
            case AUDIT -> NodeCategory.OBSERVABILITY;
            case OBSERVABILITY -> NodeCategory.OBSERVABILITY;
            case CLEANUP -> NodeCategory.OUTPUT;
        };
    }
}
