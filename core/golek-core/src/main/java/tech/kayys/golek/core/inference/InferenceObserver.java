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

package tech.kayys.golek.core.inference;

import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.core.execution.ExecutionContext;

/**
 * Observer interface for inference lifecycle events.
 * Implementations provide metrics, tracing, and logging hooks.
 * 
 * All methods are invoked synchronously in the execution thread.
 * Implementations should be non-blocking and fast.
 */
public interface InferenceObserver {

    /**
     * Called when inference starts
     */
    void onStart(ExecutionContext context);

    /**
     * Called when a phase begins
     */
    void onPhase(InferencePhase phase, ExecutionContext context);

    /**
     * Called when inference completes successfully
     */
    void onSuccess(ExecutionContext context);

    /**
     * Called when inference fails
     */
    void onFailure(Throwable error, ExecutionContext context);

    /**
     * Called when a plugin executes (optional, for detailed tracing)
     */
    default void onPluginExecute(String pluginId, ExecutionContext context) {
        // Default: no-op
    }

    /**
     * Called when a provider is invoked
     */
    default void onProviderInvoke(String providerId, ExecutionContext context) {
        // Default: no-op
    }
}