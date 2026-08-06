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

package tech.kayys.gollek.spi.inference;

import tech.kayys.gollek.spi.execution.ExecutionContext;

/**
 * Observer for inference lifecycle events.
 * Allows monitoring and logging of inference operations.
 */
public interface InferenceObserver {

    /**
     * Called when inference starts
     */
    void onStart(ExecutionContext context);

    /**
     * Called when a phase starts
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
     * Called when streaming inference produces a chunk
     */
    void onInferenceChunk(Object chunk, ExecutionContext context);

    /**
     * Called when streaming inference completes
     */
    void onStreamComplete(ExecutionContext context);
}
