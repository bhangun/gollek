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

package tech.kayys.gollek.core.pipeline;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.core.execution.ExecutionContext;

/**
 * Pipeline that executes all phases in order.
 */
public interface InferencePipeline {

    /**
     * Execute all phases for the given context
     */
    Uni<ExecutionContext> execute(ExecutionContext context);

    /**
     * Execute a specific phase
     */
    Uni<ExecutionContext> executePhase(ExecutionContext context, InferencePhase phase);
}