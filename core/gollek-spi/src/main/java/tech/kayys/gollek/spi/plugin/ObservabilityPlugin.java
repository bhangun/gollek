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

package tech.kayys.gollek.spi.plugin;

import tech.kayys.gollek.spi.inference.InferencePhase;

import java.time.Duration;
import java.util.Map;

/**
 * SPI for plugins that provide observability capabilities.
 * <p>
 * Responsible for:
 * <ul>
 * <li>Token usage tracking (input/output)</li>
 * <li>Latency recording per phase</li>
 * <li>Step count tracking</li>
 * <li>Tool usage recording</li>
 * <li>Distributed trace context</li>
 * <li>Error cause tracking</li>
 * </ul>
 * <p>
 * Designed for integration with OpenTelemetry, Prometheus, and logging
 * frameworks.
 */
public interface ObservabilityPlugin extends GollekPlugin {

    /**
     * Record token usage for an inference request.
     *
     * @param inputTokens  number of input tokens
     * @param outputTokens number of output tokens
     * @param modelId      the model identifier
     * @param apiKey       the tenant identifier
     */
    void recordTokenUsage(int inputTokens, int outputTokens, String modelId, String apiKey);

    /**
     * API-key-first alias for {@link #recordTokenUsage(int, int, String, String)}.
     */
    default void recordTokenUsageByApiKey(int inputTokens, int outputTokens, String modelId, String apiKey) {
        recordTokenUsage(inputTokens, outputTokens, modelId, apiKey);
    }

    /**
     * Record latency for a specific inference phase.
     *
     * @param phase    the inference phase
     * @param duration the duration of the phase
     * @param success  whether the phase completed successfully
     */
    void recordLatency(InferencePhase phase, Duration duration, boolean success);

    /**
     * Record tool usage.
     *
     * @param toolId   the tool identifier
     * @param success  whether the tool call was successful
     * @param duration the tool execution duration
     */
    void recordToolUsage(String toolId, boolean success, Duration duration);

    /**
     * Record an error that occurred during inference.
     *
     * @param phase   the phase where the error occurred
     * @param error   the error
     * @param context additional context about the error
     */
    void recordError(InferencePhase phase, Throwable error, Map<String, String> context);

    /**
     * Record the total number of reasoning steps for a request.
     *
     * @param steps   the number of steps taken
     * @param modelId the model identifier
     */
    void recordStepCount(int steps, String modelId);

    /**
     * Get or create a trace ID for the current request.
     *
     * @param requestId the request identifier
     * @return the trace ID (may be propagated from upstream or generated)
     */
    String getOrCreateTraceId(String requestId);

    /**
     * Flush any buffered metrics.
     * Called during shutdown or periodic flush intervals.
     */
    default void flush() {
        // Default: no-op
    }
}
