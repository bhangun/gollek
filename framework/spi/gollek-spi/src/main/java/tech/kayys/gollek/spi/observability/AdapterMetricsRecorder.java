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
 */

package tech.kayys.gollek.spi.observability;

/**
 * Adapter metrics recorder interface.
 *
 * @since 2.1.0
 */
public interface AdapterMetricsRecorder {

    /**
     * Record a successful inference call.
     *
     * @param schema Metrics schema
     * @param durationMs Call duration in milliseconds
     */
    void recordSuccess(AdapterMetricSchema schema, long durationMs);

    /**
     * Record a failed inference call.
     *
     * @param schema Metrics schema
     * @param durationMs Call duration in milliseconds
     * @param error Error message
     */
    void recordFailure(AdapterMetricSchema schema, long durationMs, String error);

    /**
     * Record tokens used.
     *
     * @param schema Metrics schema
     * @param inputTokens Number of input tokens
     * @param outputTokens Number of output tokens
     */
    void recordTokens(AdapterMetricSchema schema, int inputTokens, int outputTokens);

    /**
     * Close the recorder and release resources.
     */
    void close();
}
