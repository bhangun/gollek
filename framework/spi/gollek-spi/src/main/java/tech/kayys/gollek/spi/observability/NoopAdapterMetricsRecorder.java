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
 * No-op implementation of AdapterMetricsRecorder.
 * 
 * <p>Useful for testing or when metrics collection is not required.</p>
 *
 * @since 2.1.0
 */
public class NoopAdapterMetricsRecorder implements AdapterMetricsRecorder {

    @Override
    public void recordSuccess(AdapterMetricSchema schema, long durationMs) {
        // No-op
    }

    @Override
    public void recordFailure(AdapterMetricSchema schema, long durationMs, String error) {
        // No-op
    }

    @Override
    public void recordTokens(AdapterMetricSchema schema, int inputTokens, int outputTokens) {
        // No-op
    }

    @Override
    public void close() {
        // No-op
    }
}
