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

package tech.kayys.golek.observability;

import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.metrics.DoubleHistogram;
import io.opentelemetry.api.metrics.LongCounter;
import io.opentelemetry.api.metrics.Meter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for metrics collection
 */
public class MetricsRegistry {
    private final Meter meter;
    private final Map<String, DoubleHistogram> histograms = new ConcurrentHashMap<>();
    private final Map<String, LongCounter> counters = new ConcurrentHashMap<>();

    public MetricsRegistry(Meter meter) {
        this.meter = meter;
    }

    /**
     * Record a value for a histogram metric
     */
    public void record(String metricName, double value, Attributes attributes) {
        DoubleHistogram histogram = histograms.computeIfAbsent(metricName,
                name -> meter.histogramBuilder(name).build());
        histogram.record(value, attributes);
    }

    /**
     * Increment a counter metric
     */
    public void increment(String metricName, Attributes attributes) {
        LongCounter counter = counters.computeIfAbsent(metricName,
                name -> meter.counterBuilder(name).build());
        counter.add(1, attributes);
    }

    /**
     * Add a value to a counter metric
     */
    public void add(String metricName, long value, Attributes attributes) {
        LongCounter counter = counters.computeIfAbsent(metricName,
                name -> meter.counterBuilder(name).build());
        counter.add(value, attributes);
    }

    /**
     * Get a histogram metric
     */
    public DoubleHistogram getHistogram(String metricName) {
        return histograms.get(metricName);
    }

    /**
     * Get a counter metric
     */
    public LongCounter getCounter(String metricName) {
        return counters.get(metricName);
    }
}