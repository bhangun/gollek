/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.streaming;

import tech.kayys.golek.spi.plugin.StreamingPlugin;
import tech.kayys.golek.spi.plugin.BackpressureMode;

/**
 * Backpressure strategy configuration for streaming.
 *
 * @param mode          the backpressure mode
 * @param maxBufferSize maximum buffer size (for BUFFER and DROP_OLDEST modes)
 * @param dropOldest    whether to drop oldest items when buffer is full
 */
public record BackpressureStrategy(
        BackpressureMode mode,
        int maxBufferSize,
        boolean dropOldest) {

    /**
     * Create a buffering backpressure strategy.
     */
    public static BackpressureStrategy buffer(int maxSize) {
        return new BackpressureStrategy(
                BackpressureMode.BUFFER,
                maxSize, false);
    }

    /**
     * Create a drop-oldest backpressure strategy.
     */
    public static BackpressureStrategy dropOldest(int maxSize) {
        return new BackpressureStrategy(
                BackpressureMode.DROP_OLDEST,
                maxSize, true);
    }

    /**
     * Create a latest-only backpressure strategy.
     */
    public static BackpressureStrategy latest() {
        return new BackpressureStrategy(
                BackpressureMode.LATEST,
                1, true);
    }

    /**
     * Create an error-on-full backpressure strategy.
     */
    public static BackpressureStrategy error(int maxSize) {
        return new BackpressureStrategy(
                BackpressureMode.ERROR,
                maxSize, false);
    }
}
