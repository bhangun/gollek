/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.spi.plugin;

/**
 * Backpressure modes for streaming.
 */
public enum BackpressureMode {
    /** Buffer chunks in memory until consumed */
    BUFFER,
    /** Drop oldest chunks when buffer is full */
    DROP_OLDEST,
    /** Keep only the latest chunk when buffer is full */
    LATEST,
    /** Signal error when buffer is full */
    ERROR
}
