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
package tech.kayys.gollek.spi.context;

import java.util.Map;
import java.util.concurrent.ExecutorService;

/**
 * Engine context interface.
 * Provides access to global engine resources and configuration.
 */
public interface EngineContext {

    /**
     * Get the global executor service for background tasks.
     */
    ExecutorService executorService();

    /**
     * Get the global engine configuration.
     */
    Map<String, Object> config();

    /**
     * Get a configuration value by key and type.
     */
    default <T> T getConfig(String key, Class<T> type) {
        Object value = config().get(key);
        if (value == null) return null;
        if (type.isInstance(value)) return type.cast(value);
        return null;
    }

    /**
     * Get a configuration value with a default.
     */
    @SuppressWarnings("unchecked")
    default <T> T getConfig(String key, T defaultValue) {
        Object value = config().get(key);
        return (value != null) ? (T) value : defaultValue;
    }

    /**
     * Check if the engine is currently running.
     */
    default boolean isRunning() { return true; }

    /**
     * Get the engine start time.
     */
    default java.time.Instant startTime() { return java.time.Instant.now(); }

    /**
     * Get the engine version.
     */
    default String version() { return "1.0.0"; }
}
