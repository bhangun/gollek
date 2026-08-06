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

package tech.kayys.gollek.plugin.optimization;

import java.util.Map;
import java.util.Optional;

/**
 * Execution context provided to optimization plugins.
 * 
 * @since 2.1.0
 */
public interface ExecutionContext {

    /**
     * Get a parameter value.
     */
    <T> Optional<T> getParameter(String key, Class<T> type);

    /**
     * Get a parameter with default value.
     */
    default int getParameter(String key, int defaultValue) {
        return getParameter(key, Integer.class).orElse(defaultValue);
    }

    /**
     * Get a memory buffer by name.
     */
    Optional<MemoryBuffer> getBuffer(String name);

    /**
     * Get all available buffers.
     */
    Map<String, MemoryBuffer> getBuffers();

    /**
     * Get the CUDA stream handle (if using CUDA).
     */
    default long getCudaStream() {
        return 0L;
    }

    /**
     * Get the ROCm stream handle (if using ROCm).
     */
    default long getHipStream() {
        return 0L;
    }

    /**
     * Get the current batch size.
     */
    int getBatchSize();

    /**
     * Get the sequence length.
     */
    int getSequenceLength();

    /**
     * Get the model architecture.
     */
    Optional<String> getModelArchitecture();

    /**
     * Get the GPU device ID.
     */
    int getDeviceId();

    /**
     * Check if running on GPU.
     */
    boolean isGpu();

    /**
     * Get execution phase.
     */
    ExecutionPhase getPhase();

    /**
     * Set a context attribute.
     */
    void setAttribute(String key, Object value);

    /**
     * Get a context attribute.
     */
    <T> Optional<T> getAttribute(String key, Class<T> type);

    /**
     * Execution phases for optimization.
     */
    enum ExecutionPhase {
        PREFILL,
        DECODE,
        BOTH
    }

    /**
     * Memory buffer interface for GPU/CPU memory.
     */
    interface MemoryBuffer {
        String getName();
        long getSize();
        long getPointer();
        boolean isDeviceMemory();
        void copyToDevice(Object data);
        Object copyToHost();
    }
}
