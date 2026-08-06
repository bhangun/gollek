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

package tech.kayys.gollek.plugin.runner.safetensor.feature;

import java.util.Map;
import java.util.Set;

/**
 * SPI for Safetensor feature plugins.
 * 
 * <p>Feature plugins provide domain-specific capabilities for the Safetensor runner:
 * <ul>
 *   <li>Audio processing (Whisper, SpeechT5, etc.)</li>
 *   <li>Vision processing (CLIP, ViT, etc.)</li>
 *   <li>Text processing (LLMs, BERT, etc.)</li>
 * </ul>
 * 
 * <h2>Feature Plugin Lifecycle</h2>
 * <ol>
 *   <li>{@link #initialize(Map)} - Called once during runner startup</li>
 *   <li>{@link #isAvailable()} - Check if feature is available</li>
 *   <li>{@link #process(Object)} - Process domain-specific input</li>
 *   <li>{@link #shutdown()} - Cleanup resources</li>
 * </ol>
 * 
 * <h2>Example Implementation</h2>
 * <pre>{@code
 * public class AudioFeaturePlugin implements SafetensorFeaturePlugin {
 *     
 *     @Override
 *     public String id() {
 *         return "audio-feature";
 *     }
 *     
 *     @Override
 *     public String name() {
 *         return "Audio Processing";
 *     }
 *     
 *     @Override
 *     public Set<String> supportedModels() {
 *         return Set.of("whisper", "speecht5", "wav2vec2");
 *     }
 *     
 *     @Override
 *     public boolean isAvailable() {
 *         return audioEngine != null && audioEngine.isHealthy();
 *     }
 *     
 *     @Override
 *     public Object process(Object input) {
 *         // Process audio input
 *         return audioEngine.transcribe(input);
 *     }
 * }
 * }</pre>
 * 
 * @since 2.1.0
 */
public interface SafetensorFeaturePlugin {

    /**
     * Unique feature plugin identifier.
     * 
     * @return feature ID (e.g., "audio-feature", "vision-feature")
     */
    String id();

    /**
     * Human-readable name.
     * 
     * @return feature name
     */
    String name();

    /**
     * Feature plugin version.
     * 
     * @return semantic version
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Feature description.
     * 
     * @return description of what this feature does
     */
    String description();

    /**
     * Initialize the feature with configuration.
     * 
     * @param config Configuration parameters
     */
    default void initialize(Map<String, Object> config) {
        // Default: no-op
    }

    /**
     * Check if this feature is available.
     * 
     * @return true if feature can be used
     */
    boolean isAvailable();

    /**
     * Get the feature priority. Higher values execute first.
     * 
     * @return priority value (default: 0)
     */
    default int priority() {
        return 0;
    }

    /**
     * Get supported model architectures for this feature.
     * 
     * @return set of supported architectures (e.g., "whisper", "clip", "llama")
     */
    Set<String> supportedModels();

    /**
     * Get supported input types for this feature.
     * 
     * @return set of input types (e.g., "audio/wav", "image/png", "text/plain")
     */
    default Set<String> supportedInputTypes() {
        return Set.of();
    }

    /**
     * Get supported output types for this feature.
     * 
     * @return set of output types
     */
    default Set<String> supportedOutputTypes() {
        return Set.of();
    }

    /**
     * Process input using this feature.
     * 
     * @param input Input data
     * @return Processed output
     */
    Object process(Object input);

    /**
     * Get feature-specific metadata.
     * 
     * @return metadata map
     */
    default Map<String, Object> metadata() {
        return Map.of();
    }

    /**
     * Shutdown and cleanup resources.
     */
    default void shutdown() {
        // Default: no-op
    }

    /**
     * Check if feature is healthy and operational.
     * 
     * @return true if feature is healthy
     */
    default boolean isHealthy() {
        return true;
    }
}
