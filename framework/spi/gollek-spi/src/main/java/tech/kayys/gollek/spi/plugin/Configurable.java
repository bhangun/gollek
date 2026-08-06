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

import java.util.Map;

/**
 * Capability interface for components that support dynamic configuration.
 * 
 * <p>This is NOT a plugin interface - it's a capability that plugins or other
 * components can implement to support runtime configuration updates.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * public class MyPlugin implements GollekPlugin, Configurable {
 *     private Map<String, Object> config;
 *     
 *     @Override
 *     public void onConfigUpdate(Map<String, Object> newConfig) {
 *         this.config = newConfig;
 *         // Apply new configuration
 *     }
 *     
 *     @Override
 *     public Map<String, Object> currentConfig() {
 *         return config;
 *     }
 * }
 * }</pre>
 * 
 * @since 2.0.0
 */
public interface Configurable {

    /**
     * Update configuration at runtime.
     *
     * @param newConfig New configuration map
     * @throws ConfigurationException if config is invalid
     */
    void onConfigUpdate(Map<String, Object> newConfig) throws ConfigurationException;

    /**
     * Get current configuration.
     *
     * @return current configuration map
     */
    Map<String, Object> currentConfig();

    /**
     * Validate configuration without applying.
     *
     * @param config configuration to validate
     * @return true if valid
     */
    default boolean validateConfig(Map<String, Object> config) {
        try {
            onConfigUpdate(config);
            return true;
        } catch (ConfigurationException e) {
            return false;
        }
    }

    /**
     * Configuration exception.
     */
    class ConfigurationException extends Exception {
        public ConfigurationException(String message) {
            super(message);
        }

        public ConfigurationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
