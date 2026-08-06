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

package tech.kayys.gollek.spi.plugin;

import java.util.Map;

/**
 * Configurable plugin interface.
 * 
 * <p>Plugins that implement this interface can be configured with custom parameters.</p>
 *
 * @since 2.1.0
 */
public interface GollekConfigurablePlugin extends GollekPlugin {

    /**
     * Configure the plugin with the given configuration.
     *
     * @param config Configuration parameters
     * @throws ConfigurationException if configuration is invalid
     */
    void configure(Map<String, Object> config) throws ConfigurationException;

    /**
     * Get current configuration.
     *
     * @return Current configuration map
     */
    default Map<String, Object> getConfiguration() {
        return Map.of();
    }

    /**
     * Exception thrown when plugin configuration is invalid.
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
