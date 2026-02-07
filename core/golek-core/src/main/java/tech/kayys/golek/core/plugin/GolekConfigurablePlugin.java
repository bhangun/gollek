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
 * @author bhangun
 */

package tech.kayys.golek.core.plugin;

import tech.kayys.golek.spi.plugin.GolekPlugin;

import java.util.Map;

/**
 * Plugin that supports dynamic configuration updates.
 * Allows runtime reconfiguration without restart.
 */
public interface GolekConfigurablePlugin extends GolekPlugin {

    /**
     * Update plugin configuration at runtime.
     * 
     * @param newConfig New configuration map
     * @throws ConfigurationException if config is invalid
     */
    void onConfigUpdate(Map<String, Object> newConfig)
            throws ConfigurationException;

    /**
     * Get current configuration
     */
    Map<String, Object> currentConfig();

    /**
     * Validate configuration without applying
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
     * Configuration exception
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