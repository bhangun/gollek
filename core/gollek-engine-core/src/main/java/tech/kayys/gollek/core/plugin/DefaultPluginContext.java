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

package tech.kayys.gollek.core.plugin;

import org.eclipse.microprofile.config.Config;
import org.eclipse.microprofile.config.ConfigProvider;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.plugin.PluginContext;

import java.util.Optional;

/**
 * Default implementation of PluginContext.
 * Provides actual configuration retrieval using SmallRye/MicroProfile Config.
 */
public class DefaultPluginContext implements PluginContext {

    private static final Logger LOG = Logger.getLogger(DefaultPluginContext.class);

    private final String pluginId;
    private final PluginManager pluginManager;
    private final Config config;

    public DefaultPluginContext(String pluginId, PluginManager pluginManager) {
        this.pluginId = pluginId;
        this.pluginManager = pluginManager;
        this.config = ConfigProvider.getConfig();
    }

    @Override
    public String getPluginId() {
        return pluginId;
    }

    @Override
    public Optional<String> getConfig(String key) {
        // Look for plugin-specific config: gollek.plugins.<pluginId>.<key>
        String qualifiedKey = String.format("gollek.plugins.%s.%s", pluginId, key);
        Optional<String> value = config.getOptionalValue(qualifiedKey, String.class);

        if (value.isPresent()) {
            LOG.debugf("Found plugin-specific config for [%s]: %s", qualifiedKey, value.get());
            return value;
        }

        // Fallback to global plugin config if applicable, or direct key
        Optional<String> fallbackValue = config.getOptionalValue(key, String.class);
        if (fallbackValue.isPresent()) {
            LOG.debugf("Found fallback config for key [%s]", key);
        }
        return fallbackValue;
    }

    /**
     * Get access to the plugin manager for service discovery between plugins.
     * 
     * @return the plugin manager
     */
    public PluginManager getPluginManager() {
        return pluginManager;
    }
}
