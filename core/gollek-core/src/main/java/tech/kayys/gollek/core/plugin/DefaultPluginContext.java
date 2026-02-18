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

import tech.kayys.gollek.spi.plugin.PluginContext;
import java.util.Optional;

/**
 * Default implementation of PluginContext.
 */
public class DefaultPluginContext implements PluginContext {

    private final String pluginId;
    private final PluginManager pluginManager;

    public DefaultPluginContext(String pluginId, PluginManager pluginManager) {
        this.pluginId = pluginId;
        this.pluginManager = pluginManager;
    }

    @Override
    public String getPluginId() {
        return pluginId;
    }

    @Override
    public Optional<String> getConfig(String key) {
        // Placeholder for configuration retrieval logic
        // In a real implementation, this would query a configuration service
        return Optional.empty();
    }
}
