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

package tech.kayys.golek.core.plugin;

import tech.kayys.golek.spi.plugin.GolekPlugin;

/**
 * Plugin listener interface
 */
public interface PluginListener {
    /**
     * Called when a plugin is registered
     */
    default void onPluginRegistered(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is unregistered
     */
    default void onPluginUnregistered(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is started
     */
    default void onPluginStarted(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is stopped
     */
    default void onPluginStopped(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin state changes
     */
    // default void onPluginStateChanged(GolekPlugin plugin, PluginState oldState,
    // PluginState newState) {}
}