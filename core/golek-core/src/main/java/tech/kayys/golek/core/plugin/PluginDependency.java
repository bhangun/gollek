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

/**
 * Plugin dependency
 */
public class PluginDependency {
    private final String pluginId;
    private final String versionRange;
    private final boolean optional;

    public PluginDependency(String pluginId, String versionRange, boolean optional) {
        this.pluginId = pluginId;
        this.versionRange = versionRange;
        this.optional = optional;
    }

    public String getPluginId() {
        return pluginId;
    }

    public String getVersionRange() {
        return versionRange;
    }

    public boolean isOptional() {
        return optional;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String pluginId;
        private String versionRange;
        private boolean optional = false;

        public Builder pluginId(String pluginId) {
            this.pluginId = pluginId;
            return this;
        }

        public Builder versionRange(String versionRange) {
            this.versionRange = versionRange;
            return this;
        }

        public Builder optional(boolean optional) {
            this.optional = optional;
            return this;
        }

        public PluginDependency build() {
            return new PluginDependency(pluginId, versionRange, optional);
        }
    }
}