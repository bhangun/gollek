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

package tech.kayys.golek.core.engine;

import java.util.List;

/**
 * Engine metadata containing version and capability information.
 */
public record EngineMetadata(
        String version,
        List<String> supportedPhases,
        String buildTime,
        String gitCommit) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String version = "unknown";
        private List<String> supportedPhases = List.of();
        private String buildTime = "unknown";
        private String gitCommit = "unknown";

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder supportedPhases(List<String> phases) {
            this.supportedPhases = phases;
            return this;
        }

        public Builder buildTime(String buildTime) {
            this.buildTime = buildTime;
            return this;
        }

        public Builder gitCommit(String gitCommit) {
            this.gitCommit = gitCommit;
            return this;
        }

        public EngineMetadata build() {
            return new EngineMetadata(version, supportedPhases, buildTime, gitCommit);
        }
    }
}
