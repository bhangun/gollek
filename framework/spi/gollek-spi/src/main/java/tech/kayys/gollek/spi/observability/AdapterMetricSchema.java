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

package tech.kayys.gollek.spi.observability;

import java.util.Map;

/**
 * Adapter metric schema.
 *
 * @param adapterId Adapter identifier
 * @param modelId Model identifier
 * @param operation Operation name
 * @param tags Additional tags
 *
 * @since 2.1.0
 */
public record AdapterMetricSchema(
        String adapterId,
        String modelId,
        String operation,
        Map<String, String> tags
) {

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String adapterId;
        private String modelId;
        private String operation;
        private Map<String, String> tags = Map.of();

        public Builder adapterId(String adapterId) {
            this.adapterId = adapterId;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder operation(String operation) {
            this.operation = operation;
            return this;
        }

        public Builder tags(Map<String, String> tags) {
            this.tags = tags;
            return this;
        }

        public AdapterMetricSchema build() {
            return new AdapterMetricSchema(adapterId, modelId, operation, tags);
        }
    }
}
