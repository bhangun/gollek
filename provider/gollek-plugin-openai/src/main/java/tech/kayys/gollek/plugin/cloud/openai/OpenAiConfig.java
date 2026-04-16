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

package tech.kayys.gollek.plugin.cloud.openai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI plugin configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public record OpenAiConfig(
        @JsonProperty("apiKey") String apiKey,
        @JsonProperty("baseUrl") String baseUrl,
        @JsonProperty("organization") String organization,
        @JsonProperty("enabled") Boolean enabled,
        @JsonProperty("timeoutSeconds") Integer timeoutSeconds,
        @JsonProperty("maxRetries") Integer maxRetries
) {

    public OpenAiConfig {
        // Default values
        if (baseUrl == null) {
            baseUrl = "https://api.openai.com/v1";
        }
        if (enabled == null) {
            enabled = true;
        }
        if (timeoutSeconds == null) {
            timeoutSeconds = 30;
        }
        if (maxRetries == null) {
            maxRetries = 3;
        }
    }

    /**
     * Check if the configuration is valid.
     */
    public boolean isValid() {
        return enabled && apiKey != null && !apiKey.isBlank();
    }

    /**
     * Get the base URL without trailing slash.
     */
    public String getBaseUrlNormalized() {
        if (baseUrl.endsWith("/")) {
            return baseUrl.substring(0, baseUrl.length() - 1);
        }
        return baseUrl;
    }
}
