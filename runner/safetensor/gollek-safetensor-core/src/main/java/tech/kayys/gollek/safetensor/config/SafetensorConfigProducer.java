/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorConfigProducer.java
 * ──────────────────────────────
 * CDI producer for SafeTensor configuration.
 */
package tech.kayys.gollek.safetensor.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;

/**
 * CDI producer for SafeTensor module beans.
 */
@ApplicationScoped
public class SafetensorConfigProducer {

    /**
     * Produce ObjectMapper bean.
     */
    @Produces
    @ApplicationScoped
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }
}
