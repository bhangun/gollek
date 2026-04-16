/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorBeans.java
 * ────────────────────
 * CDI producer configuration for beans in the safetensor module that cannot be
 * annotated with @ApplicationScoped directly (e.g. because they are non-CDI
 * classes like SafetensorHeaderParser) or that need conditional creation.
 *
 * Produced beans
 * ══════════════
 *  • SafetensorHeaderParser  — stateless parser; produced as @ApplicationScoped
 *                              singleton backed by the Quarkus ObjectMapper.
 *
 * Why a producer?
 * ═══════════════
 * SafetensorHeaderParser is a plain Java class (no CDI annotations) intentionally
 * — keeping it POJO-style makes it usable in non-CDI contexts (unit tests,
 * CLI tools) without pulling in Quarkus.  The producer bridges it into the CDI
 * world for use inside the Quarkus container.
 */
package tech.kayys.gollek.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;

/**
 * CDI configuration class — produces application-scoped beans for the
 * safetensor module.
 */
@ApplicationScoped
public class SafetensorBeans {

    @Inject
    ObjectMapper objectMapper;

    /**
     * Produce the application-scoped {@link SafetensorHeaderParser} singleton.
     *
     * <p>
     * The parser is stateless (all state lives in the method-local scope)
     * so a single shared instance is safe for concurrent use from multiple
     * threads.
     *
     * @return the parser backed by Quarkus-managed Jackson ObjectMapper
     */
    @Produces
    @ApplicationScoped
    SafetensorHeaderParser safetensorHeaderParser() {
        return SafetensorHeaderParser.create(objectMapper);
    }
}
