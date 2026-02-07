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

package tech.kayys.golek.core.inference;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.core.engine.EngineMetadata;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.model.HealthStatus;

/**
 * Main entry point for inference requests.
 * Thread-safe and stateless.
 */
public interface InferenceEngine {

        /**
         * Execute synchronous inference
         */
        Uni<InferenceResponse> infer(
                        InferenceRequest request,
                        TenantContext tenantContext);

        /**
         * Execute streaming inference
         */
        Uni<StreamingResponse> inferStream(
                        InferenceRequest request,
                        TenantContext tenantContext);

        /**
         * Get engine metadata
         */
        EngineMetadata metadata();

        /**
         * Health check
         */
        HealthStatus health();
}