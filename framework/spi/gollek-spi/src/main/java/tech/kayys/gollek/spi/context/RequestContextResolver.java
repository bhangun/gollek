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

package tech.kayys.gollek.spi.context;

import jakarta.ws.rs.container.ContainerRequestContext;

/**
 * Resolver for extracting request context from JAX-RS requests.
 *
 * @since 2.1.0
 */
public interface RequestContextResolver {

    /**
     * Extract request context from JAX-RS container request.
     *
     * @param request JAX-RS container request context
     * @return Request context
     */
    tech.kayys.gollek.spi.context.RequestContext resolve(ContainerRequestContext request);

    /**
     * Request context holder.
     *
     * @param requestId Request ID
     * @param tenantId Tenant ID
     * @param apiKey API key
     * @param userId User ID
     */
    record DefaultRequestContext(
            String requestId,
            String tenantId,
            String apiKey,
            String userId
    ) implements tech.kayys.gollek.spi.context.RequestContext {

        @Override
        public String getRequestId() { return requestId; }

        @Override
        public String getTenantId() { return tenantId; }

        public static DefaultRequestContext of(String requestId, String tenantId, String apiKey, String userId) {
            return new DefaultRequestContext(requestId, tenantId, apiKey, userId);
        }

        public static DefaultRequestContext anonymous(String requestId) {
            return new DefaultRequestContext(requestId, "anonymous", null, null);
        }
    }
}
