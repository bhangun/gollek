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

import java.util.Optional;

/**
 * Holds per-request contextual information (tenant, device preference, cost sensitivity, etc.)
 * that is available throughout the inference pipeline.
 *
 * @since 1.0.0
 */
public interface RequestContext {
    String COMMUNITY_TENANT_ID = "community";
    String COMMUNITY_API_KEY = "community-key";

    /**
     * Factory for simple context with only a request ID.
     */
    static RequestContext of(String requestId) {
        return of(requestId, COMMUNITY_API_KEY);
    }

    /**
     * Factory for context with request ID and API key.
     */
    static RequestContext of(String requestId, String apiKey) {
        return new RequestContext() {
            @Override public String apiKey() { return apiKey; }
            @Override public String getRequestId() { return requestId; }
            @Override public String getTenantId() { return COMMUNITY_TENANT_ID; }
        };
    }

    // ---------- Identity / Routing ----------

    /** API key associated with this request. */
    String apiKey();

    /** Alias for apiKey() for backwards compatibility. */
    default String getApiKey() { return apiKey(); }

    /** Unique request identifier for tracing. */
    default String getRequestId() { return null; }

    /** Alias for getRequestId() for backwards compatibility. */
    default String requestId() { return getRequestId(); }

    /** Tenant id this request belongs to. */
    default String getTenantId() { return null; }

    /** Alias for getTenantId() for backwards compatibility. */
    default String tenantId() { return getTenantId(); }

    /** Model/version scope for multi-tenant registries. */
    default String getModelScope() { return null; }

    /** Alias for getUserId() or from metadata. */
    default String userId() { return null; }

    /** Alias for getSessionId() or from metadata. */
    default String sessionId() { return null; }

    /** Alias for getTraceId() or from metadata. */
    default String traceId() { return null; }

    // ---------- Device Preference ----------

    /**
     * Optional preferred device type (e.g. "cuda", "cpu", "metal").
     * Engine implementations are free to fall back to a different device when
     * the requested one is not available.
     */
    default Optional<String> getPreferredDevice() { return Optional.empty(); }

    // ---------- Cost / QoS Hints ----------

    /**
     * When {@code true} the routing layer should prefer cheaper / lower-latency
     * runners over higher-quality ones.
     */
    default boolean isCostSensitive() { return false; }

    /** Alias for isCostSensitive() for backwards compatibility. */
    default boolean costSensitive() { return isCostSensitive(); }

    /** Get preferred device type. */
    default java.util.Optional<tech.kayys.gollek.spi.model.DeviceType> preferredDevice() {
        return getPreferredDevice().map(tech.kayys.gollek.spi.model.DeviceType::fromId);
    }

    /** Create a new context with a preferred device. */
    default RequestContext withPreferredDevice(String deviceId) {
        return this; // Default: no-op transformation
    }

    /** Request timeout in milliseconds. */
    default long timeout() { return 30000L; }

    // ---------- Factory helpers ----------

    /**
     * Create a child context enriched with tenant information.
     *
     * @param tenantId  tenant identifier
     * @param modelScope model scope string
     * @return new {@link RequestContext} bound to the given tenant
     */
    static RequestContext forTenant(String tenantId, String modelScope) {
        return new RequestContext() {
            @Override public String apiKey()       { return COMMUNITY_API_KEY; }
            @Override public String getTenantId()  { return tenantId; }
            @Override public String getModelScope(){ return modelScope; }
        };
    }
}
