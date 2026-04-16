/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.paged;

import jakarta.enterprise.context.ApplicationScoped;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Minimal registry for Paged KV sessions.
 * Stub implementation to resolve build dependencies.
 */
@ApplicationScoped
public class PagedKVSessionRegistry {

    public enum SessionPriority {
        LOW, NORMAL, HIGH
    }

    private final ConcurrentHashMap<String, Object> registrations = new ConcurrentHashMap<>();

    public void register(String sessionId, Object metadata, SessionPriority priority) {
        registrations.put(sessionId, priority);
    }

    public void deregister(String sessionId) {
        registrations.remove(sessionId);
    }
}
