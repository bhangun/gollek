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

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Audit payload for tracking events and provenance.
 *
 * @param eventType Type of audit event
 * @param runId Unique execution identifier
 * @param timestamp Event timestamp
 * @param actor Actor who triggered the event
 * @param action Action performed
 * @param resource Resource affected
 * @param metadata Additional metadata
 *
 * @since 2.1.0
 */
public record AuditPayload(
        String eventType,
        String runId,
        String level,
        Instant timestamp,
        String actor,
        String action,
        String resource,
        Map<String, Object> metadata
) {

    public AuditPayload {
        if (level == null) {
            level = "INFO";
        }
        if (timestamp == null) {
            timestamp = Instant.now();
        }
        if (metadata == null) {
            metadata = Map.of();
        }
    }

    /**
     * Helper for actor identifiers.
     */
    public static class Actor {
        public static String system(String name) {
            return "system:" + name;
        }
        public static String user(String id) {
            return "user:" + id;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String eventType;
        private String runId;
        private String level = "INFO";
        private Instant timestamp;
        private String actor;
        private String action;
        private String resource;
        private Map<String, Object> metadata = new HashMap<>();

        public Builder eventType(String eventType) {
            this.eventType = eventType;
            return this;
        }

        public Builder event(String event) {
            return eventType(event);
        }

        public Builder runId(String runId) {
            this.runId = runId;
            return this;
        }

        public Builder level(String level) {
            this.level = level;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder actor(String actor) {
            this.actor = actor;
            return this;
        }

        public Builder action(String action) {
            this.action = action;
            return this;
        }

        public Builder resource(String resource) {
            this.resource = resource;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = new HashMap<>(metadata);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder tag(String tag) {
            this.metadata.put("tag:" + tag, true);
            return this;
        }

        public Builder contextSnapshot(Map<String, Object> snapshot) {
            this.metadata.put("context_snapshot", snapshot);
            return this;
        }

        public AuditPayload build() {
            return new AuditPayload(eventType, runId, level, timestamp, actor, action, resource, Map.copyOf(metadata));
        }
    }
}
