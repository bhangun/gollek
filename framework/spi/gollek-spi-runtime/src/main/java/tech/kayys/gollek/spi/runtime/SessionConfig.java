/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import tech.kayys.alkhawarizm.spi.model.ModelDescriptor;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Configuration for creating a {@link RuntimeSession}.
 *
 * <p>Carries the model descriptor, device preference, and any
 * runner-specific parameters needed to initialise the session.</p>
 *
 * @param model      the model to load into the session
 * @param deviceHint preferred device (e.g. "cpu", "cuda:0", "mps")
 * @param parameters runner-specific key-value configuration overrides
 */
public record SessionConfig(
        ModelDescriptor model,
        String deviceHint,
        Map<String, Object> parameters) {

    public SessionConfig {
        Objects.requireNonNull(model, "model");
        if (deviceHint == null) {
            deviceHint = "cpu";
        }
        parameters = parameters != null
                ? Map.copyOf(parameters)
                : Map.of();
    }

    /**
     * Create a minimal session config for the given model on CPU.
     */
    public static SessionConfig ofCpu(ModelDescriptor model) {
        return new SessionConfig(model, "cpu", Map.of());
    }

    /**
     * Look up an optional parameter by key.
     */
    public <T> Optional<T> param(String key, Class<T> type) {
        Object value = parameters.get(key);
        return type.isInstance(value) ? Optional.of(type.cast(value)) : Optional.empty();
    }
}
