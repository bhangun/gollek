/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable profile describing which {@link Capability capabilities} an
 * {@link ExecutionProvider} supports.
 *
 * <p>Used by the router to filter providers that satisfy a request's requirements
 * before applying cost/latency scoring.</p>
 *
 * @param supported the set of capabilities this provider supports
 */
public record CapabilityProfile(Set<Capability> supported) {

    public CapabilityProfile {
        Objects.requireNonNull(supported, "supported capabilities must not be null");
        supported = Collections.unmodifiableSet(EnumSet.copyOf(supported));
    }

    /**
     * Check whether this profile satisfies all required capabilities.
     *
     * @param required the capabilities the request demands
     * @return {@code true} if every required capability is present in this profile
     */
    public boolean satisfies(Set<Capability> required) {
        return supported.containsAll(required);
    }

    /**
     * Check if a single capability is supported.
     */
    public boolean has(Capability capability) {
        return supported.contains(capability);
    }

    /**
     * Create a profile from individual capabilities.
     */
    public static CapabilityProfile of(Capability... capabilities) {
        return new CapabilityProfile(EnumSet.copyOf(Set.of(capabilities)));
    }

    /**
     * An empty profile — provider supports nothing beyond basic inference.
     */
    public static CapabilityProfile none() {
        return new CapabilityProfile(EnumSet.noneOf(Capability.class));
    }
}
