/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import tech.kayys.alkhawarizm.spi.model.ModelDescriptor;

import java.util.Objects;

/**
 * Binds a model, its execution provider, and the resolved capability profile
 * into a single decision unit.
 *
 * <p>The router produces {@code ModelBinding} instances after matching a
 * request to a provider. Downstream components (scheduler, pipeline) use
 * the binding to create sessions and enforce capability contracts.</p>
 *
 * @param model        the model descriptor
 * @param provider     the selected execution provider
 * @param capabilities the intersection of provider capabilities and request requirements
 */
public record ModelBinding(
        ModelDescriptor model,
        ExecutionProvider provider,
        CapabilityProfile capabilities) {

    public ModelBinding {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(provider, "provider");
        Objects.requireNonNull(capabilities, "capabilities");
    }
}
