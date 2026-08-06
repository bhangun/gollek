/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

/**
 * Extension point for reporting whether a detachable extension is packaged,
 * healthy, and production-ready.
 */
public interface ExtensionAvailabilityProvider extends ExtensionPoint {
    String extensionId();

    String extensionName();

    default String extensionKind() {
        return "extension";
    }

    ExtensionAvailability availability();

    @Override
    default String getId() {
        return extensionId();
    }

    @Override
    default String getName() {
        return extensionName();
    }

    @Override
    default Class<?> getExtensionType() {
        return ExtensionAvailabilityProvider.class;
    }

    @Override
    default String getDescription() {
        return "Availability provider for " + extensionName();
    }
}
