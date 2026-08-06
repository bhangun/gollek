/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

import java.util.Locale;

/**
 * Runtime readiness state for detachable unified multimodal extensions.
 */
public enum UnifiedRuntimeReadiness {
    READY,
    EXPERIMENTAL,
    PENDING,
    UNAVAILABLE;

    public boolean productionReady() {
        return this == READY;
    }

    public boolean attached() {
        return this == READY || this == EXPERIMENTAL || this == PENDING;
    }

    public String statusLabel() {
        return name().toLowerCase(Locale.ROOT);
    }
}
