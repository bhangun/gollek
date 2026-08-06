/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.inject.Instance;
import org.jboss.logging.Logger;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Owns the common request envelope for synchronous direct generation.
 */
final class DirectSyncExecution {
    private DirectSyncExecution() {
    }

    static <T> Uni<T> create(Instance<?> metalBackend, Logger log, String logMessage,
            String failurePrefix, SyncWork<T> work) {
        Objects.requireNonNull(work, "work");

        return Uni.createFrom().item((Supplier<T>) () -> run(metalBackend, log, logMessage, failurePrefix, work));
    }

    private static <T> T run(Instance<?> metalBackend, Logger log, String logMessage,
            String failurePrefix, SyncWork<T> work) {
        DirectGenerationRequestContext request = DirectGenerationRequestContext.sync(metalBackend);
        try {
            DirectRuntimePlatformBanner.print(metalBackend);
            return work.run(request);
        } catch (Exception e) {
            throw DirectGenerationFailures.wrap(log, logMessage, failurePrefix, e);
        } finally {
            DirectInferenceProfiler.clearProfile();
        }
    }

    @FunctionalInterface
    interface SyncWork<T> {
        T run(DirectGenerationRequestContext request) throws Exception;
    }
}
