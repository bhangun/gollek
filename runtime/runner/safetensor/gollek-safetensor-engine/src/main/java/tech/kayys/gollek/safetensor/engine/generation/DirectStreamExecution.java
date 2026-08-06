/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.MultiEmitter;
import org.jboss.logging.Logger;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Owns the common worker-thread envelope for direct streaming generation.
 */
final class DirectStreamExecution {
    private DirectStreamExecution() {
    }

    static <T> Multi<T> create(ThreadFactory threadFactory, Logger log, String failureMessage,
            Runnable cleanup, StreamWork<T> work) {
        Objects.requireNonNull(threadFactory, "threadFactory");
        Objects.requireNonNull(log, "log");
        Objects.requireNonNull(failureMessage, "failureMessage");
        Objects.requireNonNull(work, "work");

        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(threadFactory);
            executor.submit(() -> runWork(executor, emitter, log, failureMessage, cleanup, work));
        });
    }

    private static <T> void runWork(ExecutorService executor, MultiEmitter<? super T> emitter, Logger log,
            String failureMessage, Runnable cleanup, StreamWork<T> work) {
        Throwable failure = null;
        try {
            work.run(emitter);
        } catch (Throwable t) {
            failure = t;
            log.error(failureMessage, t);
        } finally {
            failure = cleanup(log, failure, cleanup);
            try {
                if (failure == null) {
                    emitter.complete();
                } else {
                    emitter.fail(failure);
                }
            } finally {
                executor.shutdownNow();
            }
        }
    }

    private static Throwable cleanup(Logger log, Throwable failure, Runnable cleanup) {
        if (cleanup == null) {
            return failure;
        }
        try {
            cleanup.run();
            return failure;
        } catch (Throwable cleanupFailure) {
            log.error("Streaming cleanup failed", cleanupFailure);
            if (failure == null) {
                return cleanupFailure;
            }
            failure.addSuppressed(cleanupFailure);
            return failure;
        }
    }

    @FunctionalInterface
    interface StreamWork<T> {
        void run(MultiEmitter<? super T> emitter) throws Throwable;
    }
}
