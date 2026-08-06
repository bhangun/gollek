/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.util.Objects;
import java.util.function.Consumer;

/**
 * Owns a conversation KV-cache session until it is transferred to a trace.
 */
final class DirectConversationSessionOwner implements AutoCloseable {
    private KVCacheManager.KVCacheSession session;

    private DirectConversationSessionOwner(KVCacheManager.KVCacheSession session) {
        this.session = Objects.requireNonNull(session, "session");
    }

    static DirectConversationSessionOwner of(KVCacheManager.KVCacheSession session) {
        return new DirectConversationSessionOwner(session);
    }

    KVCacheManager.KVCacheSession session() {
        if (session == null) {
            throw new IllegalStateException("Conversation session ownership has already been transferred");
        }
        return session;
    }

    <T> T transfer(T value) {
        session = null;
        return value;
    }

    void transferTo(Consumer<DirectInferenceEngine.DirectConversationTrace> recipient,
            DirectInferenceEngine.DirectConversationTrace trace) {
        if (recipient == null) {
            return;
        }
        recipient.accept(trace);
        transfer(trace);
    }

    @Override
    public void close() {
        closeQuietly(session);
        session = null;
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }
}
