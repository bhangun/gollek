/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectConversationSessionOwnerTest {

    @Test
    void closesSessionWhenOwnershipIsNotTransferred() {
        TestSession session = new TestSession();

        try (DirectConversationSessionOwner ignored = DirectConversationSessionOwner.of(session)) {
        }

        assertTrue(session.closed);
    }

    @Test
    void leavesSessionOpenAfterTransfer() {
        TestSession session = new TestSession();
        Object result = new Object();
        Object transferred;

        try (DirectConversationSessionOwner owner = DirectConversationSessionOwner.of(session)) {
            assertSame(session, owner.session());
            transferred = owner.transfer(result);
            assertThrows(IllegalStateException.class, owner::session);
        }

        assertSame(result, transferred);
        assertFalse(session.closed);
    }

    @Test
    void transfersSessionToCompletionCallback() {
        TestSession session = new TestSession();
        DirectInferenceEngine.DirectConversationTrace trace = trace(session);
        AtomicBoolean callbackCalled = new AtomicBoolean(false);

        try (DirectConversationSessionOwner owner = DirectConversationSessionOwner.of(session)) {
            owner.transferTo(received -> {
                assertSame(trace, received);
                callbackCalled.set(true);
            }, trace);
        }

        assertTrue(callbackCalled.get());
        assertFalse(session.closed);
    }

    @Test
    void closesSessionWhenCompletionCallbackIsMissing() {
        TestSession session = new TestSession();

        try (DirectConversationSessionOwner owner = DirectConversationSessionOwner.of(session)) {
            owner.transferTo(null, trace(session));
        }

        assertTrue(session.closed);
    }

    @Test
    void closesSessionWhenCompletionCallbackFails() {
        TestSession session = new TestSession();
        IllegalStateException failure = new IllegalStateException("callback failed");

        IllegalStateException thrown = assertThrows(IllegalStateException.class, () -> {
            try (DirectConversationSessionOwner owner = DirectConversationSessionOwner.of(session)) {
                owner.transferTo(ignored -> {
                    throw failure;
                }, trace(session));
            }
        });

        assertSame(failure, thrown);
        assertTrue(session.closed);
    }

    private static DirectInferenceEngine.DirectConversationTrace trace(KVCacheManager.KVCacheSession session) {
        InferenceResponse response = InferenceResponse.builder()
                .requestId("request")
                .content("")
                .build();
        return new DirectInferenceEngine.DirectConversationTrace(
                response,
                new long[0],
                new long[0],
                session);
    }

    private static final class TestSession extends KVCacheManager.KVCacheSession {
        private boolean closed;

        private TestSession() {
            super(1, new BlockManager());
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
