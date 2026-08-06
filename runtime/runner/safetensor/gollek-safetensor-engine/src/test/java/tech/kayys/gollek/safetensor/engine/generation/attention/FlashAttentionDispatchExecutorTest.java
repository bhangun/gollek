/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlashAttentionDispatchExecutorTest {

    @Test
    void denseSharedUsesMetalResultWhenAvailable() {
        try (AccelTensor metalOut = AccelTensor.zeros(1);
                AccelTensor javaOut = AccelTensor.zeros(1)) {
            RecordingMetalBackend metal = new RecordingMetalBackend();
            RecordingJavaBackend java = new RecordingJavaBackend();
            metal.denseSharedResult = metalOut;
            java.denseSharedResult = javaOut;

            AccelTensor result = executor(metal, java).execute(FlashAttentionDispatchPath.DENSE_SHARED, request());

            assertSame(metalOut, result);
            assertEquals("denseShared", metal.lastCall);
            assertNull(java.lastCall);
        }
    }

    @Test
    void denseSharedFallsBackToJavaWhenMetalReturnsNull() {
        try (AccelTensor javaOut = AccelTensor.zeros(1)) {
            RecordingMetalBackend metal = new RecordingMetalBackend();
            RecordingJavaBackend java = new RecordingJavaBackend();
            java.denseSharedResult = javaOut;

            AccelTensor result = executor(metal, java).execute(FlashAttentionDispatchPath.DENSE_SHARED, request());

            assertSame(javaOut, result);
            assertEquals("denseShared", metal.lastCall);
            assertEquals("denseShared", java.lastCall);
        }
    }

    @Test
    void routesMetalPathsToMetalBackend() {
        try (AccelTensor tiledOut = AccelTensor.zeros(1);
                AccelTensor slidingOut = AccelTensor.zeros(1)) {
            RecordingMetalBackend metal = new RecordingMetalBackend();
            RecordingJavaBackend java = new RecordingJavaBackend();
            metal.tiledResult = tiledOut;
            metal.slidingResult = slidingOut;
            FlashAttentionDispatchExecutor executor = executor(metal, java);

            assertSame(tiledOut, executor.execute(FlashAttentionDispatchPath.FA4_PAGED_METAL, request()));
            assertEquals("tiled", metal.lastCall);
            assertSame(tiledOut, executor.execute(FlashAttentionDispatchPath.METAL_TILED, request()));
            assertEquals("tiled", metal.lastCall);
            assertSame(slidingOut, executor.execute(FlashAttentionDispatchPath.SLIDING_DECODE_METAL, request()));
            assertEquals("slidingDecode", metal.lastCall);
            assertNull(java.lastCall);
        }
    }

    @Test
    void routesJavaPathsToJavaBackend() {
        try (AccelTensor denseRestrictedOut = AccelTensor.zeros(1);
                AccelTensor pagedOut = AccelTensor.zeros(1)) {
            RecordingMetalBackend metal = new RecordingMetalBackend();
            RecordingJavaBackend java = new RecordingJavaBackend();
            java.denseRestrictedResult = denseRestrictedOut;
            java.pagedResult = pagedOut;
            FlashAttentionDispatchExecutor executor = executor(metal, java);

            assertSame(denseRestrictedOut,
                    executor.execute(FlashAttentionDispatchPath.DENSE_RESTRICTED_JAVA, request()));
            assertEquals("denseRestricted", java.lastCall);
            assertSame(pagedOut, executor.execute(FlashAttentionDispatchPath.PAGED_JAVA, request()));
            assertEquals("paged", java.lastCall);
            assertNull(metal.lastCall);
        }
    }

    @Test
    void rejectsMissingBackends() {
        RecordingMetalBackend metal = new RecordingMetalBackend();
        RecordingJavaBackend java = new RecordingJavaBackend();

        assertThrows(NullPointerException.class, () -> new FlashAttentionDispatchExecutor(null, java));
        assertThrows(NullPointerException.class, () -> new FlashAttentionDispatchExecutor(metal, null));
    }

    private static FlashAttentionDispatchExecutor executor(RecordingMetalBackend metal,
            RecordingJavaBackend java) {
        return new FlashAttentionDispatchExecutor(metal, java);
    }

    private static FlashAttentionDispatchRequest request() {
        return new FlashAttentionDispatchRequest(
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0f,
                false,
                0.0f,
                null,
                false);
    }

    private static final class RecordingMetalBackend implements FlashAttentionMetalDispatchBackend {
        private String lastCall;
        private AccelTensor denseSharedResult;
        private AccelTensor tiledResult;
        private AccelTensor slidingResult;

        @Override
        public AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request) {
            lastCall = "denseShared";
            return denseSharedResult;
        }

        @Override
        public AccelTensor tiledAttention(FlashAttentionDispatchRequest request) {
            lastCall = "tiled";
            return tiledResult;
        }

        @Override
        public AccelTensor slidingDecodeAttention(FlashAttentionDispatchRequest request) {
            lastCall = "slidingDecode";
            return slidingResult;
        }
    }

    private static final class RecordingJavaBackend implements FlashAttentionJavaDispatchBackend {
        private String lastCall;
        private AccelTensor denseSharedResult;
        private AccelTensor denseRestrictedResult;
        private AccelTensor pagedResult;

        @Override
        public AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request) {
            lastCall = "denseShared";
            return denseSharedResult;
        }

        @Override
        public AccelTensor denseRestrictedAttention(FlashAttentionDispatchRequest request) {
            lastCall = "denseRestricted";
            return denseRestrictedResult;
        }

        @Override
        public AccelTensor pagedAttention(FlashAttentionDispatchRequest request) {
            lastCall = "paged";
            return pagedResult;
        }
    }
}
