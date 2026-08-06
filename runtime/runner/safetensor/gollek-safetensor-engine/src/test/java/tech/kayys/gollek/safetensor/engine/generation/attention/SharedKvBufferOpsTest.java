/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SharedKvBufferOpsTest {

    @Test
    void appendsTokenMajorDeltaIntoTokenMajorAndHeadMajorBuffers() {
        try (AccelTensor tokenMajor = AccelTensor.zeros(1, 3, 2, 2);
                AccelTensor headMajor = AccelTensor.zeros(1, 2, 3, 2);
                AccelTensor delta = tensor(1.0f, 1, 2, 2, 2)) {
            SharedKvBufferOps.appendTokenMajor(tokenMajor, delta, 1, 2, 2);
            SharedKvBufferOps.appendTokenMajorToHeadMajor(headMajor, delta, 1, 2, 2);

            assertArrayEquals(new float[] {
                    0.0f, 0.0f, 0.0f, 0.0f,
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f
            }, tokenMajor.toFloatArray(), 0.0001f);
            assertArrayEquals(new float[] {
                    0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 6.0f,
                    0.0f, 0.0f, 3.0f, 4.0f, 7.0f, 8.0f
            }, headMajor.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void packsTokenMajorIntoHeadMajorLayout() {
        try (AccelTensor source = tensor(1.0f, 1, 2, 2, 2);
                AccelTensor destination = AccelTensor.zeros(1, 2, 3, 2)) {
            SharedKvBufferOps.packTokenMajorToHeadMajor(source, destination, 2, 2, 2);

            assertArrayEquals(new float[] {
                    1.0f, 2.0f, 5.0f, 6.0f, 0.0f, 0.0f,
                    3.0f, 4.0f, 7.0f, 8.0f, 0.0f, 0.0f
            }, destination.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void growsBuffersWhilePreservingActiveTokensOnly() {
        AccelTensor tokenMajor = tensor(1.0f, 1, 2, 2, 2);
        AccelTensor grownTokenMajor = SharedKvBufferOps.growTokenMajor(tokenMajor, 4, 2, 2, 2);
        try {
            assertTrue(tokenMajor.isClosed());
            assertArrayEquals(new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f
            }, grownTokenMajor.toFloatArray(), 0.0001f);
        } finally {
            grownTokenMajor.close();
        }

        AccelTensor headMajor = AccelTensor.fromFloatArray(new float[] {
                1.0f, 2.0f, 5.0f, 6.0f,
                3.0f, 4.0f, 7.0f, 8.0f
        }, 1, 2, 2, 2);
        AccelTensor grownHeadMajor = SharedKvBufferOps.growHeadMajor(headMajor, 4, 2, 2, 2);
        try {
            assertTrue(headMajor.isClosed());
            assertArrayEquals(new float[] {
                    1.0f, 2.0f, 5.0f, 6.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    3.0f, 4.0f, 7.0f, 8.0f,
                    0.0f, 0.0f, 0.0f, 0.0f
            }, grownHeadMajor.toFloatArray(), 0.0001f);
        } finally {
            grownHeadMajor.close();
        }
    }

    private static AccelTensor tensor(float start, long... shape) {
        AccelTensor tensor = AccelTensor.zeros(shape);
        for (int i = 0; i < tensor.numel(); i++) {
            tensor.setFlat(i, start + i);
        }
        return tensor;
    }
}
