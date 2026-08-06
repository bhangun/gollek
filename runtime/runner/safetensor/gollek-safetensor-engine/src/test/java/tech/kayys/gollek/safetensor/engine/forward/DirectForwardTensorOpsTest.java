/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

/**
 * Verifies low-level tensor view helpers used by direct forward fast paths.
 */
class DirectForwardTensorOpsTest {

    @Test
    void linearRowViewFlattensLeadingDimensionsWithoutCopying() {
        AccelTensor tensor = AccelTensor.fromFloatArray(
                new float[] { 1f, 2f, 3f, 4f, 5f, 6f },
                1L,
                2L,
                3L);

        try {
            AccelTensor secondRow = DirectForwardTensorOps.linearRowView(tensor, 1L);
            try {
                assertArrayEquals(new float[] { 4f, 5f, 6f }, secondRow.toFloatArray());
                secondRow.setFlat(1L, 9f);
            } finally {
                secondRow.close();
            }

            assertFalse(tensor.isClosed());
            assertArrayEquals(new float[] { 1f, 2f, 3f, 4f, 9f, 6f }, tensor.toFloatArray());
        } finally {
            tensor.close();
        }
    }

    @Test
    void packedF32LinearInputRejectsQuantizedViews() {
        AccelTensor tensor = AccelTensor.zeros(2L, 3L)
                .withQuantization(AccelTensor.QuantType.BF16, null, null, -1);

        try {
            assertFalse(DirectForwardTensorOps.isPackedF32LinearInput(tensor));
        } finally {
            tensor.close();
        }
    }

    @Test
    void packedF32LinearInputAcceptsContiguousFloatRows() {
        AccelTensor tensor = AccelTensor.zeros(2L, 3L);

        try {
            assertTrue(DirectForwardTensorOps.isPackedF32LinearInput(tensor));
        } finally {
            tensor.close();
        }
    }
}
