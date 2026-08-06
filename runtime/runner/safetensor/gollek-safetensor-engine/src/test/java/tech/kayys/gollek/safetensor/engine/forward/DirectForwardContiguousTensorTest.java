/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardContiguousTensorTest {

    @Test
    void contiguousInputIsBorrowedAndNotClosedByLease() {
        AccelTensor input = AccelTensor.zeros(2, 3);

        try (DirectForwardContiguousTensor lease = DirectForwardContiguousTensor.from(input)) {
            assertSame(input, lease.tensor());
            assertFalse(lease.ownsTemporary());
        } finally {
            assertFalse(input.isClosed());
            input.close();
        }
    }

    @Test
    void offsetViewIsMaterializedAndTemporaryIsClosedByLease() {
        AccelTensor input = AccelTensor.zeros(2, 3);
        AccelTensor view = input.slice(0, 1, 2);
        AccelTensor materialized;

        try (DirectForwardContiguousTensor lease = DirectForwardContiguousTensor.from(view)) {
            materialized = lease.tensor();
            assertNotSame(view, materialized);
            assertTrue(lease.ownsTemporary());
            assertTrue(materialized.isContiguous());
        }

        try {
            assertTrue(materialized.isClosed());
            assertFalse(view.isClosed());
            assertFalse(input.isClosed());
        } finally {
            view.close();
            input.close();
        }
    }

    @Test
    void nullInputIsRejectedAtBoundary() {
        assertThrows(NullPointerException.class, () -> DirectForwardContiguousTensor.from(null));
    }
}
