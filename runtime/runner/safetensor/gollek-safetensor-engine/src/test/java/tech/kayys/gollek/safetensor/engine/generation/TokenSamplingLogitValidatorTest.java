/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenSamplingLogitValidatorTest {

    @Test
    void rejectsMissingLogitsBeforeOptionalValidation() {
        assertFalse(TokenSamplingLogitValidator.accepts(null));
        assertFalse(TokenSamplingLogitValidator.accepts(new float[0]));
    }

    @Test
    void acceptsCleanLogits() {
        assertTrue(TokenSamplingLogitValidator.validate(new float[] { -1.0f, 0.0f, 3.5f }));
    }

    @Test
    void allowsSmallNumberOfNonFiniteLogits() {
        float[] logits = {
                Float.NaN, 0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        };

        assertTrue(TokenSamplingLogitValidator.validate(logits));
    }

    @Test
    void rejectsTooManyNanLogits() {
        float[] logits = {
                Float.NaN, Float.NaN, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        };

        assertFalse(TokenSamplingLogitValidator.validate(logits));
    }

    @Test
    void rejectsTooManyInfiniteLogits() {
        float[] logits = {
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        };

        assertFalse(TokenSamplingLogitValidator.validate(logits));
    }
}
