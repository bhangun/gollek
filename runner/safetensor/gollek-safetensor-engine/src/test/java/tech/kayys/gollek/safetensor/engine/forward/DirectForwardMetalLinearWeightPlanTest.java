/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

class DirectForwardMetalLinearWeightPlanTest {
    private static final String PROFILE_KEY = "ffn_gate_up_pair";

    @Test
    void f16SinglePlanKeepsWeightAndReportsUniformHalf() {
        AccelTensor weight = f16(2, 3);

        try {
            DirectForwardMetalLinearWeightPlan plan =
                    DirectForwardMetalLinearWeightPlan.single(generic(), PROFILE_KEY, false, 1, weight);

            assertTrue(plan.completeSingle());
            assertFalse(plan.completePair());
            assertFalse(plan.nativeBf16Weights());
            assertTrue(plan.hasUniformHalfOrBf16Weights());
            assertSame(weight, plan.weight());
        } finally {
            weight.close();
        }
    }

    @Test
    void nativeBf16PairPlanKeepsNativeBf16Weights() {
        AccelTensor first = bf16(2, 3);
        AccelTensor second = bf16(2, 3);

        try {
            DirectForwardMetalLinearWeightPlan plan =
                    DirectForwardMetalLinearWeightPlan.pair(gemma4(), PROFILE_KEY, false, 1, first, second);

            assertTrue(plan.completePair());
            assertTrue(plan.nativeBf16Weights());
            assertTrue(plan.hasUniformHalfOrBf16Weights());
            assertTrue(plan.hasUniformPairHalfOrBf16Weights());
            assertSame(first, plan.firstWeight());
            assertSame(second, plan.secondWeight());
        } finally {
            first.close();
            second.close();
        }
    }

    @Test
    void pairPlanRejectsUnconvertedMixedHalfWeights() {
        AccelTensor first = f16(2, 3);
        AccelTensor second = bf16(2, 3);

        try {
            DirectForwardMetalLinearWeightPlan plan =
                    DirectForwardMetalLinearWeightPlan.pair(gemma4(), first, second, false, false);

            assertFalse(plan.completePair());
            assertFalse(plan.hasUniformPairHalfOrBf16Weights());
        } finally {
            first.close();
            second.close();
        }
    }

    @Test
    void forcedHalfConversionUsesBf16ToF16Allowance() {
        AccelTensor weight = bf16(2, 3);

        try {
            AccelTensor converted = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                    weight, gemma4(), false, true);

            assertNotNull(converted);
            assertTrue(converted.quantType() == AccelTensor.QuantType.F16);
        } finally {
            weight.close();
        }
    }

    private static AccelTensor f16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);
    }

    private static AccelTensor bf16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.BF16, null, null, -1);
    }

    private static ModelConfigTraits generic() {
        return traits(false);
    }

    private static ModelConfigTraits gemma4() {
        return traits(true);
    }

    private static ModelConfigTraits traits(boolean gemma4Text) {
        return new ModelConfigTraits(null, gemma4Text ? "gemma4_text" : "llama",
                0, 0, gemma4Text, false, false, false, false, false, false);
    }
}
