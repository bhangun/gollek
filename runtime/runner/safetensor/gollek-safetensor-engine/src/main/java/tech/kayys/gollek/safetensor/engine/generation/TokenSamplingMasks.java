/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.BitSet;

record TokenSamplingMasks(BitSet firstStep, BitSet continuation) {
    BitSet maskFor(boolean isFirstStep) {
        BitSet mask = isFirstStep ? firstStep : continuation;
        return mask == null ? new BitSet() : mask;
    }
}
