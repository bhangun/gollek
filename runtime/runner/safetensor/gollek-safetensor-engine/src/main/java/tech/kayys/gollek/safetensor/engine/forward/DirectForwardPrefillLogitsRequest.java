/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

record DirectForwardPrefillLogitsRequest(
        DirectForwardOutputProjectionContext context,
        MemorySegment hiddenSegment,
        long[] hiddenShape,
        AccelTensor embeddings,
        int seqLen,
        boolean verboseTokens) {
}
