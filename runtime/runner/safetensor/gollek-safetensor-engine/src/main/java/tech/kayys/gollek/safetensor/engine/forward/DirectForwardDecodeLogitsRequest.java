/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.lang.foreign.MemorySegment;

record DirectForwardDecodeLogitsRequest(
        DirectForwardOutputProjectionContext context,
        MemorySegment hiddenSegment,
        long[] hiddenShape,
        boolean reuseLogitsOutput) {
}
