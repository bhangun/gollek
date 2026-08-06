/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class DirectForwardNorms {
    private DirectForwardNorms() {
    }

    static void rmsNormRows(MemorySegment out,
                            MemorySegment in,
                            AccelTensor weight,
                            long[] shape,
                            int seqLen,
                            ModelConfig config,
                            boolean addOne,
                            boolean useMetalElementwise,
                            DirectForwardRuntimeContext runtime) {
        if (useMetalElementwise) {
            DirectForwardElementwiseOps.rmsNormRowsMetal(runtime.metalBinding(), out, in,
                    weight.dataPtr(), seqLen, config.getHiddenSize(), (float) config.getRmsNormEps(), addOne);
            return;
        }

        AccelTensor inTensor = AccelTensor.view(in, shape);
        AccelTensor outTensor = AccelOps.rmsNorm(inTensor, weight, config.getRmsNormEps(), addOne);
        MemorySegment.copy(outTensor.dataPtr(), 0, out, 0,
                (long) seqLen * config.getHiddenSize() * Float.BYTES);
        outTensor.close();
    }
}
