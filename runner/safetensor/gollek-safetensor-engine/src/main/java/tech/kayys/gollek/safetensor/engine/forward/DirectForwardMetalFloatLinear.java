/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

final class DirectForwardMetalFloatLinear {
    private DirectForwardMetalFloatLinear() {
    }

    static AccelTensor tryLinear(Logger log,
            MetalBinding metalBinding,
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor weight,
            AccelTensor bias) {
        if (!canUseCandidate(metalLinearEnabled, metalBinding, input, weight)) {
            return null;
        }
        DirectForwardMetalLinearShapePlan shapePlan = DirectForwardMetalLinearShapePlan.single(input, weight);
        if (shapePlan == null) {
            return null;
        }
        AccelTensor out = AccelTensor.zeros(shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int m = Math.toIntExact(shapePlan.rows());
            int kk = Math.toIntExact(shapePlan.inputDim());
            int n = Math.toIntExact(shapePlan.outputDim());
            int rc = metalBinding.matmulTransposedRight(
                    out.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weight.dataPtr(),
                    m, kk, n,
                    1.0f, 0.0f);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRight failed with code " + rc);
            }
            if (bias == null) {
                return out;
            }
            AccelTensor biased = AccelOps.add(out, bias);
            out.close();
            return biased;
        } catch (RuntimeException e) {
            out.close();
            log.debugf("Falling back from Metal float linear to AccelOps: %s", e.getMessage());
            return null;
        }
    }

    // MPS hangs indefinitely on very large F32 matrices (e.g. lm_head for
    // 152k-vocab models).
    // Cap at 256M weight elements (~1 GB in F32) to prevent a hard hang.
    private static final long MAX_METAL_FLOAT_WEIGHT_ELEMENTS = 256L * 1024L * 1024L;

    private static boolean canUseCandidate(boolean metalLinearEnabled,
            MetalBinding metalBinding,
            AccelTensor input,
            AccelTensor weight) {
        if (!metalLinearEnabled || metalBinding == null) {
            return false;
        }
        if (input == null || weight == null) {
            return false;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        int inputRank = input.rank();
        if (inputRank < 2 || weight.rank() != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, input.size(-1));
        if (rows <= 0L) {
            return false;
        }
        // Guard against oversized weight tensors that cause MPS to hang.
        if (weight.numel() > MAX_METAL_FLOAT_WEIGHT_ELEMENTS) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputRank - 2; i++) {
            batchProduct *= input.size(i);
        }
        return batchProduct == 1L;
    }
}
