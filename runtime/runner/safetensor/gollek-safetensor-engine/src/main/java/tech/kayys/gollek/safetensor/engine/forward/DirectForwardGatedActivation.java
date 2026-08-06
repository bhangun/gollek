/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.fusedGeglu;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

final class DirectForwardGatedActivation {
    private DirectForwardGatedActivation() {
    }

    static AccelTensor combine(MetalBinding metalBinding,
            FFNActivationType activationType,
            boolean useMetalElementwise,
            AccelTensor gate,
            AccelTensor up,
            AccelTensor workspaceOutput,
            int elements) {
        if (activationType == FFNActivationType.GELU) {
            return combineGelu(metalBinding, useMetalElementwise, gate, up, workspaceOutput, elements);
        }
        return combineSilu(metalBinding, useMetalElementwise, gate, up, workspaceOutput, elements);
    }

    private static AccelTensor combineGelu(MetalBinding metalBinding,
            boolean useMetalElementwise,
            AccelTensor gate,
            AccelTensor up,
            AccelTensor workspaceOutput,
            int elements) {
        if (workspaceOutput == null) {
            AccelTensor activated = AccelOps.gelu(gate);
            try {
                return AccelOps.mul(activated, up);
            } finally {
                activated.close();
            }
        }

        if (useMetalElementwise) {
            try {
                int rc = metalBinding.geluFfn(workspaceOutput.dataPtr(), gate.dataPtr(), up.dataPtr(), elements);
                if (rc != 0) {
                    throw new IllegalStateException("Metal geluFfn failed with code " + rc);
                }
                return workspaceOutput;
            } catch (IllegalStateException | UnsupportedOperationException e) {
                fusedGeglu(workspaceOutput, gate, up);
                return workspaceOutput;
            }
        }

        fusedGeglu(workspaceOutput, gate, up);
        return workspaceOutput;
    }

    private static AccelTensor combineSilu(MetalBinding metalBinding,
            boolean useMetalElementwise,
            AccelTensor gate,
            AccelTensor up,
            AccelTensor workspaceOutput,
            int elements) {
        if (useMetalElementwise && workspaceOutput != null) {
            try {
                int rc = metalBinding.siluFfn(workspaceOutput.dataPtr(), gate.dataPtr(), up.dataPtr(), elements);
                if (rc != 0) {
                    throw new IllegalStateException("Metal siluFfn failed with code " + rc);
                }
                return workspaceOutput;
            } catch (IllegalStateException | UnsupportedOperationException e) {
                return AccelOps.swiglu(gate, up);
            }
        }
        return AccelOps.swiglu(gate, up);
    }
}
