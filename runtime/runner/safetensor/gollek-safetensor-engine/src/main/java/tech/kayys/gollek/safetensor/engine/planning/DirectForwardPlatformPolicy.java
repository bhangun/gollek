/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

final class DirectForwardPlatformPolicy {
    static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    static final String FORCE_CPU_GEMMA4_PROPERTY = "gollek.safetensor.force_cpu_gemma4";
    static final String ALLOW_METAL_GEMMA4_PROPERTY = "gollek.safetensor.allow_metal_gemma4";

    private DirectForwardPlatformPolicy() {
    }

    static Decision apply(ModelRuntimeTraits runtimeTraits, String modelLabel) {
        Decision decision = decide(runtimeTraits, modelLabel);
        if (decision.forceCpuForward()) {
            System.setProperty(FORCE_CPU_FORWARD_PROPERTY, "true");
        } else {
            System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
        }
        return decision;
    }

    static Decision decide(ModelRuntimeTraits runtimeTraits, String modelLabel) {
        String label = modelLabel == null || modelLabel.isBlank() ? "unknown" : modelLabel;
        if (requiresCpuForward(runtimeTraits)) {
            return new Decision(true, false,
                    "forcing CPU for " + label + " because " + FORCE_CPU_GEMMA4_PROPERTY + "=true");
        }
        if (allowsExperimentalMetal(runtimeTraits)) {
            return new Decision(false, true, "allowing Metal for Gemma4 experimental validation");
        }
        return new Decision(false, false, "using default forward platform policy");
    }

    static boolean requiresCpuForward(ModelRuntimeTraits runtimeTraits) {
        return isGemma4(runtimeTraits) && Boolean.getBoolean(FORCE_CPU_GEMMA4_PROPERTY);
    }

    private static boolean allowsExperimentalMetal(ModelRuntimeTraits runtimeTraits) {
        return isGemma4(runtimeTraits) && Boolean.getBoolean(ALLOW_METAL_GEMMA4_PROPERTY);
    }

    private static boolean isGemma4(ModelRuntimeTraits runtimeTraits) {
        return runtimeTraits != null && runtimeTraits.nativeBf16Matvec();
    }

    record Decision(boolean forceCpuForward, boolean experimentalMetalAllowed, String reason) {
    }
}
