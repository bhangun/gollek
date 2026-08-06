/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

final class DirectForwardExecutionOptions {
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final boolean FORCE_CPU_FORWARD_ENABLED =
            Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY);
    private static final String REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY =
            "gollek.safetensor.reuse_ffn_projection_workspace";
    private static final String DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY =
            "gollek.safetensor.disable_ffn_projection_workspace_reuse";
    private static final boolean DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_ENABLED =
            Boolean.getBoolean(DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY);
    private static final String REUSE_FFN_PROJECTION_WORKSPACE_VALUE =
            System.getProperty(REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY);
    private static final Boolean REUSE_FFN_PROJECTION_WORKSPACE_EXPLICIT =
            parseOptionalBoolean(REUSE_FFN_PROJECTION_WORKSPACE_VALUE);
    private static final String VERBOSE_TOKENS_PROPERTY = "gollek.verbose";
    private static final String VERBOSE_LAYERS_PROPERTY = "gollek.verbose.layers";
    private static final boolean VERBOSE_TOKENS_ENABLED = Boolean.getBoolean(VERBOSE_TOKENS_PROPERTY);
    private static final boolean VERBOSE_LAYERS_ENABLED = Boolean.getBoolean(VERBOSE_LAYERS_PROPERTY);

    private DirectForwardExecutionOptions() {
    }

    static boolean forceCpuForwardEnabled() {
        return FORCE_CPU_FORWARD_ENABLED;
    }

    static boolean verboseTokensEnabled() {
        return VERBOSE_TOKENS_ENABLED;
    }

    static boolean verboseLayersEnabled() {
        return VERBOSE_LAYERS_ENABLED;
    }

    static boolean canReuseFfnProjectionWorkspace() {
        if (DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_ENABLED) {
            return false;
        }
        if (REUSE_FFN_PROJECTION_WORKSPACE_EXPLICIT != null) {
            return REUSE_FFN_PROJECTION_WORKSPACE_EXPLICIT;
        }
        return true;
    }
}
