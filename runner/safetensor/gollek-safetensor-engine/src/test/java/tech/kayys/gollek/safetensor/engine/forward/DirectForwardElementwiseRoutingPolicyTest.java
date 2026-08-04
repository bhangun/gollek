/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardElementwiseRoutingPolicyTest {

    @Test
    void forceCpuBlocksAllMetalElementwise() {
        assertFalse(policy(DirectForwardElementwiseOptions.defaults())
                .canUseMetalElementwise(generic(), 128, true, true, true, true));
    }

    @Test
    void genericElementwiseUsesDefaultMinSeqAndFallbackCapability() {
        DirectForwardElementwiseRoutingPolicy policy = policy(DirectForwardElementwiseOptions.defaults());

        assertFalse(policy.canUseMetalElementwise(generic(), 15, false, true, false, false));
        assertTrue(policy.canUseMetalElementwise(generic(), 16, false, true, false, false));
        assertFalse(policy.canUseMetalElementwise(generic(), 16, false, false, false, false));
        assertTrue(policy.canUseMetalElementwise(generic(), 16, false, false, false, true));
    }

    @Test
    void elementwiseMinSeqOverrideAppliesToGenericAndNativeBf16() {
        DirectForwardElementwiseRoutingPolicy policy = policy(
                DirectForwardElementwiseOptions.defaults().withMetalElementwise(false, false, 4));

        assertFalse(policy.canUseMetalElementwise(generic(), 3, false, true, false, false));
        assertTrue(policy.canUseMetalElementwise(generic(), 4, false, true, false, false));
        assertFalse(policy.canUseMetalElementwise(gemma4(), 3, false, true, true, false));
        assertTrue(policy.canUseMetalElementwise(gemma4(), 4, false, true, true, false));
    }

    @Test
    void gemma4ElementwiseRequiresMetalAndNativeKernelAndCanBeDisabled() {
        DirectForwardElementwiseOptions options = DirectForwardElementwiseOptions.defaults();
        DirectForwardElementwiseRoutingPolicy policy = policy(options);

        assertFalse(policy.canUseMetalElementwise(gemma4(), 1, false, false, true, true));
        assertFalse(policy.canUseMetalElementwise(gemma4(), 1, false, true, false, true));
        assertTrue(policy.canUseMetalElementwise(gemma4(), 1, false, true, true, false));
        assertFalse(policy(options.withMetalElementwise(false, true, -1))
                .canUseMetalElementwise(gemma4(), 1, false, true, true, false));
    }

    @Test
    void layerScalarRequiresElementwiseMetalBindingAndKernel() {
        DirectForwardElementwiseRoutingPolicy policy = policy(
                DirectForwardElementwiseOptions.defaults().withLayerScalar(false, false, 4));

        assertFalse(policy.canUseMetalLayerScalarScale(false, 4, true, true));
        assertFalse(policy.canUseMetalLayerScalarScale(true, 4, false, true));
        assertFalse(policy.canUseMetalLayerScalarScale(true, 4, true, false));
        assertFalse(policy.canUseMetalLayerScalarScale(true, 3, true, true));
        assertTrue(policy.canUseMetalLayerScalarScale(true, 4, true, true));
    }

    @Test
    void layerScalarDecodeIsExplicitlyControlled() {
        assertFalse(policy(DirectForwardElementwiseOptions.defaults()
                .withLayerScalar(false, false, 64))
                .canUseMetalLayerScalarScale(true, 1, true, true));
        assertTrue(policy(DirectForwardElementwiseOptions.defaults()
                .withLayerScalar(false, true, 64))
                .canUseMetalLayerScalarScale(true, 1, true, true));
    }

    @Test
    void postFfnNormDefaultsToNativeBf16FamilyAndHonorsOverrides() {
        DirectForwardElementwiseOptions options = DirectForwardElementwiseOptions.defaults();
        DirectForwardElementwiseRoutingPolicy policy = policy(options);

        assertFalse(policy.shouldUseMetalPostFfnNorm(generic()));
        assertTrue(policy.shouldUseMetalPostFfnNorm(gemma4()));
        assertTrue(policy.shouldUseMetalPostFfnNorm(perLayerStyle()));
        assertTrue(policy(options.withPostFfnNorm(true, false)).shouldUseMetalPostFfnNorm(generic()));
        assertFalse(policy(options.withPostFfnNorm(true, true)).shouldUseMetalPostFfnNorm(gemma4()));
    }

    @Test
    void perLayerInputsRequireHiddenSizeAndNativeBf16DisableOnlyAffectsNativeBf16Text() {
        DirectForwardElementwiseRoutingPolicy policy = policy(
                DirectForwardElementwiseOptions.defaults().withPerLayerInputDisabled(true));

        assertFalse(policy.shouldBuildPerLayerInputs(gemma4(), 0));
        assertFalse(policy.shouldBuildPerLayerInputs(gemma4(), 128));
        assertTrue(policy.shouldBuildPerLayerInputs(perLayerStyle(), 128));
        assertTrue(policy.shouldBuildPerLayerInputs(generic(), 128));
    }

    @Test
    void layerScalarDisableOnlyAffectsNativeBf16Text() {
        DirectForwardElementwiseRoutingPolicy policy = policy(
                DirectForwardElementwiseOptions.defaults().withLayerScalar(true, false, 64));

        assertTrue(policy.shouldApplyLayerScalar(generic()));
        assertFalse(policy.shouldApplyLayerScalar(gemma4()));
        assertTrue(policy.shouldApplyLayerScalar(perLayerStyle()));
    }

    private static DirectForwardElementwiseRoutingPolicy policy(DirectForwardElementwiseOptions options) {
        return DirectForwardElementwiseRoutingPolicy.from(options);
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false, false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true, false);
    }

    private static ModelConfigTraits perLayerStyle() {
        return traits("custom_per_layer", false, true);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean gemma4Text,
            boolean gemma4StylePerLayerInputs) {
        return new ModelConfigTraits(
                null, modelType, 0, 0, gemma4Text, false, false, gemma4StylePerLayerInputs, false, false, false);
    }
}
