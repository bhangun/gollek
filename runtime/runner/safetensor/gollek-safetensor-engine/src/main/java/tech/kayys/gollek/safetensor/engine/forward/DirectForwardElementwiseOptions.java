/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardElementwiseOptions(
        boolean enableNativeBf16MetalElementwise,
        boolean disableNativeBf16MetalElementwise,
        int metalElementwiseMinSeq,
        boolean disablePerLayerInputEmbedding,
        boolean disableNativeBf16LayerScalar,
        boolean enableMetalLayerScalarDecode,
        int metalLayerScalarMinSeq,
        Boolean enableMetalPostFfnNorm,
        boolean disableMetalPostFfnNorm) {

    private static final String ENABLE_NATIVE_BF16_METAL_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.enable_native_bf16_metal_elementwise";
    private static final String DISABLE_NATIVE_BF16_METAL_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.disable_native_bf16_metal_elementwise";
    private static final String METAL_ELEMENTWISE_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_elementwise_min_seq";
    private static final String DISABLE_PER_LAYER_INPUT_EMBEDDING_PROPERTY =
            "gollek.safetensor.disable_per_layer_input_embedding";
    private static final String DISABLE_NATIVE_BF16_LAYER_SCALAR_PROPERTY =
            "gollek.safetensor.disable_native_bf16_layer_scalar";
    private static final String ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY =
            "gollek.safetensor.enable_metal_layer_scalar_decode";
    private static final String METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_layer_scalar_min_seq";
    private static final int DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ = 64;
    private static final String ENABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_post_ffn_norm";
    private static final String DISABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_post_ffn_norm";

    static DirectForwardElementwiseOptions fromSystemProperties() {
        return new DirectForwardElementwiseOptions(
                Boolean.getBoolean(ENABLE_NATIVE_BF16_METAL_ELEMENTWISE_PROPERTY),
                Boolean.getBoolean(DISABLE_NATIVE_BF16_METAL_ELEMENTWISE_PROPERTY),
                Integer.getInteger(METAL_ELEMENTWISE_MIN_SEQ_PROPERTY, -1),
                Boolean.getBoolean(DISABLE_PER_LAYER_INPUT_EMBEDDING_PROPERTY),
                Boolean.getBoolean(DISABLE_NATIVE_BF16_LAYER_SCALAR_PROPERTY),
                Boolean.getBoolean(ENABLE_METAL_LAYER_SCALAR_DECODE_PROPERTY),
                Integer.getInteger(METAL_LAYER_SCALAR_MIN_SEQ_PROPERTY, DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_POST_FFN_NORM_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_POST_FFN_NORM_PROPERTY));
    }

    static DirectForwardElementwiseOptions defaults() {
        return new DirectForwardElementwiseOptions(
                false,
                false,
                -1,
                false,
                false,
                false,
                DEFAULT_METAL_LAYER_SCALAR_MIN_SEQ,
                null,
                false);
    }

    DirectForwardElementwiseOptions withMetalElementwise(
            boolean enableNativeBf16, boolean disableNativeBf16, int minSeq) {
        return new DirectForwardElementwiseOptions(
                enableNativeBf16,
                disableNativeBf16,
                minSeq,
                disablePerLayerInputEmbedding,
                disableNativeBf16LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withPerLayerInputDisabled(boolean disabled) {
        return new DirectForwardElementwiseOptions(
                enableNativeBf16MetalElementwise,
                disableNativeBf16MetalElementwise,
                metalElementwiseMinSeq,
                disabled,
                disableNativeBf16LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withLayerScalar(boolean disableNativeBf16, boolean enableDecode, int minSeq) {
        return new DirectForwardElementwiseOptions(
                enableNativeBf16MetalElementwise,
                disableNativeBf16MetalElementwise,
                metalElementwiseMinSeq,
                disablePerLayerInputEmbedding,
                disableNativeBf16,
                enableDecode,
                minSeq,
                enableMetalPostFfnNorm,
                disableMetalPostFfnNorm);
    }

    DirectForwardElementwiseOptions withPostFfnNorm(Boolean enable, boolean disable) {
        return new DirectForwardElementwiseOptions(
                enableNativeBf16MetalElementwise,
                disableNativeBf16MetalElementwise,
                metalElementwiseMinSeq,
                disablePerLayerInputEmbedding,
                disableNativeBf16LayerScalar,
                enableMetalLayerScalarDecode,
                metalLayerScalarMinSeq,
                enable,
                disable);
    }
}
