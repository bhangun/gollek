/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionStageOptions(
        FlashAttentionRoutingOptions routingOptions,
        FlashAttentionRopeOptions ropeOptions,
        FlashAttentionNormalizerOptions normalizerOptions,
        FlashAttentionNormalizationOptions normalizationOptions,
        FlashAttentionPrecisionOptions precisionOptions,
        PagedAttentionVectorOptions pagedAttentionOptions,
        FlashAttentionBackendOptions backendOptions,
        FlashAttentionLinearOptions linearOptions,
        FlashAttentionMatvecOptions matvecOptions) {

    FlashAttentionStageOptions(
            FlashAttentionRoutingOptions routingOptions,
            FlashAttentionRopeOptions ropeOptions,
            FlashAttentionNormalizerOptions normalizerOptions,
            FlashAttentionNormalizationOptions normalizationOptions) {
        this(routingOptions, ropeOptions, normalizerOptions, normalizationOptions, null, null, null, null, null);
    }

    FlashAttentionStageOptions(
            FlashAttentionRoutingOptions routingOptions,
            FlashAttentionRopeOptions ropeOptions,
            FlashAttentionNormalizerOptions normalizerOptions,
            FlashAttentionNormalizationOptions normalizationOptions,
            FlashAttentionPrecisionOptions precisionOptions) {
        this(routingOptions, ropeOptions, normalizerOptions, normalizationOptions, precisionOptions, null, null, null,
                null);
    }

    FlashAttentionStageOptions(
            FlashAttentionRoutingOptions routingOptions,
            FlashAttentionRopeOptions ropeOptions,
            FlashAttentionNormalizerOptions normalizerOptions,
            FlashAttentionNormalizationOptions normalizationOptions,
            FlashAttentionPrecisionOptions precisionOptions,
            PagedAttentionVectorOptions pagedAttentionOptions) {
        this(routingOptions, ropeOptions, normalizerOptions, normalizationOptions, precisionOptions,
                pagedAttentionOptions, null, null, null);
    }

    FlashAttentionStageOptions(
            FlashAttentionRoutingOptions routingOptions,
            FlashAttentionRopeOptions ropeOptions,
            FlashAttentionNormalizerOptions normalizerOptions,
            FlashAttentionNormalizationOptions normalizationOptions,
            FlashAttentionPrecisionOptions precisionOptions,
            PagedAttentionVectorOptions pagedAttentionOptions,
            FlashAttentionBackendOptions backendOptions) {
        this(routingOptions, ropeOptions, normalizerOptions, normalizationOptions, precisionOptions,
                pagedAttentionOptions, backendOptions, null, null);
    }

    FlashAttentionStageOptions(
            FlashAttentionRoutingOptions routingOptions,
            FlashAttentionRopeOptions ropeOptions,
            FlashAttentionNormalizerOptions normalizerOptions,
            FlashAttentionNormalizationOptions normalizationOptions,
            FlashAttentionPrecisionOptions precisionOptions,
            PagedAttentionVectorOptions pagedAttentionOptions,
            FlashAttentionBackendOptions backendOptions,
            FlashAttentionLinearOptions linearOptions) {
        this(routingOptions, ropeOptions, normalizerOptions, normalizationOptions, precisionOptions,
                pagedAttentionOptions, backendOptions, linearOptions, null);
    }

    FlashAttentionStageOptions {
        if (routingOptions == null) {
            routingOptions = FlashAttentionRoutingOptions.defaults();
        }
        if (ropeOptions == null) {
            ropeOptions = FlashAttentionRopeOptions.defaults();
        }
        if (normalizerOptions == null) {
            normalizerOptions = FlashAttentionNormalizerOptions.defaults();
        }
        if (normalizationOptions == null) {
            normalizationOptions = FlashAttentionNormalizationOptions.defaults();
        }
        if (precisionOptions == null) {
            precisionOptions = FlashAttentionPrecisionOptions.defaults();
        }
        if (pagedAttentionOptions == null) {
            pagedAttentionOptions = PagedAttentionVectorOptions.defaults();
        }
        if (backendOptions == null) {
            backendOptions = FlashAttentionBackendOptions.defaults();
        }
        if (linearOptions == null) {
            linearOptions = FlashAttentionLinearOptions.defaults();
        }
        if (matvecOptions == null) {
            matvecOptions = FlashAttentionMatvecOptions.defaults();
        }
    }

    static FlashAttentionStageOptions fromSystemPropertiesAndEnvironment() {
        return new FlashAttentionStageOptions(
                FlashAttentionRoutingOptions.fromSystemPropertiesAndEnvironment(),
                FlashAttentionRopeOptions.fromSystemProperties(),
                FlashAttentionNormalizerOptions.fromSystemProperties(),
                FlashAttentionNormalizationOptions.fromSystemProperties(),
                FlashAttentionPrecisionOptions.fromSystemProperties(),
                PagedAttentionVectorOptions.fromSystemProperties(),
                FlashAttentionBackendOptions.fromSystemProperties(),
                FlashAttentionLinearOptions.fromSystemProperties(),
                FlashAttentionMatvecOptions.fromSystemProperties());
    }

    static FlashAttentionStageOptions defaults() {
        return new FlashAttentionStageOptions(null, null, null, null, null, null, null, null, null);
    }
}
