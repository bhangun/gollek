/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

record DirectForwardLinearCacheOptions(
        long metalF16WeightCacheMaxBytes,
        long logitsLargeHalfCacheMaxBytes,
        long ffnDownLargeHalfCacheTotalMaxBytes,
        long ffnDownLargeHalfCachePerTensorMaxBytes) {

    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 8L * 1024L * 1024L * 1024L;
    private static final String LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.logits_large_half_cache_max_bytes";
    private static final long DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final String FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_total_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES = 1536L * 1024L * 1024L;
    private static final String FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_per_tensor_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES = 64L * 1024L * 1024L;

    static DirectForwardLinearCacheOptions fromSystemProperties() {
        return new DirectForwardLinearCacheOptions(
                Long.getLong(METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
                        DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES),
                Long.getLong(LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY,
                        DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES),
                Long.getLong(FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY,
                        DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES),
                Long.getLong(FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY,
                        DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES));
    }

    static DirectForwardLinearCacheOptions defaults() {
        return new DirectForwardLinearCacheOptions(
                DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES,
                DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES,
                DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES,
                DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES);
    }

    DirectForwardLinearCacheOptions withMetalF16WeightCacheMaxBytes(long maxBytes) {
        return new DirectForwardLinearCacheOptions(
                maxBytes,
                logitsLargeHalfCacheMaxBytes,
                ffnDownLargeHalfCacheTotalMaxBytes,
                ffnDownLargeHalfCachePerTensorMaxBytes);
    }

    DirectForwardLinearCacheOptions withLogitsLargeHalfCacheMaxBytes(long maxBytes) {
        return new DirectForwardLinearCacheOptions(
                metalF16WeightCacheMaxBytes,
                maxBytes,
                ffnDownLargeHalfCacheTotalMaxBytes,
                ffnDownLargeHalfCachePerTensorMaxBytes);
    }

    DirectForwardLinearCacheOptions withFfnDownLargeHalfCacheBudgets(
            long totalMaxBytes,
            long perTensorMaxBytes) {
        return new DirectForwardLinearCacheOptions(
                metalF16WeightCacheMaxBytes,
                logitsLargeHalfCacheMaxBytes,
                totalMaxBytes,
                perTensorMaxBytes);
    }
}
