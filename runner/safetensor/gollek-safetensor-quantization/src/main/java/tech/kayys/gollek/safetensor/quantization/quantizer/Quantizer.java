/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * Quantizer.java
 * ───────────────────────
 * Quantizer interface.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;

/**
 * Interface for quantization implementations.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public interface Quantizer {

    /**
     * Quantize a tensor.
     *
     * @param tensor input tensor to quantize
     * @param config quantization configuration
     * @return quantized tensor
     */
    TorchTensor quantizeTensor(TorchTensor tensor, QuantConfig config);

    /**
     * Dequantize a tensor.
     *
     * @param quantizedTensor quantized tensor
     * @param config          quantization configuration used
     * @return dequantized tensor
     */
    TorchTensor dequantizeTensor(TorchTensor quantizedTensor, QuantConfig config);

    /**
     * Get the quantizer name.
     *
     * @return quantizer name
     */
    default String getName() {
        return getClass().getSimpleName();
    }

    /**
     * Check if this quantizer supports a given configuration.
     *
     * @param config configuration to check
     * @return true if supported
     */
    default boolean supports(QuantConfig config) {
        return true;
    }
}
