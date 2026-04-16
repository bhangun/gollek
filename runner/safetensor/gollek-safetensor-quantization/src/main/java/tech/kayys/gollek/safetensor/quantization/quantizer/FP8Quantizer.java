/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP8Quantizer.java
 * ───────────────────────
 * FP8 (8-bit floating point) quantization implementation.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;

/**
 * FP8 quantizer supporting E4M3 and E5M2 formats.
 * <p>
 * FP8 quantization is designed for hardware with FP8 tensor core support
 * (NVIDIA H100, AMD MI300, etc.). It provides better dynamic range than
 * INT8 while maintaining the same memory footprint.
 * </p>
 * <p>
 * Supported formats:
 * <ul>
 * <li>E4M3: 4 exponent bits, 3 mantissa bits - better for inference</li>
 * <li>E5M2: 5 exponent bits, 2 mantissa bits - better dynamic range</li>
 * </ul>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class FP8Quantizer implements Quantizer {

    private static final Logger log = Logger.getLogger(FP8Quantizer.class);

    /**
     * FP8 format types.
     */
    public enum FP8Format {
        /**
         * E4M3 format - 4 exponent, 3 mantissa bits.
         * Better precision in normal range, recommended for inference.
         */
        E4M3(4, 3, 0b01111000, 240),

        /**
         * E5M2 format - 5 exponent, 2 mantissa bits.
         * Better dynamic range, similar to FP16.
         */
        E5M2(5, 2, 0b01111000, 30);

        private final int expBits;
        private final int mantissaBits;
        private final int expBias;
        private final int finiteValue;

        FP8Format(int expBits, int mantissaBits, int expBias, int finiteValue) {
            this.expBits = expBits;
            this.mantissaBits = mantissaBits;
            this.expBias = expBias;
            this.finiteValue = finiteValue;
        }

        public int getExpBits() {
            return expBits;
        }

        public int getMantissaBits() {
            return mantissaBits;
        }

        public int getExpBias() {
            return expBias;
        }

        public int getFiniteValue() {
            return finiteValue;
        }
    }

    /**
     * Default FP8 format (E4M3 for inference).
     */
    private static final FP8Format DEFAULT_FORMAT = FP8Format.E4M3;

    /**
     * Quantize a tensor using FP8.
     *
     * @param tensor input tensor to quantize
     * @param config quantization configuration
     * @return quantized tensor
     */
    @Override
    public TorchTensor quantizeTensor(TorchTensor tensor, QuantConfig config) {
        if (tensor == null) {
            throw new IllegalArgumentException("TorchTensor cannot be null");
        }

        log.debugf("FP8 quantizing tensor (symmetric=%b)", config.isSymmetric());

        try {
            float[] data = getTensorData(tensor);
            int[] shape = getTensorShape(tensor);

            // Determine FP8 format based on config
            FP8Format format = selectFormat(config);

            // Quantize to FP8
            byte[] quantizedData = quantizeToFP8(data, format);

            // Calculate scale for per-tensor quantization
            float[] scales = new float[] { calculateScale(data, format) };

            return createQuantizedTensor(quantizedData, scales, shape, config, format);

        } catch (Exception e) {
            log.errorf(e, "FP8 quantization failed");
            throw new RuntimeException("FP8 quantization failed", e);
        }
    }

    /**
     * Dequantize an FP8 tensor.
     *
     * @param quantizedTensor quantized tensor
     * @param config          quantization configuration
     * @return dequantized tensor
     */
    @Override
    public TorchTensor dequantizeTensor(TorchTensor quantizedTensor, QuantConfig config) {
        if (quantizedTensor == null) {
            throw new IllegalArgumentException("TorchTensor cannot be null");
        }

        log.debugf("Dequantizing FP8 tensor");

        try {
            byte[] quantizedData = getQuantizedData(quantizedTensor);
            float[] scales = getScales(quantizedTensor);
            int[] shape = getTensorShape(quantizedTensor);
            FP8Format format = getFormat(quantizedTensor);

            float[] dequantizedData = new float[quantizedData.length];
            float scale = scales[0];

            for (int i = 0; i < quantizedData.length; i++) {
                dequantizedData[i] = dequantizeFP8(quantizedData[i], format) * scale;
            }

            return createDequantizedTensor(dequantizedData, shape);

        } catch (Exception e) {
            log.errorf(e, "FP8 dequantization failed");
            throw new RuntimeException("FP8 dequantization failed", e);
        }
    }

    @Override
    public String getName() {
        return "FP8Quantizer";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == QuantizationEngine.QuantStrategy.FP8;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FP8 Quantization Implementation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Select FP8 format based on configuration.
     *
     * @param config quantization configuration
     * @return selected FP8 format
     */
    private FP8Format selectFormat(QuantConfig config) {
        // Use symmetric flag to choose format
        // E4M3 for symmetric (inference), E5M2 for asymmetric (training/dynamic)
        return config.isSymmetric() ? FP8Format.E4M3 : FP8Format.E5M2;
    }

    /**
     * Quantize float array to FP8.
     *
     * @param data   input float data
     * @param format FP8 format
     * @return quantized byte array
     */
    private byte[] quantizeToFP8(float[] data, FP8Format format) {
        byte[] result = new byte[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = floatToFP8(data[i], format);
        }

        return result;
    }

    /**
     * Convert float to FP8 representation.
     *
     * @param value  float value
     * @param format FP8 format
     * @return FP8 byte representation
     */
    private byte floatToFP8(float value, FP8Format format) {
        // Handle special cases
        if (Float.isNaN(value)) {
            return (byte) 0x80; // NaN
        }
        if (value == 0.0f) {
            return 0x00; // +0
        }

        // Get sign bit
        int sign = (value < 0) ? 0x80 : 0x00;
        value = Math.abs(value);

        // Calculate exponent and mantissa
        int bits = Float.floatToIntBits(value);
        int exp = ((bits >> 23) & 0xFF) - 127 + format.getExpBias();
        int mantissa = (bits & 0x7FFFFF) >> (23 - format.getMantissaBits());

        // Handle overflow/underflow
        if (exp >= (1 << format.getExpBits()) - 1) {
            // Overflow - return finite value or infinity
            return (byte) (sign | format.getFiniteValue());
        }
        if (exp <= 0) {
            // Underflow - return 0 or subnormal
            return (byte) sign;
        }

        // Pack FP8 value
        int fp8 = (sign << 7) | (exp << format.getMantissaBits()) | (mantissa & ((1 << format.getMantissaBits()) - 1));
        return (byte) (fp8 & 0xFF);
    }

    /**
     * Dequantize FP8 value to float.
     *
     * @param fp8    FP8 byte value
     * @param format FP8 format
     * @return float value
     */
    private float dequantizeFP8(byte fp8, FP8Format format) {
        int bits = fp8 & 0xFF;

        // Extract sign, exponent, and mantissa
        int sign = (bits >> 7) & 0x01;
        int exp = (bits >> format.getMantissaBits()) & ((1 << format.getExpBits()) - 1);
        int mantissa = bits & ((1 << format.getMantissaBits()) - 1);

        // Handle special cases
        if (exp == (1 << format.getExpBits()) - 1) {
            if (mantissa == 0) {
                return sign == 1 ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            } else {
                return Float.NaN;
            }
        }

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 1 ? -0.0f : 0.0f;
            }
            // Subnormal number
            exp = 1 - format.getExpBias();
        } else {
            // Normal number
            exp = exp - format.getExpBias();
            mantissa |= (1 << format.getMantissaBits()); // Add implicit leading 1
        }

        // Convert to float
        int floatBits = (sign << 31) | ((exp + 127) << 23) | (mantissa << (23 - format.getMantissaBits()));
        return Float.intBitsToFloat(floatBits);
    }

    /**
     * Calculate optimal scale factor for FP8 quantization.
     *
     * @param data   input data
     * @param format FP8 format
     * @return scale factor
     */
    private float calculateScale(float[] data, FP8Format format) {
        // Find max absolute value
        float maxAbs = 0.0f;
        for (float val : data) {
            float abs = Math.abs(val);
            if (abs > maxAbs)
                maxAbs = abs;
        }

        if (maxAbs == 0)
            return 1.0f;

        // Calculate scale to map max value to FP8 max representable value
        float fp8Max = getMaxFP8Value(format);
        return maxAbs / fp8Max;
    }

    /**
     * Get maximum representable value for FP8 format.
     *
     * @param format FP8 format
     * @return maximum value
     */
    private float getMaxFP8Value(FP8Format format) {
        // Max value is when exp is max-1 and mantissa is all 1s
        int maxExp = (1 << format.getExpBits()) - 2;
        float mantissaSum = 0.0f;
        for (int i = 0; i < format.getMantissaBits(); i++) {
            mantissaSum += Math.pow(2, -1 - i);
        }
        return (float) (Math.pow(2, maxExp - format.getExpBias()) * (1.0 + mantissaSum));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TorchTensor manipulation helpers
    // ─────────────────────────────────────────────────────────────────────────

    private float[] getTensorData(TorchTensor tensor) {
        // TODO: Implement based on actual TorchTensor API
        return new float[1024]; // Placeholder
    }

    private int[] getTensorShape(TorchTensor tensor) {
        // TODO: Implement based on actual TorchTensor API
        return new int[] { 1, 1024 }; // Placeholder
    }

    private byte[] getQuantizedData(TorchTensor tensor) {
        // TODO: Extract quantized data from tensor
        return new byte[1024]; // Placeholder
    }

    private float[] getScales(TorchTensor tensor) {
        // TODO: Extract scales from tensor metadata
        return new float[] { 1.0f }; // Placeholder
    }

    private FP8Format getFormat(TorchTensor tensor) {
        // TODO: Extract format from tensor metadata
        return DEFAULT_FORMAT; // Placeholder
    }

    private TorchTensor createQuantizedTensor(byte[] data, float[] scales, int[] shape, QuantConfig config,
            FP8Format format) {
        // TODO: Create tensor with quantized data and metadata
        return null; // Placeholder
    }

    private TorchTensor createDequantizedTensor(float[] data, int[] shape) {
        // TODO: Create tensor with dequantized data
        return null; // Placeholder
    }
}
