/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8Quantizer.java
 * ───────────────────────
 * INT8 quantization implementation.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;

/**
 * INT8 quantizer with per-channel or per-tensor scaling.
 * <p>
 * INT8 quantization provides a good balance between model size reduction
 * and inference quality. It uses symmetric quantization with zero-point
 * at 0 for efficient integer arithmetic.
 * </p>
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class INT8Quantizer implements Quantizer {

    private static final Logger log = Logger.getLogger(INT8Quantizer.class);

    /**
     * INT8 minimum value.
     */
    private static final int INT8_MIN = -128;

    /**
     * INT8 maximum value.
     */
    private static final int INT8_MAX = 127;

    /**
     * Quantize a tensor using INT8.
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

        log.debugf("INT8 quantizing tensor (perChannel=%b)", config.isPerChannel());

        try {
            float[] data = getTensorData(tensor);
            int[] shape = getTensorShape(tensor);

            byte[] quantizedData;
            float[] scales;

            if (config.isPerChannel()) {
                // Per-channel quantization (better accuracy)
                int channels = shape[0];
                int elementsPerChannel = data.length / channels;

                quantizedData = new byte[data.length];
                scales = new float[channels];

                for (int c = 0; c < channels; c++) {
                    int offset = c * elementsPerChannel;
                    float[] channelData = new float[elementsPerChannel];
                    System.arraycopy(data, offset, channelData, 0, elementsPerChannel);

                    float[] result = quantizeToInt8(channelData);
                    scales[c] = result[0]; // Scale factor

                    for (int i = 0; i < elementsPerChannel; i++) {
                        quantizedData[offset + i] = (byte) Math.round(result[i + 1]);
                    }
                }
            } else {
                // Per-tensor quantization (simpler, faster)
                float[] result = quantizeToInt8(data);
                scales = new float[] { result[0] };
                quantizedData = new byte[data.length];

                for (int i = 0; i < data.length; i++) {
                    quantizedData[i] = (byte) Math.round(result[i + 1]);
                }
            }

            return createQuantizedTensor(quantizedData, scales, shape, config);

        } catch (Exception e) {
            log.errorf(e, "INT8 quantization failed");
            throw new RuntimeException("INT8 quantization failed", e);
        }
    }

    /**
     * Dequantize an INT8 tensor.
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

        log.debugf("Dequantizing INT8 tensor");

        try {
            byte[] quantizedData = getQuantizedData(quantizedTensor);
            float[] scales = getScales(quantizedTensor);
            int[] shape = getTensorShape(quantizedTensor);

            float[] dequantizedData = new float[quantizedData.length];

            if (config.isPerChannel()) {
                int channels = shape[0];
                int elementsPerChannel = quantizedData.length / channels;

                for (int c = 0; c < channels; c++) {
                    int offset = c * elementsPerChannel;
                    float scale = scales[c];

                    for (int i = 0; i < elementsPerChannel; i++) {
                        dequantizedData[offset + i] = quantizedData[offset + i] * scale;
                    }
                }
            } else {
                float scale = scales[0];
                for (int i = 0; i < quantizedData.length; i++) {
                    dequantizedData[i] = quantizedData[i] * scale;
                }
            }

            return createDequantizedTensor(dequantizedData, shape);

        } catch (Exception e) {
            log.errorf(e, "INT8 dequantization failed");
            throw new RuntimeException("INT8 dequantization failed", e);
        }
    }

    @Override
    public String getName() {
        return "INT8Quantizer";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getBits() == 8;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INT8 Quantization Implementation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Quantize float array to INT8.
     *
     * @param data input float data
     * @return array with scale at index 0, followed by quantized values
     */
    private float[] quantizeToInt8(float[] data) {
        // Find min and max values
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;

        for (float val : data) {
            if (val < min)
                min = val;
            if (val > max)
                max = val;
        }

        // Use symmetric quantization
        float absMax = Math.max(Math.abs(min), Math.abs(max));
        if (absMax == 0)
            absMax = 1e-6f;

        // Calculate scale
        float scale = absMax / INT8_MAX;
        if (scale == 0)
            scale = 1e-6f;

        // Quantize
        float[] result = new float[data.length + 1];
        result[0] = scale; // Store scale at index 0

        for (int i = 0; i < data.length; i++) {
            int qval = Math.round(data[i] / scale);
            qval = Math.max(INT8_MIN, Math.min(INT8_MAX, qval));
            result[i + 1] = qval;
        }

        return result;
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

    private TorchTensor createQuantizedTensor(byte[] data, float[] scales, int[] shape, QuantConfig config) {
        // TODO: Create tensor with quantized data and metadata
        return null; // Placeholder
    }

    private TorchTensor createDequantizedTensor(float[] data, int[] shape) {
        // TODO: Create tensor with dequantized data
        return null; // Placeholder
    }
}
