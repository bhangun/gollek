/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizedTensorInfo.java
 * ───────────────────────
 * Quantized tensor metadata.
 */
package tech.kayys.gollek.safetensor.quantization;

import java.util.Arrays;
import java.util.Objects;

/**
 * Metadata for a quantized tensor.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class QuantizedTensorInfo {

    /**
     * TorchTensor name.
     */
    private final String name;

    /**
     * TorchTensor shape.
     */
    private final int[] shape;

    /**
     * Original data type (before quantization).
     */
    private final String originalDtype;

    /**
     * Quantized data type.
     */
    private final String quantizedDtype;

    /**
     * Quantization strategy used.
     */
    private final QuantizationEngine.QuantStrategy strategy;

    /**
     * Scale factors for dequantization.
     */
    private final float[] scales;

    /**
     * Zero points for dequantization.
     */
    private final float[] zeros;

    /**
     * Group size used for quantization.
     */
    private final int groupSize;

    /**
     * Axis for per-channel quantization.
     */
    private final int channelAxis;

    /**
     * Additional metadata.
     */
    private final byte[] metadata;

    private QuantizedTensorInfo(Builder builder) {
        this.name = builder.name;
        this.shape = builder.shape;
        this.originalDtype = builder.originalDtype;
        this.quantizedDtype = builder.quantizedDtype;
        this.strategy = builder.strategy;
        this.scales = builder.scales;
        this.zeros = builder.zeros;
        this.groupSize = builder.groupSize;
        this.channelAxis = builder.channelAxis;
        this.metadata = builder.metadata;
    }

    /**
     * Create builder for QuantizedTensorInfo.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    public String getName() {
        return name;
    }

    public int[] getShape() {
        return shape;
    }

    public String getOriginalDtype() {
        return originalDtype;
    }

    public String getQuantizedDtype() {
        return quantizedDtype;
    }

    public QuantizationEngine.QuantStrategy getStrategy() {
        return strategy;
    }

    public float[] getScales() {
        return scales;
    }

    public float[] getZeros() {
        return zeros;
    }

    public int getGroupSize() {
        return groupSize;
    }

    public int getChannelAxis() {
        return channelAxis;
    }

    public byte[] getMetadata() {
        return metadata;
    }

    /**
     * Builder for QuantizedTensorInfo.
     */
    public static class Builder {
        private String name;
        private int[] shape;
        private String originalDtype = "F32";
        private String quantizedDtype = "INT8";
        private QuantizationEngine.QuantStrategy strategy = QuantizationEngine.QuantStrategy.INT8;
        private float[] scales = new float[0];
        private float[] zeros = new float[0];
        private int groupSize = 128;
        private int channelAxis = 0;
        private byte[] metadata = new byte[0];

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder shape(int[] shape) {
            this.shape = shape;
            return this;
        }

        public Builder originalDtype(String originalDtype) {
            this.originalDtype = originalDtype;
            return this;
        }

        public Builder quantizedDtype(String quantizedDtype) {
            this.quantizedDtype = quantizedDtype;
            return this;
        }

        public Builder strategy(QuantizationEngine.QuantStrategy strategy) {
            this.strategy = strategy;
            return this;
        }

        public Builder scales(float[] scales) {
            this.scales = scales;
            return this;
        }

        public Builder zeros(float[] zeros) {
            this.zeros = zeros;
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder channelAxis(int channelAxis) {
            this.channelAxis = channelAxis;
            return this;
        }

        public Builder metadata(byte[] metadata) {
            this.metadata = metadata;
            return this;
        }

        public QuantizedTensorInfo build() {
            return new QuantizedTensorInfo(this);
        }
    }
}
