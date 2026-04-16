/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantConfig.java
 * ───────────────────────
 * Quantization configuration.
 */
package tech.kayys.gollek.safetensor.quantization;

import java.util.Objects;

/**
 * Configuration for model quantization.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class QuantConfig {

    /**
     * Target quantization strategy.
     */
    private final QuantizationEngine.QuantStrategy strategy;

    /**
     * Group size for GPTQ quantization (e.g., 128).
     */
    private final int groupSize;

    /**
     * Bits for symmetric quantization.
     */
    private final int bits;

    /**
     * Whether to use symmetric quantization.
     */
    private final boolean symmetric;

    /**
     * Whether to use per-channel quantization.
     */
    private final boolean perChannel;

    /**
     * Act order for GPTQ (true for ActOrder, false for static).
     */
    private final boolean actOrder;

    /**
     * Dampening percentage for GPTQ (0.0 to 1.0).
     */
    private final double dampPercent;

    /**
     * Number of samples for calibration.
     */
    private final int numSamples;

    /**
     * Sequence length for calibration samples.
     */
    private final int seqLen;

    /**
     * Whether to use desc_act (descending activation order).
     */
    private final boolean descAct;

    /**
     * Default constructor with sensible defaults for INT4 GPTQ.
     */
    public QuantConfig() {
        this(QuantizationEngine.QuantStrategy.INT4, 128, 4, false, true, false, 0.01, 128, 2048, false);
    }

    /**
     * Create quantization configuration.
     *
     * @param strategy    quantization strategy
     * @param groupSize   group size for quantization
     * @param bits        number of bits
     * @param symmetric   symmetric quantization
     * @param perChannel  per-channel quantization
     * @param actOrder    activation order for GPTQ
     * @param dampPercent dampening percentage
     * @param numSamples  number of calibration samples
     * @param seqLen      sequence length
     * @param descAct     descending activation order
     */
    public QuantConfig(QuantizationEngine.QuantStrategy strategy, int groupSize, int bits, boolean symmetric,
            boolean perChannel, boolean actOrder, double dampPercent, int numSamples, int seqLen, boolean descAct) {
        this.strategy = Objects.requireNonNull(strategy, "strategy cannot be null");
        this.groupSize = groupSize;
        this.bits = bits;
        this.symmetric = symmetric;
        this.perChannel = perChannel;
        this.actOrder = actOrder;
        this.dampPercent = dampPercent;
        this.numSamples = numSamples;
        this.seqLen = seqLen;
        this.descAct = descAct;
    }

    /**
     * Create builder for QuantConfig.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create default INT4 GPTQ config.
     *
     * @return INT4 GPTQ configuration
     */
    public static QuantConfig int4Gptq() {
        return builder().strategy(QuantizationEngine.QuantStrategy.INT4).build();
    }

    /**
     * Create default INT8 config.
     *
     * @return INT8 configuration
     */
    public static QuantConfig int8() {
        return builder().strategy(QuantizationEngine.QuantStrategy.INT8).perChannel(true).build();
    }

    /**
     * Create default FP8 config.
     *
     * @return FP8 configuration
     */
    public static QuantConfig fp8() {
        return builder().strategy(QuantizationEngine.QuantStrategy.FP8).symmetric(true).build();
    }

    public QuantizationEngine.QuantStrategy getStrategy() {
        return strategy;
    }

    public int getGroupSize() {
        return groupSize;
    }

    public int getBits() {
        return bits;
    }

    public boolean isSymmetric() {
        return symmetric;
    }

    public boolean isPerChannel() {
        return perChannel;
    }

    public boolean isActOrder() {
        return actOrder;
    }

    public double getDampPercent() {
        return dampPercent;
    }

    public int getNumSamples() {
        return numSamples;
    }

    public int getSeqLen() {
        return seqLen;
    }

    public boolean isDescAct() {
        return descAct;
    }

    /**
     * Builder for QuantConfig.
     */
    public static class Builder {
        private QuantizationEngine.QuantStrategy strategy = QuantizationEngine.QuantStrategy.INT4;
        private int groupSize = 128;
        private int bits = 4;
        private boolean symmetric = false;
        private boolean perChannel = true;
        private boolean actOrder = false;
        private double dampPercent = 0.01;
        private int numSamples = 128;
        private int seqLen = 2048;
        private boolean descAct = false;

        public Builder strategy(QuantizationEngine.QuantStrategy strategy) {
            this.strategy = strategy;
            return this;
        }

        public Builder groupSize(int groupSize) {
            this.groupSize = groupSize;
            return this;
        }

        public Builder bits(int bits) {
            this.bits = bits;
            return this;
        }

        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder perChannel(boolean perChannel) {
            this.perChannel = perChannel;
            return this;
        }

        public Builder actOrder(boolean actOrder) {
            this.actOrder = actOrder;
            return this;
        }

        public Builder dampPercent(double dampPercent) {
            this.dampPercent = dampPercent;
            return this;
        }

        public Builder numSamples(int numSamples) {
            this.numSamples = numSamples;
            return this;
        }

        public Builder seqLen(int seqLen) {
            this.seqLen = seqLen;
            return this;
        }

        public Builder descAct(boolean descAct) {
            this.descAct = descAct;
            return this;
        }

        public QuantConfig build() {
            return new QuantConfig(strategy, groupSize, bits, symmetric, perChannel, actOrder, dampPercent, numSamples,
                    seqLen, descAct);
        }
    }
}
