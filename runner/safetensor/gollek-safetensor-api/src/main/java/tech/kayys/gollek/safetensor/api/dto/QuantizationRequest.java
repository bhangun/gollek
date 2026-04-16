/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationRequest.java
 * ───────────────────────
 * Quantization request DTO.
 */
package tech.kayys.gollek.safetensor.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Request payload for quantization operations.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class QuantizationRequest {

    /**
     * Path to the input model.
     */
    @JsonProperty("input_path")
    private String inputPath;

    /**
     * Path for the quantized output.
     */
    @JsonProperty("output_path")
    private String outputPath;

    /**
     * Quantization strategy (INT4, INT8, FP8).
     */
    @JsonProperty("strategy")
    private String strategy = "INT4";

    /**
     * Number of bits for quantization.
     */
    @JsonProperty("bits")
    private int bits = 4;

    /**
     * Group size for quantization.
     */
    @JsonProperty("group_size")
    private int groupSize = 128;

    /**
     * Whether to use symmetric quantization.
     */
    @JsonProperty("symmetric")
    private boolean symmetric = false;

    /**
     * Whether to use per-channel quantization.
     */
    @JsonProperty("per_channel")
    private boolean perChannel = true;

    /**
     * Whether to use activation order for GPTQ.
     */
    @JsonProperty("act_order")
    private boolean actOrder = false;

    /**
     * Dampening percentage for GPTQ.
     */
    @JsonProperty("damp_percent")
    private double dampPercent = 0.01;

    /**
     * Number of calibration samples.
     */
    @JsonProperty("num_samples")
    private int numSamples = 128;

    /**
     * Sequence length for calibration.
     */
    @JsonProperty("seq_len")
    private int seqLen = 2048;

    public String getInputPath() {
        return inputPath;
    }

    public void setInputPath(String inputPath) {
        this.inputPath = inputPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = outputPath;
    }

    public String getStrategy() {
        return strategy;
    }

    public void setStrategy(String strategy) {
        this.strategy = strategy;
    }

    public int getBits() {
        return bits;
    }

    public void setBits(int bits) {
        this.bits = bits;
    }

    public int getGroupSize() {
        return groupSize;
    }

    public void setGroupSize(int groupSize) {
        this.groupSize = groupSize;
    }

    public boolean isSymmetric() {
        return symmetric;
    }

    public void setSymmetric(boolean symmetric) {
        this.symmetric = symmetric;
    }

    public boolean isPerChannel() {
        return perChannel;
    }

    public void setPerChannel(boolean perChannel) {
        this.perChannel = perChannel;
    }

    public boolean isActOrder() {
        return actOrder;
    }

    public void setActOrder(boolean actOrder) {
        this.actOrder = actOrder;
    }

    public double getDampPercent() {
        return dampPercent;
    }

    public void setDampPercent(double dampPercent) {
        this.dampPercent = dampPercent;
    }

    public int getNumSamples() {
        return numSamples;
    }

    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    public int getSeqLen() {
        return seqLen;
    }

    public void setSeqLen(int seqLen) {
        this.seqLen = seqLen;
    }
}
