/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Applies attention-local normalization, preferring Metal kernels when
 * available
 * while still exposing in-place CPU fallbacks for reusable projection buffers.
 */
final class FlashAttentionNormalizer {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private final Supplier<MetalBinding> metalBinding;
    private final FlashAttentionNormalizerPolicy normalizerPolicy;
    private final Map<Integer, AccelTensor> unitScaleRmsNormWeights = new ConcurrentHashMap<>();

    FlashAttentionNormalizer(Supplier<MetalBinding> metalBinding) {
        this(metalBinding, FlashAttentionNormalizerOptions.fromSystemProperties());
    }

    FlashAttentionNormalizer(Supplier<MetalBinding> metalBinding, FlashAttentionNormalizerOptions options) {
        this.metalBinding = metalBinding;
        this.normalizerPolicy = FlashAttentionNormalizerPolicy.from(options);
    }

    AccelTensor perHeadRmsNorm(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy) {
        AccelTensor normed = tryMetalPerHeadRmsNorm(x, weight, eps, addOne, modelPolicy);
        if (normed != null) {
            return normed;
        }
        return AccelOps.perHeadRmsNorm(x, weight, eps, addOne);
    }

    AccelTensor perHeadRmsNormReusingInput(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy) {
        if (canReuseInputForPerHeadRmsNorm(x, weight)) {
            return AccelOps.perHeadRmsNorm(x, weight, eps, addOne, x);
        }
        return perHeadRmsNorm(x, weight, eps, addOne, modelPolicy);
    }

    AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps, FlashAttentionModelPolicy modelPolicy) {
        AccelTensor normed = tryMetalPerHeadRmsNormNoWeight(x, eps, modelPolicy);
        if (normed != null) {
            return normed;
        }
        return perHeadRmsNormNoWeight(x, eps);
    }

    AccelTensor perHeadRmsNormNoWeightReusingInput(AccelTensor x, double eps,
            FlashAttentionModelPolicy modelPolicy) {
        if (canReuseInputForPerHeadRmsNorm(x, null)) {
            return perHeadRmsNormNoWeight(x, eps, x);
        }
        return perHeadRmsNormNoWeight(x, eps, modelPolicy);
    }

    AccelTensor rmsNorm(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        return AccelOps.rmsNorm(x, w, eps, addOne);
    }

    AccelTensor rmsNormReusingInput(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        if (canReuseInputForRmsNorm(x, w)) {
            return AccelOps.rmsNorm(x, w, eps, addOne, x);
        }
        return rmsNorm(x, w, eps, addOne);
    }

    private AccelTensor tryMetalPerHeadRmsNorm(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy) {
        if (!shouldUseMetalPerHeadRmsNorm(modelPolicy) || x == null || weight == null) {
            return null;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.nativeElementwiseKernelsAvailable()) {
            return null;
        }
        if (x.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        long[] shape = x.shape();
        if (shape.length < 2) {
            return null;
        }
        int headDim = Math.toIntExact(shape[shape.length - 1]);
        if (headDim <= 0 || weight.numel() != headDim) {
            return null;
        }

        AccelTensor weightView = weight;
        AccelTensor contiguousInput = null;
        AccelTensor out = null;
        try {
            if (weight.quantType() != AccelTensor.QuantType.F32) {
                weightView = weight.dequantize();
            }
            if (weightView.quantType() != AccelTensor.QuantType.F32 || weightView.numel() != headDim) {
                return null;
            }
            contiguousInput = x.contiguous();
            out = AccelTensor.zeros(shape);
            int rows = Math.toIntExact(contiguousInput.numel() / headDim);
            int rc = binding.rmsNormRows(out.dataPtr(), contiguousInput.dataPtr(), weightView.dataPtr(),
                    rows, headDim, (float) eps, addOne);
            if (rc == 0) {
                return out;
            }
            out.close();
            out = null;
            return null;
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (contiguousInput != null && contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
            if (weightView != weight && !weightView.isClosed()) {
                weightView.close();
            }
        }
    }

    private AccelTensor tryMetalPerHeadRmsNormNoWeight(AccelTensor x, double eps,
            FlashAttentionModelPolicy modelPolicy) {
        if (!shouldUseMetalPerHeadRmsNorm(modelPolicy) || x == null) {
            return null;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.nativeElementwiseKernelsAvailable()) {
            return null;
        }
        if (x.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        long[] shape = x.shape();
        if (shape.length < 2) {
            return null;
        }
        int headDim = Math.toIntExact(shape[shape.length - 1]);
        if (headDim <= 0) {
            return null;
        }

        AccelTensor contiguousInput = null;
        AccelTensor out = null;
        try {
            AccelTensor zeroWeight = unitScaleRmsNormWeights.computeIfAbsent(headDim, dim -> AccelTensor.zeros(dim));
            contiguousInput = x.contiguous();
            out = AccelTensor.zeros(shape);
            int rows = Math.toIntExact(contiguousInput.numel() / headDim);
            int rc = binding.rmsNormRows(out.dataPtr(), contiguousInput.dataPtr(), zeroWeight.dataPtr(),
                    rows, headDim, (float) eps, true);
            if (rc == 0) {
                return out;
            }
            out.close();
            out = null;
            return null;
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (contiguousInput != null && contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private boolean shouldUseMetalPerHeadRmsNorm(FlashAttentionModelPolicy modelPolicy) {
        return normalizerPolicy.shouldUseMetalPerHeadRmsNorm(modelPolicy);
    }

    private static boolean canReuseInputForPerHeadRmsNorm(AccelTensor x, AccelTensor weight) {
        return canReuseInputForRmsNorm(x, weight);
    }

    private static boolean canReuseInputForRmsNorm(AccelTensor x, AccelTensor weight) {
        return x != null
                && !x.isClosed()
                && x.rank() >= 2
                && (weight == null || !weight.isClosed())
                && x.quantType() == AccelTensor.QuantType.F32
                && x.isContiguous();
    }

    private AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps) {
        return perHeadRmsNormNoWeight(x, eps, (AccelTensor) null);
    }

    private AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps, AccelTensor outputBuffer) {
        AccelTensor contiguousInput = x.contiguous();
        try {
            long[] shape = contiguousInput.shape();
            int headDim = (int) shape[shape.length - 1];
            int numHeads = (int) shape[shape.length - 2];
            int outer = (int) (contiguousInput.numel() / (numHeads * headDim));

            AccelTensor out = reusableFloatOutput(outputBuffer, shape);
            if (out == null) {
                out = AccelTensor.zeros(shape);
            }
            MemorySegment xSeg = contiguousInput.dataSegment();
            MemorySegment oSeg = out.dataPtr();

            for (int b = 0; b < outer; b++) {
                for (int h = 0; h < numHeads; h++) {
                    long base = (long) (b * numHeads + h) * headDim;
                    int upperBound = FLOAT_SPECIES.loopBound(headDim);
                    FloatVector sumSqVec = FloatVector.zero(FLOAT_SPECIES);
                    int i = 0;
                    for (; i < upperBound; i += FLOAT_SPECIES.length()) {
                        FloatVector valueVec = FloatVector.fromMemorySegment(
                                FLOAT_SPECIES, xSeg, (base + i) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        sumSqVec = valueVec.fma(valueVec, sumSqVec);
                    }
                    float sumSq = sumSqVec.reduceLanes(VectorOperators.ADD);
                    for (; i < headDim; i++) {
                        float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                        sumSq += val * val;
                    }
                    float rms = (float) (1.0 / Math.sqrt(sumSq / headDim + eps));
                    FloatVector rmsVec = FloatVector.broadcast(FLOAT_SPECIES, rms);
                    int j = 0;
                    for (; j < upperBound; j += FLOAT_SPECIES.length()) {
                        FloatVector valueVec = FloatVector.fromMemorySegment(
                                FLOAT_SPECIES, xSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                        valueVec.mul(rmsVec).intoMemorySegment(
                                oSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                    }
                    for (; j < headDim; j++) {
                        float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + j);
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + j, val * rms);
                    }
                }
            }
            return out;
        } finally {
            if (contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static AccelTensor reusableFloatOutput(AccelTensor outputBuffer, long[] shape) {
        if (outputBuffer == null
                || outputBuffer.isClosed()
                || outputBuffer.quantType() != AccelTensor.QuantType.F32
                || !outputBuffer.isContiguous()
                || !outputBuffer.hasShape(shape)) {
            return null;
        }
        long requiredBytes = Math.multiplyExact(outputBuffer.numel(), (long) Float.BYTES);
        return outputBuffer.dataPtr().byteSize() >= requiredBytes ? outputBuffer : null;
    }
}
