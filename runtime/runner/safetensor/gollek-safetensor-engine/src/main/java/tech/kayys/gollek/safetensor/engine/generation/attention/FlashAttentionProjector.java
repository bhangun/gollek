/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Routes attention linear projections through Metal when possible and through
 * reusable Accelerate-backed fallback buffers otherwise.
 */
final class FlashAttentionProjector {
    private final FlashAttentionMetalLinear metalLinear;

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal) {
        this(metalBinding, canUseMetal, FlashAttentionLinearOptions.defaults());
    }

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions) {
        this(metalBinding, canUseMetal, linearOptions, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionProjector(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions, FlashAttentionMatvecOptions matvecOptions) {
        this(new FlashAttentionMetalLinear(metalBinding, canUseMetal, linearOptions, matvecOptions));
    }

    FlashAttentionProjector(FlashAttentionMetalLinear metalLinear) {
        this.metalLinear = metalLinear;
    }

    AccelTensor attentionOutputBufferView(AttentionInput in, AccelTensor projectedInput) {
        if (in.attentionOutputBuffer == null || projectedInput == null || in.oW == null || in.oW.rank() != 2) {
            return null;
        }
        long[] outputShape = projectedInput.shapeWithLastDim(in.oW.size(0));
        long requiredBytes = Math.multiplyExact(numElements(outputShape), (long) Float.BYTES);
        if (in.attentionOutputBuffer.byteSize() < requiredBytes) {
            return null;
        }
        return AccelTensor.view(in.attentionOutputBuffer, outputShape);
    }

    ProjectionBuffers attentionProjectionBuffers(AttentionInput in, boolean enabled) {
        return attentionProjectionBuffers(in, enabled, true);
    }

    ProjectionBuffers attentionProjectionBuffers(AttentionInput in, boolean enabled, boolean includeValue) {
        if (!enabled || in == null || in.kvCache == null || in.qW == null || in.kW == null
                || (includeValue && in.vW == null)) {
            return null;
        }
        MemorySegment scratch = in.kvCache.getWorkspace().getCombinedSeg();
        if (scratch == null || in.x == null || in.x.rank() < 2) {
            return null;
        }
        long qDim = in.qW.size(0);
        long kDim = in.kW.size(0);
        long vDim = includeValue ? in.vW.size(0) : 0L;
        if (qDim <= 0 || kDim <= 0 || (includeValue && vDim <= 0)) {
            return null;
        }
        long batchTokens = in.x.numel() / Math.max(1L, in.x.size(-1));
        if (batchTokens <= 0L) {
            return null;
        }
        long qElements = Math.multiplyExact(batchTokens, qDim);
        long kElements = Math.multiplyExact(batchTokens, kDim);
        long vElements = Math.multiplyExact(batchTokens, vDim);
        long qBytes = Math.multiplyExact(qElements, (long) Float.BYTES);
        long kBytes = Math.multiplyExact(kElements, (long) Float.BYTES);
        long vBytes = Math.multiplyExact(vElements, (long) Float.BYTES);
        long requiredBytes = Math.addExact(Math.addExact(qBytes, kBytes), vBytes);
        if (scratch.byteSize() < requiredBytes) {
            return null;
        }
        return new ProjectionBuffers(
                AccelTensor.view(scratch.asSlice(0, qBytes), in.x.shapeWithLastDim(qDim)),
                AccelTensor.view(scratch.asSlice(qBytes, kBytes), in.x.shapeWithLastDim(kDim)),
                includeValue
                        ? AccelTensor.view(scratch.asSlice(qBytes + kBytes, vBytes), in.x.shapeWithLastDim(vDim))
                        : null);
    }

    AccelTensor packedQkvProjectionBuffer(AttentionInput in, boolean enabled, FlashAttentionHeadLayout layout) {
        if (!enabled || in == null || in.kvCache == null || in.x == null || in.x.rank() < 2
                || layout == null || !layout.packedQkvProjection()) {
            return null;
        }
        MemorySegment scratch = in.kvCache.getWorkspace().getCombinedSeg();
        if (scratch == null) {
            return null;
        }
        long packedDim = layout.packedQkvProjectionDim();
        long batchTokens = in.x.numel() / Math.max(1L, in.x.size(-1));
        if (packedDim <= 0L || batchTokens <= 0L) {
            return null;
        }
        long requiredBytes = Math.multiplyExact(Math.multiplyExact(batchTokens, packedDim), (long) Float.BYTES);
        if (scratch.byteSize() < requiredBytes) {
            return null;
        }
        return AccelTensor.view(scratch.asSlice(0, requiredBytes), in.x.shapeWithLastDim(packedDim));
    }

    AccelTensor project(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy) {
        return project(input, weight, bias, profileKey, config, modelPolicy, null);
    }

    LinearTriple projectPackedQkv(AttentionInput in, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            FlashAttentionHeadLayout layout) {
        return projectPackedQkv(in, config, modelPolicy, layout, null);
    }

    LinearTriple projectPackedQkv(AttentionInput in, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            FlashAttentionHeadLayout layout, AccelTensor outputBuffer) {
        if (in == null || in.qW == null || layout == null || !layout.packedQkvProjection()) {
            return null;
        }
        FlashAttentionShapeAdmissionPlan weightAdmission = FlashAttentionShapeAdmissionPlan.packedQkvWeight(in.qW,
                layout);
        if (!weightAdmission.admitted()) {
            throw weightAdmission.asException();
        }

        AccelTensor packed = null;
        boolean packedOwnershipTransferred = false;
        try {
            packed = project(in.x, in.qW, in.qB, "attn_qkv_proj_packed", config, modelPolicy, outputBuffer);
            FlashAttentionShapeAdmissionPlan outputAdmission = FlashAttentionShapeAdmissionPlan.packedQkvOutput(packed,
                    layout);
            if (!outputAdmission.admitted()) {
                throw outputAdmission.asException();
            }
            long qEnd = layout.queryProjectionDim();
            long kEnd = qEnd + layout.keyValueProjectionDim();
            long vEnd = kEnd + layout.keyValueProjectionDim();
            if (canAliasPackedLastDimSlices(packed)) {
                packedOwnershipTransferred = true;
                return new LinearTriple(
                        ownedLastDimSliceView(packed, 0, qEnd),
                        ownedLastDimSliceView(packed, qEnd, kEnd),
                        ownedLastDimSliceView(packed, kEnd, vEnd),
                        packed);
            }
            return new LinearTriple(
                    materializedLastDimSlice(packed, 0, qEnd),
                    materializedLastDimSlice(packed, qEnd, kEnd),
                    materializedLastDimSlice(packed, kEnd, vEnd));
        } finally {
            if (!packedOwnershipTransferred && packed != null && !packed.isClosed()) {
                packed.close();
            }
            if (!packedOwnershipTransferred && packed == null && outputBuffer != null && !outputBuffer.isClosed()) {
                outputBuffer.close();
            }
        }
    }

    AccelTensor project(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, AccelTensor outputBuffer) {
        long t0 = System.nanoTime();
        AccelTensor projected = metalLinear.tryHalfLinear(input, weight, bias, config, modelPolicy, profileKey,
                outputBuffer);
        if (projected == null) {
            projected = metalLinear.tryFloatLinear(input, weight, bias, outputBuffer);
            if (projected != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "metal_float_matmul");
            }
        }
        if (projected == null) {
            projected = AccelOps.linear(input, weight, bias, outputBuffer);
            DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear");
        }
        DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
        return projected;
    }

    LinearPair tryMetalHalfLinearPairMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer) {
        return metalLinear.tryHalfLinearPairMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer);
    }

    LinearTriple tryMetalHalfLinearTripleMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            AccelTensor thirdWeight,
            AccelTensor thirdBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer,
            AccelTensor thirdOutputBuffer) {
        return metalLinear.tryHalfLinearTripleMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                thirdWeight, thirdBias, profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer,
                thirdOutputBuffer);
    }

    private AccelTensor materializedLastDimSlice(AccelTensor packed, long start, long end) {
        AccelTensor view = packed.slice(-1, start, end);
        AccelTensor contiguous = view.contiguous();
        if (contiguous == view) {
            AccelTensor copy = AccelTensor.copyOf(view.dataPtr(), view.shape());
            view.close();
            return copy;
        }
        view.close();
        return contiguous;
    }

    private static boolean canAliasPackedLastDimSlices(AccelTensor packed) {
        if (packed == null || packed.rank() == 0 || !packed.isContiguous()) {
            return false;
        }
        long lastDim = packed.size(-1);
        return lastDim > 0 && packed.numel() / lastDim == 1L;
    }

    private static AccelTensor ownedLastDimSliceView(AccelTensor packed, long start, long end) {
        long elements = Math.subtractExact(end, start);
        long byteOffset = Math.multiplyExact(start, (long) Float.BYTES);
        long byteSize = Math.multiplyExact(elements, (long) Float.BYTES);
        return AccelTensor.view(packed.dataPtr().asSlice(byteOffset, byteSize),
                packed.shapeWithLastDim(elements), packed);
    }

    private static long numElements(long[] shape) {
        long elements = 1L;
        for (long dim : shape) {
            elements = Math.multiplyExact(elements, dim);
        }
        return elements;
    }

    record LinearPair(AccelTensor first, AccelTensor second) {
    }

    record LinearTriple(AccelTensor first, AccelTensor second, AccelTensor third, AccelTensor sharedOwner) {
        LinearTriple(AccelTensor first, AccelTensor second, AccelTensor third) {
            this(first, second, third, null);
        }

        void closeSharedOwner() {
            if (sharedOwner != null && !sharedOwner.isClosed()) {
                sharedOwner.close();
            }
        }
    }

    record ProjectionBuffers(AccelTensor q, AccelTensor k, AccelTensor v) {
    }
}
