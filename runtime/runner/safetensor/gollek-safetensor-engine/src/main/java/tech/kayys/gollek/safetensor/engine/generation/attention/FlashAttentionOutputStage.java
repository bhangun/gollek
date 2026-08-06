/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

/**
 * Reshapes attention context and applies the final O projection, preserving
 * reusable output workspaces through optional post-attention normalization.
 */
final class FlashAttentionOutputStage {
    private final FlashAttentionProjector projector;
    private final FlashAttentionNormalizer normalizer;

    FlashAttentionOutputStage(FlashAttentionProjector projector, FlashAttentionNormalizer normalizer) {
        this.projector = projector;
        this.normalizer = normalizer;
    }

    AccelTensor project(AttentionInput in, AccelTensor attentionOutput, int seqLen,
            FlashAttentionHeadLayout headLayout, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            boolean addOneRmsNorm) {
        AccelTensor projectedInput = reshapeAttentionOutput(in, attentionOutput, seqLen, headLayout);
        AccelTensor projected = projectOutput(in, projectedInput, config, modelPolicy);
        if (in.postAttnNormW == null) {
            return projected;
        }

        AccelTensor normed = normalizer.rmsNormReusingInput(projected, in.postAttnNormW, config.getRmsNormEps(),
                addOneRmsNorm);
        if (normed != projected) {
            projected.close();
        }
        return normed;
    }

    private AccelTensor reshapeAttentionOutput(AttentionInput in, AccelTensor attentionOutput, int seqLen,
            FlashAttentionHeadLayout headLayout) {
        AccelTensor reshaped = attentionOutput.reshape(
                in.x.size(0), seqLen, (long) headLayout.numQueryHeads() * headLayout.headDim());
        attentionOutput.close();
        return reshaped;
    }

    private AccelTensor projectOutput(AttentionInput in, AccelTensor projectedInput, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy) {
        AccelTensor projectionBuffer = projector.attentionOutputBufferView(in, projectedInput);
        AccelTensor projected = projector.project(projectedInput, in.oW, in.oB, "attn_o_proj", config, modelPolicy,
                projectionBuffer);
        if (projectionBuffer != null && projectionBuffer != projected && !projectionBuffer.isClosed()) {
            projectionBuffer.close();
        }
        projectedInput.close();
        return projected;
    }
}
