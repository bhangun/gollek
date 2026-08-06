/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;

final class FlashAttentionJavaDispatchAdapter implements FlashAttentionJavaDispatchBackend {
    private final PagedAttentionVectorOptions pagedAttentionOptions;

    FlashAttentionJavaDispatchAdapter() {
        this(PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionJavaDispatchAdapter(PagedAttentionVectorOptions pagedAttentionOptions) {
        this.pagedAttentionOptions =
                pagedAttentionOptions == null ? PagedAttentionVectorOptions.defaults() : pagedAttentionOptions;
    }

    @Override
    public AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request) {
        DirectInferenceProfiler.recordAttentionPath("dense_shared_java");
        return FlashAttentionJavaFallback.denseSharedAttention(
                request.query(), request.key(), request.value(), request.config(), request.layerIdx(),
                request.startPos(), request.numQueryHeads(), request.numKeyValueHeads(), request.headDim(),
                request.scale(), request.causal(), request.attentionSoftCap(), request.attentionContextBuffer());
    }

    @Override
    public AccelTensor denseRestrictedAttention(FlashAttentionDispatchRequest request) {
        DirectInferenceProfiler.recordAttentionPath("dense_restricted_java");
        return FlashAttentionJavaFallback.denseCachedAttention(
                request.query(), request.kvSession(), request.kvLayerIdx(), request.startPos(),
                request.numQueryHeads(), request.numKeyValueHeads(), request.headDim(), request.scale(),
                request.causal(), request.attentionSoftCap(), request.config(), request.layerIdx(),
                request.attentionContextBuffer());
    }

    @Override
    public AccelTensor pagedAttention(FlashAttentionDispatchRequest request) {
        DirectInferenceProfiler.recordAttentionPath("paged_java");
        return PagedAttentionVectorAPI.compute(
                request.query(), request.config(), request.kvSession(), request.layerIdx(), request.kvLayerIdx(),
                request.startPos(), request.numQueryHeads(), request.numKeyValueHeads(), request.headDim(),
                request.scale(), request.causal(), request.attentionSoftCap(), pagedAttentionOptions,
                request.attentionContextBuffer());
    }
}
