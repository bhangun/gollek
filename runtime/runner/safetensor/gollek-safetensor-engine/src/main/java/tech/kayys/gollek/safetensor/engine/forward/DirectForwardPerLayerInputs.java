/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class DirectForwardPerLayerInputs {
    private DirectForwardPerLayerInputs() {
    }

    static AccelTensor[] build(long[] inputIds, AccelTensor inputsEmbeds,
            ModelConfigTraits traits, ModelConfig config, ResolvedModelWeights resolvedWeights,
            DirectForwardOperators operators) {
        if (!DirectForwardElementwisePolicy.shouldBuildPerLayerInputs(
                traits,
                config.getHiddenSizePerLayerInput())) {
            return null;
        }

        AccelTensor packedPleEmbeddings = resolvedWeights.packedPleEmbeddings();
        AccelTensor pleProjection = resolvedWeights.pleProjection();
        AccelTensor pleProjectionNorm = resolvedWeights.pleProjectionNorm();
        if (packedPleEmbeddings == null || pleProjection == null || pleProjectionNorm == null) {
            System.err.printf("[DEBUG] DirectForwardPerLayerInputs.build returning null! packedPleEmbeddings=%s, pleProjection=%s, pleProjectionNorm=%s%n",
                    packedPleEmbeddings != null ? "PRESENT" : "NULL",
                    pleProjection != null ? "PRESENT" : "NULL",
                    pleProjectionNorm != null ? "PRESENT" : "NULL");
            return null;
        }

        int numLayers = config.getNumHiddenLayers();
        int pleDim = config.getHiddenSizePerLayerInput();
        int seqLen = inputIds.length;

        AccelTensor tokenPleRaw = packedPleEmbeddings.indexSelect(inputIds)
                .reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor tokenPle = AccelOps.mulScalar(tokenPleRaw, (float) Math.sqrt(Math.max(1, pleDim)));
        tokenPleRaw.close();
        AccelTensor projectedPle = operators.linear(inputsEmbeds, pleProjection, null, "ple_projection", config);
        float pleProjectionScale = (float) (1.0 / Math.sqrt(Math.max(1, config.getHiddenSize())));
        AccelTensor projectedPleScaled = AccelOps.mulScalar(projectedPle, pleProjectionScale);
        projectedPle.close();

        AccelTensor projectedPle4d = projectedPleScaled.reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor projectedPleNormed = AccelOps.rmsNorm(projectedPle4d, pleProjectionNorm,
                config.getRmsNormEps(), resolvedWeights.addOneRmsNorm());
        projectedPle4d.close();

        AccelTensor combinedPle = AccelOps.add(projectedPleNormed, tokenPle);
        projectedPleNormed.close();
        tokenPle.close();

        AccelTensor scaledPle = AccelOps.mulScalar(combinedPle, (float) (1.0 / Math.sqrt(2.0)));
        combinedPle.close();

        AccelTensor[] layers = new AccelTensor[numLayers];
        if (seqLen == 1) {
            MemorySegment packed = scaledPle.dataPtr();
            long layerBytes = (long) pleDim * Float.BYTES;
            for (int layer = 0; layer < numLayers; layer++) {
                layers[layer] = AccelTensor.view(
                        packed.asSlice((long) layer * layerBytes, layerBytes),
                        new long[] { 1L, 1L, pleDim },
                        scaledPle);
            }
            return layers;
        }

        MemorySegment src = scaledPle.dataPtr();
        for (int layer = 0; layer < numLayers; layer++) {
            AccelTensor layerTensor = AccelTensor.zeros(new long[] { 1L, seqLen, pleDim });
            MemorySegment dst = layerTensor.dataPtr();
            for (int token = 0; token < seqLen; token++) {
                long srcIndex = ((long) token * numLayers + layer) * pleDim;
                long dstIndex = (long) token * pleDim;
                MemorySegment.copy(src, srcIndex * Float.BYTES, dst, dstIndex * Float.BYTES,
                        (long) pleDim * Float.BYTES);
            }
            layers[layer] = layerTensor;
        }
        scaledPle.close();
        return layers;
    }

    static boolean needed(ModelConfigTraits traits, ModelConfig config, ResolvedModelWeights resolvedWeights) {
        if (!DirectForwardElementwisePolicy.shouldBuildPerLayerInputs(
                traits,
                config.getHiddenSizePerLayerInput())) {
            return false;
        }
        return resolvedWeights.packedPleEmbeddings() != null
                && resolvedWeights.pleProjection() != null
                && resolvedWeights.pleProjectionNorm() != null;
    }

    static void applyResidual(DirectForwardPerLayerResidualRequest request) {
        if (request == null) {
            return;
        }
        AccelTensor perLayerInput = request.perLayerInput();
        if (perLayerInput == null) {
            return;
        }

        DirectForwardPerLayerResidualWeights weights = request.weights();
        if (weights == null || !weights.complete()) {
            System.err.printf("[DEBUG] applyResidual returning! weights=%s, complete=%s, gate=%s, proj=%s, norm=%s%n",
                    weights != null ? "PRESENT" : "NULL",
                    weights != null && weights.complete(),
                    weights != null && weights.gateWeight() != null,
                    weights != null && weights.projectionWeight() != null,
                    weights != null && weights.normWeight() != null);
            return;
        }

        AccelTensor hidden = AccelTensor.view(request.hiddenSeg(), request.hiddenShape());
        AccelTensor gate = request.operators().linear(
                hidden, weights.gateWeight(), null, "per_layer_gate", request.config());
        hidden.close();

        AccelTensor mixed = activationMultiply(gate, perLayerInput, request);
        gate.close();

        AccelTensor projected = request.operators().linear(
                mixed, weights.projectionWeight(), null, "per_layer_projection", request.config());
        mixed.close();

        AccelTensor normed;
        if (request.useMetalElementwise() && request.workspace() != null) {
            normed = AccelTensor.view(request.workspace().getNormedFfnSeg(), request.hiddenShape());
            DirectForwardElementwiseOps.rmsNormRowsMetal(
                    request.runtime().metalBinding(),
                    normed.dataPtr(),
                    projected.dataPtr(),
                    weights.normWeight().dataPtr(),
                    request.seqLen(),
                    request.config().getHiddenSize(),
                    (float) request.config().getRmsNormEps(),
                    request.addOneRmsNorm());
        } else {
            normed = AccelOps.rmsNorm(projected, weights.normWeight(), request.config().getRmsNormEps(),
                    request.addOneRmsNorm());
        }
        projected.close();

        DirectForwardElementwiseOps.residualAdd(
                request.runtime().log(),
                request.runtime().metalBinding(),
                request.hiddenSeg(),
                normed,
                request.hiddenSeg(),
                request.seqLen(),
                request.config().getHiddenSize(),
                request.useNativeElementwiseAdd());
        normed.close();
    }

    static void close(AccelTensor[] perLayerInputs) {
        if (perLayerInputs == null) {
            return;
        }
        for (AccelTensor perLayerInput : perLayerInputs) {
            if (perLayerInput != null) {
                perLayerInput.closeWithParent();
            }
        }
    }

    private static AccelTensor activationMultiply(AccelTensor gate, AccelTensor perLayerInput,
            DirectForwardPerLayerResidualRequest request) {
        ModelArchitecture arch = request.arch();
        DirectForwardRuntimeContext runtime = request.runtime();
        long[] shape = gate.shape();
        int elements = Math.toIntExact(gate.numel());
        if (request.useMetalElementwise() && request.workspace() != null && runtime.metalBinding() != null) {
            AccelTensor mixed = DirectForwardTensorOps.reusableWorkspaceView(request.workspace().getCombinedSeg(), shape);
            if (mixed == null) {
                mixed = AccelTensor.zeros(shape);
            }
            try {
                int rc = arch.activationType() == FFNActivationType.GELU
                        ? runtime.metalBinding().geluFfn(
                                mixed.dataPtr(), gate.dataPtr(), perLayerInput.dataPtr(), elements)
                        : runtime.metalBinding().siluFfn(
                                mixed.dataPtr(), gate.dataPtr(), perLayerInput.dataPtr(), elements);
                if (rc == 0) {
                    return mixed;
                }
            } catch (IllegalStateException | UnsupportedOperationException e) {
                runtime.log().debugf("Falling back from Metal per-layer activation multiply: %s", e.getMessage());
            }
            mixed.close();
        }

        AccelTensor activated = arch.activationType() == FFNActivationType.GELU
                ? AccelOps.gelu(gate)
                : AccelOps.silu(gate);
        try {
            return AccelOps.mul(activated, perLayerInput);
        } finally {
            activated.close();
        }
    }
}
