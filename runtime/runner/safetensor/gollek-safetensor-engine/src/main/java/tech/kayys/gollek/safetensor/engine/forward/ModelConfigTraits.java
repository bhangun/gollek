/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

record ModelConfigTraits(
        ModelConfig source,
        String modelType,
        int hiddenSizePerLayerInput,
        int vocabSizePerLayerInput,
        boolean requiresTurnAwarePromptBos,
        boolean preferLargeLogitsMatvec,
        boolean perLayerInputPath,
        // Capability flags — policy classes MUST use these, never model-identity fields above.
        boolean nativeBf16Matvec,
        boolean geluGatedFfn,
        boolean perLayerInputEmbedding) {

    static final ModelConfigTraits EMPTY =
            new ModelConfigTraits(null, "", 0, 0, false, false, false, false, false, false);

    /**
     * Returns true for models that use SwiGLU (SILU activation) with gated FFNs.
     *
     * <p>This lets the inference engine route models like Qwen and Granite to the native
     * Metal SwiGLU matvec FFN fast path dynamically, without needing hardcoded model checks.
     */
    boolean siluGated() {
        return source != null && "silu".equalsIgnoreCase(source.getHiddenAct());
    }

    static ModelConfigTraits create(ModelConfig config) {
        return create(config, null);
    }

    static ModelConfigTraits create(ModelConfig config, ModelArchitecture arch) {
        String modelType = config.getModelType() == null ? "" : config.getModelType();
        int hiddenSizePerLayerInput = config.getHiddenSizePerLayerInput();
        int vocabSizePerLayerInput = config.getVocabSizePerLayerInput();
        ModelRuntimeTraits runtimeTraits = ModelRuntimeTraitsResolver.resolve(arch, config);
        boolean perLayerInputPath = runtimeTraits.perLayerInputPath() || hiddenSizePerLayerInput > 0;
        return new ModelConfigTraits(
                config,
                modelType,
                hiddenSizePerLayerInput,
                vocabSizePerLayerInput,
                runtimeTraits.promptBosPolicy() == tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE,
                runtimeTraits.attention().largeAttentionMatvecCandidate(),
                perLayerInputPath,
                runtimeTraits.nativeBf16Matvec(),
                runtimeTraits.geluGatedFfn(),
                runtimeTraits.perLayerInputEmbedding());
    }

    boolean matches(ModelConfig config) {
        return source == config
                && modelType.equals(config.getModelType() == null ? "" : config.getModelType())
                && hiddenSizePerLayerInput == config.getHiddenSizePerLayerInput()
                && vocabSizePerLayerInput == config.getVocabSizePerLayerInput();
    }
}
