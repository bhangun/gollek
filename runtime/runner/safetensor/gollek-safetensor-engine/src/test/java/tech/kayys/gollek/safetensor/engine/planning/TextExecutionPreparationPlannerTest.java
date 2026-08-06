/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendCapabilities;
import tech.kayys.gollek.safetensor.engine.session.ConversationExecutionState;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TextExecutionPreparationPlannerTest {

    @Test
    void contextUsesRuntimeTraitForMultimodalPlanning() {
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .multimodalModel(true)
                .build();
        PreparedTextModel model = new PreparedTextModel(
                "direct",
                Path.of("/tmp/plain-causal-model"),
                new StubLoadedModel(new ModelConfig(), traits),
                TextExecutionBackendCapabilities.directReference());

        TextExecutionPreparationContext context = new TextExecutionPreparationPlanner()
                .contextFor(model, ConversationExecutionState.noneRequested());

        assertTrue(context.multimodalArchitecture());
    }

    private record StubLoadedModel(ModelConfig config, ModelRuntimeTraits runtimeTraits)
            implements SafetensorEngine.LoadedModel {

        @Override
        public Path path() {
            return Path.of("/tmp/plain-causal-model");
        }

        @Override
        public Map<String, ?> weights() {
            return Map.of();
        }

        @Override
        public Tokenizer tokenizer() {
            return null;
        }

        @Override
        public String key() {
            return "plain-causal-model";
        }

        @Override
        public boolean isQuantized() {
            return false;
        }
    }
}
