/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextGeneration;
import tech.kayys.gollek.safetensor.engine.backend.PreparedTextModel;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackend;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendCapabilities;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelection;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelector;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionPreparationPlan;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionManager;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionRegistry;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionReuseDecision;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InferenceRequestPlannerWiringTest {

    @TempDir
    Path tempDir;

    @Test
    void prepareWiresExtractedPlanningCollaboratorsThroughBackend() throws Exception {
        Path modelPath = Files.createDirectory(tempDir.resolve("wired-model"));
        ModelRuntimeTraits runtimeTraits = ModelRuntimeTraits.builder()
                .qwenText()
                .audioModel()
                .build();
        StubLoadedModel loadedModel = new StubLoadedModel(modelPath, modelConfig(), runtimeTraits);
        CapturingBackend backend = new CapturingBackend(loadedModel);
        InferenceRequestPlanner planner = new InferenceRequestPlanner(
                null,
                config(tempDir),
                new FixedBackendSelector(backend),
                new TextExecutionSessionManager(new TextExecutionSessionRegistry()),
                new TextExecutionPreparationPlanner(),
                new tech.kayys.gollek.safetensor.engine.session.ConversationExecutionStateResolver());

        ProviderRequest request = ProviderRequest.builder()
                .model("fallback-model-id")
                .message(Message.user("first"))
                .message(Message.assistant("ack"))
                .message(Message.user("second"))
                .parameter("model_path", "wired-model")
                .parameter("quantize_strategy", " int4 ")
                .parameter("kv_cache_quant", "int8")
                .temperature(0.5)
                .topK(0)
                .topP(0.8)
                .maxTokens(32)
                .repeatPenalty(1.2)
                .build();

        PreparedInferenceRequest prepared = planner.prepare(request);

        assertEquals(modelPath, prepared.modelPath());
        assertEquals(modelPath, backend.preparedModelPath);
        assertEquals(QuantizationEngine.QuantStrategy.INT4, backend.preparedQuantStrategy);
        assertTrue(prepared.audioModel());
        assertEquals("second", prepared.ttsPrompt());
        assertSame(backend.preparedGeneration, prepared.preparedGeneration());

        PreparedPrompt prompt = prepared.preparedPrompt();
        assertTrue(prompt.defaultSystemInjected());
        assertTrue(prompt.formattedPrompt().startsWith("<|im_start|>system\nYou are Qwen"));
        assertEquals("custom", prompt.getModelType());

        GenerationConfig generationConfig = backend.preparedGeneration.generationConfig();
        assertEquals(32, generationConfig.maxNewTokens());
        assertEquals(GenerationConfig.SamplingStrategy.TOP_P, generationConfig.strategy());
        assertEquals(GenerationConfig.KvCacheQuantization.INT8, generationConfig.kvCacheQuant());
        assertEquals(0.5f, generationConfig.temperature(), 1.0e-6f);
        assertEquals(1.2f, generationConfig.repetitionPenalty(), 1.0e-6f);

        assertEquals("fake", prepared.executionBackendId());
        assertEquals("custom", prepared.preparationContext().modelFamily());
        assertEquals("CustomForCausalLM", prepared.preparationContext().primaryArchitecture());
        assertFalse(prepared.preparationContext().hasConversationSession());
        assertEquals(TextExecutionSessionReuseDecision.Status.NOT_REUSABLE,
                prepared.sessionReuseDecision().status());
    }

    private static ModelConfig modelConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "custom",
                  "architectures": ["CustomForCausalLM"]
                }
                """, ModelConfig.class);
    }

    private static SafetensorProviderConfig config(Path basePath) {
        return new SafetensorProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

            @Override
            public String basePath() {
                return basePath.toString();
            }

            @Override
            public String extensions() {
                return ".safetensors,.safetensor";
            }

            @Override
            public String backend() {
                return "direct";
            }

            @Override
            public String ggufOutputDir() {
                return basePath.resolve("gguf").toString();
            }
        };
    }

    private static final class FixedBackendSelector extends TextExecutionBackendSelector {
        private final TextExecutionBackend backend;

        private FixedBackendSelector(TextExecutionBackend backend) {
            this.backend = backend;
        }

        @Override
        public TextExecutionBackendSelection select() {
            return new TextExecutionBackendSelection("fake", backend, null);
        }
    }

    private static final class CapturingBackend implements TextExecutionBackend {
        private final SafetensorEngine.LoadedModel loadedModel;
        private Path preparedModelPath;
        private QuantizationEngine.QuantStrategy preparedQuantStrategy;
        private PreparedTextGeneration preparedGeneration;

        private CapturingBackend(SafetensorEngine.LoadedModel loadedModel) {
            this.loadedModel = loadedModel;
        }

        @Override
        public String id() {
            return "fake";
        }

        @Override
        public TextExecutionBackendCapabilities capabilities() {
            return new TextExecutionBackendCapabilities(
                    true,
                    true,
                    true,
                    false,
                    false,
                    false,
                    false);
        }

        @Override
        public PreparedTextModel prepareModel(
                Path modelPath,
                Path adapterPath,
                QuantizationEngine.QuantStrategy quantStrategy) {
            preparedModelPath = modelPath;
            preparedQuantStrategy = quantStrategy;
            return new PreparedTextModel(id(), modelPath, loadedModel, capabilities());
        }

        @Override
        public PreparedTextGeneration prepareGeneration(
                PreparedPrompt prompt,
                PreparedTextModel model,
                TextExecutionPreparationPlan preparationPlan,
                GenerationConfig cfg) {
            preparedGeneration = TextExecutionBackend.super.prepareGeneration(prompt, model, preparationPlan, cfg);
            return preparedGeneration;
        }

        @Override
        public Uni<InferenceResponse> generate(PreparedTextGeneration generation) {
            return Uni.createFrom().failure(new UnsupportedOperationException("not used in planner test"));
        }

        @Override
        public Multi<InferenceResponse> generateStream(PreparedTextGeneration generation) {
            return Multi.createFrom().empty();
        }

        @Override
        public void unloadModel(Path modelPath) {
        }
    }

    private record StubLoadedModel(
            Path path,
            ModelConfig config,
            ModelRuntimeTraits runtimeTraits)
            implements SafetensorEngine.LoadedModel {

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
            return "stub-model";
        }

        @Override
        public boolean isQuantized() {
            return false;
        }
    }
}
