package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the lightweight direct SafeTensor profile used before model weights are loaded.
 */
class DirectSafetensorRunProfileTest {

    @TempDir
    Path tempDir;

    @Test
    void gemma4UnifiedProfileUsesDedicatedRuntimeTraitsForPerLayerInputPolicy() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "vocab_size_per_layer_input": 262144
                  },
                  "vision_config": {"model_type": "gemma4_vision"},
                  "audio_config": {"model_type": "gemma4_audio"}
                }
                """);

        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(tempDir);

        assertTrue(profile.gemma4Unified());
        assertTrue(profile.nativeBf16Matvec());
        assertFalse(profile.runtimeTraits().perLayerInputPath());
        assertTrue(profile.runtimeTraits().skipDefaultSystemPromptInjection());
        assertTrue(profile.runtimeTraits().visionModel());
        assertTrue(profile.runtimeTraits().audioModel());
        assertTrue(profile.runtimeTraits().multimodalModel());
    }

    @Test
    void gemma4UnifiedProfileEnablesPerLayerInputOnlyWhenHiddenSizeIsExecutable() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size_per_layer_input": 4096
                  }
                }
                """);

        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(tempDir);

        assertTrue(profile.nativeBf16Matvec());
        assertTrue(profile.runtimeTraits().perLayerInputPath());
    }

    @Test
    void gemma4ImageTextAliasUsesUnifiedProfileAndDetectedVisionTraits() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4",
                  "architectures": ["Gemma4ForImageTextToText"],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size_per_layer_input": 4096
                  },
                  "vision_config": {"model_type": "gemma4_vision"}
                }
                """);

        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(tempDir);

        assertTrue(profile.gemma4Unified());
        assertTrue(profile.nativeBf16Matvec());
        assertTrue(profile.runtimeTraits().perLayerInputPath());
        assertTrue(profile.runtimeTraits().visionModel());
        assertTrue(profile.runtimeTraits().multimodalModel());
        assertFalse(profile.runtimeTraits().audioModel());
    }

    @Test
    void gemma4RootMoeAliasesAreVisibleToDirectProfileBeforeWeightsLoad() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "enable_moe_block": true,
                  "num_experts": 128,
                  "top_k_experts": 8,
                  "moe_intermediate_size": 704,
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "vocab_size": 32
                  }
                }
                """);

        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(tempDir);

        assertTrue(profile.gemma4Unified());
        assertTrue(profile.config().requiresGemma4PackedMoeRuntime());
        assertEquals(128, profile.config().getNumLocalExperts());
        assertEquals(8, profile.config().getNumExpertsPerTok());
        assertEquals(704, profile.config().getMoeIntermediateSize());
    }
}
