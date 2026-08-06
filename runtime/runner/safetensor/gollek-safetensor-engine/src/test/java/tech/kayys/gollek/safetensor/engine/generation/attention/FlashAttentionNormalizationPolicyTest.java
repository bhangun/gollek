/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.mapper.GgufMetadataMapper;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionNormalizationPolicyTest {
    private static final String DISABLE_GEMMA4_QK_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_qk_norm";
    private static final String DISABLE_GEMMA4_V_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_v_norm";

    @Test
    void usesQueryPreAttentionScaleForClassicModels() {
        ModelConfig config = new ModelConfig();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);

        FlashAttentionNormalizationPolicy policy =
                FlashAttentionNormalizationPolicy.resolve(null, config, modelPolicy);

        assertEquals((float) (1.0 / Math.sqrt(config.getQueryPreAttnScalar())),
                policy.attentionScale());
    }

    @Test
    void usesUnitAttentionScaleForGemma4Text() {
        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(Map.of("general.architecture", "gemma4"));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);

        FlashAttentionNormalizationPolicy policy =
                FlashAttentionNormalizationPolicy.resolve(null, config, modelPolicy);

        assertEquals(1.0f, policy.attentionScale());
    }

    @Test
    void disablesAddOneRmsNormForGemma4TextEvenWhenArchitectureStoresOffsetWeights() {
        ModelArchitecture architecture = architectureAddingOne();
        ModelConfig classicConfig = new ModelConfig();
        FlashAttentionNormalizationPolicy classicPolicy = FlashAttentionNormalizationPolicy.resolve(
                architecture, classicConfig, FlashAttentionModelPolicy.resolve(architecture, classicConfig));
        ModelConfig gemma4Config = new GgufMetadataMapper().fromGgufMetadata(Map.of("general.architecture", "gemma4"));
        FlashAttentionNormalizationPolicy gemma4Policy = FlashAttentionNormalizationPolicy.resolve(
                architecture, gemma4Config, FlashAttentionModelPolicy.resolve(architecture, gemma4Config));

        assertTrue(classicPolicy.addOneToRmsNormWeight());
        assertFalse(gemma4Policy.addOneToRmsNormWeight());
    }

    @Test
    void gemma4QkAndValueNormsHonorInjectedDisableOptions() {
        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(Map.of("general.architecture", "gemma4"));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);
        FlashAttentionNormalizationPolicy defaultPolicy =
                FlashAttentionNormalizationPolicy.resolve(null, config, modelPolicy,
                        FlashAttentionNormalizationOptions.defaults());

        assertTrue(defaultPolicy.qkNormEnabled());
        assertTrue(defaultPolicy.valueNormEnabled());

        FlashAttentionNormalizationPolicy disabledPolicy =
                FlashAttentionNormalizationPolicy.resolve(null, config, modelPolicy,
                        new FlashAttentionNormalizationOptions(true, true));

        assertFalse(disabledPolicy.qkNormEnabled());
        assertFalse(disabledPolicy.valueNormEnabled());
    }

    @Test
    void classicModelsKeepQkNormAndDisableValueNormEvenWhenGemma4OptionsAreDisabled() {
        ModelConfig config = new ModelConfig();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);

        FlashAttentionNormalizationPolicy policy =
                FlashAttentionNormalizationPolicy.resolve(null, config, modelPolicy,
                        new FlashAttentionNormalizationOptions(true, true));

        assertTrue(policy.qkNormEnabled());
        assertFalse(policy.valueNormEnabled());
    }

    @Test
    void systemPropertiesControlRawGemma4DisableRequests() {
        String previousQkNorm = System.getProperty(DISABLE_GEMMA4_QK_NORM_PROPERTY);
        String previousValueNorm = System.getProperty(DISABLE_GEMMA4_V_NORM_PROPERTY);
        try {
            System.setProperty(DISABLE_GEMMA4_QK_NORM_PROPERTY, "true");
            System.setProperty(DISABLE_GEMMA4_V_NORM_PROPERTY, "true");

            FlashAttentionNormalizationOptions options = FlashAttentionNormalizationOptions.fromSystemProperties();

            assertTrue(options.disableGemma4QkNorm());
            assertTrue(options.disableGemma4ValueNorm());
        } finally {
            restoreProperty(DISABLE_GEMMA4_QK_NORM_PROPERTY, previousQkNorm);
            restoreProperty(DISABLE_GEMMA4_V_NORM_PROPERTY, previousValueNorm);
        }
    }

    private static ModelArchitecture architectureAddingOne() {
        return (ModelArchitecture) Proxy.newProxyInstance(
                FlashAttentionNormalizationPolicyTest.class.getClassLoader(),
                new Class<?>[] { ModelArchitecture.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "addOneToRmsNormWeight" -> true;
                    case "runtimeTraits" -> ModelRuntimeTraits.fallbackFromConfig((ModelConfig) args[0]);
                    case "supportedArchClassNames", "supportedModelTypes" -> List.of();
                    case "toString" -> "architectureAddingOne";
                    default -> defaultValue(method.getReturnType());
                });
    }

    private static Object defaultValue(Class<?> returnType) {
        if (returnType == boolean.class) {
            return false;
        }
        if (returnType == int.class) {
            return 0;
        }
        if (returnType == float.class) {
            return 0.0f;
        }
        if (returnType == double.class) {
            return 0.0;
        }
        return null;
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
