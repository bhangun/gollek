/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;

import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Lightweight direct SafeTensor run profile derived before the full model payload is loaded.
 *
 * <p>The CLI uses this profile for routing, prompt formatting, and preflight guards, so
 * family-specific runtime policy should be applied here instead of falling back to broad
 * config-name heuristics when the family module is available.</p>
 */
public record DirectSafetensorRunProfile(
        ModelConfig config,
        String modelType,
        ModelRuntimeTraits runtimeTraits) {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public DirectSafetensorRunProfile {
        modelType = modelType == null ? "" : modelType;
        runtimeTraits = runtimeTraits == null ? ModelRuntimeTraits.EMPTY : runtimeTraits;
    }

    public static DirectSafetensorRunProfile load(Path modelPath) {
        if (modelPath == null) {
            return unresolved();
        }
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir == null) {
                return unresolved();
            }
            ModelConfig config = ModelConfig.fromDirectory(configDir, OBJECT_MAPPER);
            return new DirectSafetensorRunProfile(
                    config,
                    config.modelType(),
                    runtimeTraits(config));
        } catch (Exception ignored) {
            return unresolved();
        }
    }

    public static DirectSafetensorRunProfile unresolved() {
        return new DirectSafetensorRunProfile(null, "", ModelRuntimeTraits.EMPTY);
    }

    public boolean nativeBf16Matvec() {
        return runtimeTraits.nativeBf16Matvec();
    }

    public boolean unifiedMultimodal() {
        if (config == null) {
            return false;
        }
        String normalizedModelType = normalize(config.modelType());
        String normalizedArchitecture = normalize(config.primaryArchitecture());
        return normalizedModelType.equals("gemma4_unified")
                || normalizedArchitecture.equals("gemma4unifiedforconditionalgeneration")
                || normalizedArchitecture.equals("gemma4formultimodallm")
                || normalizedArchitecture.equals("gemma4forimagetexttotext");
    }
    public boolean supportsAlternateRuntime(String runtimeName) {
        if ("litert".equals(runtimeName)) {
            // Previously tied to gemma3Text. Checking the raw model type
            // to fulfill the alternate runtime support contract.
            return modelType().startsWith("gemma3");
        }
        return false;
    }

    public boolean requiresChatTemplateFormatting() {
        return config != null && config.modelType() != null && config.modelType().contains("qwen");
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }
}
