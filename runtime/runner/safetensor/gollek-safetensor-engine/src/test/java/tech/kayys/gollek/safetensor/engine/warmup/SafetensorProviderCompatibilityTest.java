/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.exception.ProviderException;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorProviderCompatibilityTest {

    @TempDir
    Path tempDir;

    @Test
    void rejectsVisionModelFromConfigRatherThanPathName() throws Exception {
        Path modelDir = modelDir("plain-name", """
                {"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"]}
                """);

        ProviderException failure = validate(modelDir.resolve("model.safetensors"));

        assertNotNull(failure);
        assertTrue(failure.getMessage().contains("vision/VLM architecture"));
    }

    @Test
    void allowsAudioModelThroughVisionCompatibilityGate() throws Exception {
        Path modelDir = modelDir("audio-name", """
                {"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}
                """);

        assertNull(validate(modelDir.resolve("model.safetensors")));
    }

    @Test
    void pathNameWithoutConfigReportsMissingConfigInsteadOfVisionGuess() throws Exception {
        Path modelDir = tempDir.resolve("vision-friendly-name");
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("model.safetensors"), "");

        ProviderException failure = validate(modelDir.resolve("model.safetensors"));

        assertNotNull(failure);
        assertTrue(failure.getMessage().contains("missing config.json"));
    }

    private Path modelDir(String name, String configJson) throws Exception {
        Path modelDir = tempDir.resolve(name);
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("config.json"), configJson);
        Files.writeString(modelDir.resolve("model.safetensors"), "");
        return modelDir;
    }

    private ProviderException validate(Path modelPath) throws Exception {
        Method method = SafetensorProvider.class.getDeclaredMethod("validateDirectModelCompatibility", Path.class);
        method.setAccessible(true);
        return (ProviderException) method.invoke(new SafetensorProvider(), modelPath);
    }
}
