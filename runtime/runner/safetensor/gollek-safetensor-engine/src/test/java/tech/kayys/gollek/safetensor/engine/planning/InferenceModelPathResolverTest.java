/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class InferenceModelPathResolverTest {
    private final InferenceModelPathResolver resolver = new InferenceModelPathResolver();

    @TempDir
    Path tempDir;

    @Test
    void requestParameterModelPathWinsOverModelId() throws IOException {
        Path parameterModel = Files.createDirectory(tempDir.resolve("from-param"));
        ProviderRequest request = request("fallback-model")
                .parameter("model_path", "from-param")
                .build();

        Path resolved = resolver.resolve(request, config(tempDir));

        assertEquals(parameterModel, resolved);
    }

    @Test
    void metadataModelPathIsUsedWhenParameterIsMissing() throws IOException {
        Path metadataModel = Files.createDirectory(tempDir.resolve("from-meta"));
        ProviderRequest request = request("fallback-model")
                .metadata("model_path", "from-meta")
                .build();

        Path resolved = resolver.resolve(request, config(tempDir));

        assertEquals(metadataModel, resolved);
    }

    @Test
    void existingAbsoluteModelPathIsReturnedDirectly() throws IOException {
        Path absoluteModel = Files.createDirectory(tempDir.resolve("absolute-model"));
        ProviderRequest request = request("fallback-model")
                .parameter("model_path", absoluteModel.toString())
                .build();

        Path resolved = resolver.resolve(request, config(tempDir));

        assertEquals(absoluteModel, resolved);
    }

    @Test
    void missingRelativeModelFallsBackToRawModelReference() {
        ProviderRequest request = request("missing-model").build();

        Path resolved = resolver.resolve(request, config(tempDir));

        assertEquals(Path.of("missing-model"), resolved);
    }

    @Test
    void blankModelReferenceFailsFast() {
        ProviderRequest request = request(" ").build();

        assertThrows(ProviderException.class, () -> resolver.resolve(request, config(tempDir)));
    }

    private static ProviderRequest.Builder request(String model) {
        return ProviderRequest.builder()
                .model(model)
                .message(Message.user("hello"));
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
}
