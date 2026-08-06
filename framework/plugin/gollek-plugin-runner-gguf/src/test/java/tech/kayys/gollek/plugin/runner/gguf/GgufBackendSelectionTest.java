package tech.kayys.gollek.plugin.runner.gguf;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.runner.ModelLoadRequest;
import tech.kayys.gollek.plugin.runner.RunnerContext;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.Map;
import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufBackendSelectionTest {
    @Test
    void defaultsToLlamaCppForProductionGeneration() {
        GgufBackendSelection selection = resolve(Map.of(), RunnerContext.empty());

        assertEquals(GgufBackendSelection.Backend.LLAMACPP, selection.backend());
        assertEquals("auto", selection.normalizedValue());
        assertFalse(selection.explicit());
    }

    @Test
    void allowsExplicitJavaNativeDiagnostics() {
        GgufBackendSelection selection = resolve(Map.of("gguf.backend", "java-native"), RunnerContext.empty());

        assertEquals(GgufBackendSelection.Backend.JAVA, selection.backend());
        assertTrue(selection.explicit());
    }

    @Test
    void readsRunnerManagerParametersNotOnlyMetadata() {
        RunnerContext context = RunnerContext.withParameters(Map.of("gguf.backend", "llama.cpp"));
        GgufBackendSelection selection = resolve(Map.of(), context);

        assertEquals(GgufBackendSelection.Backend.LLAMACPP, selection.backend());
        assertEquals("context.parameter.gguf.backend", selection.source());
        assertTrue(selection.explicit());
    }

    @Test
    void invalidBackendFallsBackToLlamaCppNotJava() {
        GgufBackendSelection selection = resolve(Map.of("gguf.backend", "fastest-please"), RunnerContext.empty());

        assertEquals(GgufBackendSelection.Backend.LLAMACPP, selection.backend());
        assertEquals("fastest-please", selection.requestedValue());
        assertFalse(selection.explicit());
    }

    @Test
    void serviceLoaderEntryResolvesToCompiledPlugin() {
        boolean found = ServiceLoader.load(RunnerPlugin.class).stream()
                .anyMatch(provider -> provider.type().equals(GgufRunnerPlugin.class));

        assertTrue(found);
    }

    @Test
    void runnerPluginManagerDiscoversGgufPlugin() {
        assertTrue(RunnerPluginManager.getInstance().getPlugin(GgufRunnerPlugin.ID).isPresent());
    }

    private static GgufBackendSelection resolve(Map<String, Object> metadata, RunnerContext context) {
        ModelLoadRequest request = ModelLoadRequest.builder()
                .modelPath("/tmp/model.gguf")
                .metadata(metadata)
                .build();
        return GgufBackendSelection.resolve(request, context);
    }
}
