package tech.kayys.golek.runtime;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.QuarkusApplication;
import io.quarkus.runtime.annotations.QuarkusMain;

import jakarta.inject.Inject;
import tech.kayys.golek.core.inference.InferenceEngine;
import tech.kayys.golek.engine.plugin.GolekPluginRegistry;

/**
 * Portable inference engine for standalone agents.
 * Minimal footprint with file-based configuration.
 */
@QuarkusMain
public class PortableInferenceEngine implements QuarkusApplication {

    @Inject
    InferenceEngine engine;

    @Inject
    GolekPluginRegistry pluginRegistry;

    // @Inject
    // LocalModelLoader modelLoader;

    public static void main(String... args) {
        Quarkus.run(PortableInferenceEngine.class, args);
    }

    @Override
    public int run(String... args) throws Exception {
        // Load bundled models
        // modelLoader.loadLocalModels();

        // Print available models
        System.out.println("Available models:");
        // modelLoader.getModels().forEach(model -> System.out.println(" - " +
        // model.modelId()));

        // Print loaded plugins
        System.out.println("Loaded plugins:");
        pluginRegistry.all().forEach(plugin -> System.out.println("  - " + plugin.id()));

        // Start REST API (optional)
        System.out.println("Portable inference engine started");
        Quarkus.waitForExit();
        return 0;
    }
}