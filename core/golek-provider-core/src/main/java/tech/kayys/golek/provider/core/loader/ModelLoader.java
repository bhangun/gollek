package tech.kayys.golek.provider.core.loader;

import io.smallrye.mutiny.Uni;

import java.nio.file.Path;

/**
 * Interface for loading model artifacts
 */
public interface ModelLoader {

    /**
     * Load model by ID
     * 
     * @param modelId Model identifier
     * @return Path to loaded model
     */
    Uni<Path> load(String modelId);

    /**
     * Check if model is already loaded
     */
    boolean isLoaded(String modelId);

    /**
     * Unload model and free resources
     */
    Uni<Void> unload(String modelId);

    /**
     * Get model path if loaded
     */
    Path getPath(String modelId);
}