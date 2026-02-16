package tech.kayys.golek.engine.model;

import java.util.Map;

import tech.kayys.golek.spi.context.RequestContext;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.RunnerMetadata;
import tech.kayys.golek.model.exception.ModelLoadException;

/**
 * Provider interface for discovering and creating model runners.
 * Separates discovery metadata from runner lifecycle.
 */
public interface ModelRunnerProvider {

    /**
     * Get runner metadata for discovery and selection
     */
    RunnerMetadata metadata();

    /**
     * Create a new runner instance
     * 
     * @param manifest       Model metadata and artifact locations
     * @param config         Runner-specific configuration
     * @param requestContext Current tenant context for isolation
     * @return Initialized runner instance
     * @throws ModelLoadException if initialization fails
     */
    ModelRunner create(
            ModelManifest manifest,
            Map<String, Object> config,
            RequestContext requestContext) throws ModelLoadException;
}
