package tech.kayys.golek.engine.model;

import java.util.Map;

import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.model.RunnerMetadata;
import tech.kayys.golek.model.exception.ModelLoadException;
import tech.kayys.wayang.tenant.TenantContext;

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
     * @param manifest      Model metadata and artifact locations
     * @param config        Runner-specific configuration
     * @param tenantContext Current tenant context for isolation
     * @return Initialized runner instance
     * @throws ModelLoadException if initialization fails
     */
    ModelRunner create(
            ModelManifest manifest,
            Map<String, Object> config,
            TenantContext tenantContext) throws ModelLoadException;
}
